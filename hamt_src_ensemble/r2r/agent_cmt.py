import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.distributed import is_default_gpu
from utils.misc import length2mask
from utils.logger import print_progress
from utils1 import padding_idx, print_progress
from models.model_HAMT import VLNBertCMT, Critic
import utils1;
from .eval_utils import cal_dtw
import model;
from .agent_base import BaseAgent
import model_PREVALENT
from parser import args;
class Seq2SeqCMTAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
      'left': [[0],[-1], [0]], # left
      'right': [[0], [1], [0]], # right
      'up': [[0], [0], [1]], # up
      'down': [[0], [0],[-1]], # down
      'forward': [[1], [0], [0]], # forward
      '<end>': [[0], [0], [0]], # <end>
      '<start>': [[0], [0], [0]], # <start>
      '<ignore>': [[0], [0], [0]]  # <ignore>
    }
    # for k, v in env_actions.items():
    #     env_actions[k] = [[vx] for vx in v]

    def __init__(self, args, env,env1,Envbatch,tok, rank=0):
        super().__init__(env,env1)
        self.Env=Envbatch;
        self.args = args
        self.tok=tok;
        self.default_gpu = is_default_gpu(self.args)
        self.rank = rank
        self.feature_size = 2048
        # Models
        self._build_model()

        if self.args.world_size > 1:
            self.vln_bert = DDP(self.vln_bert, device_ids=[self.rank], find_unused_parameters=True)
            self.critic = DDP(self.critic, device_ids=[self.rank], find_unused_parameters=True)

        self.models = (self.vln_bert, self.critic)
        self.device = torch.device('cuda:%d'%self.rank) #TODO 

        # Optimizers
        if self.args.optim == 'rms':
            optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            optimizer = torch.optim.Adam
        elif self.args.optim == 'adamW':
            optimizer = torch.optim.AdamW
        elif self.args.optim == 'sgd':
            optimizer = torch.optim.SGD
        else:
            assert False
        if self.default_gpu:
            print('Optimizer: %s' % self.args.optim)
        
        self.vln_bert_optimizer = optimizer(self.vln_bert.parameters(), lr=self.args.lr)
        self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.args.lr)
        self.vln_bert_scheduler=LambdaLR(self.vln_bert_optimizer,lr_lambda=lambda epoch: 1.0/(epoch+1));
        self.critic_scheduler=LambdaLR(self.critic_optimizer, lr_lambda=lambda epoch: 1.0/(epoch+1))
        self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)

        # Evaluations
        self.losses = []
        # self.criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignoreid, size_average=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignoreid, reduction='sum')
        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)
        self.split_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.softmax_loss = nn.CrossEntropyLoss(ignore_index=self.tok.vocab['[PAD]'])
        
    def _build_model(self):
        self.vln_bert = VLNBertCMT(self.args).cuda()
        self.critic = Critic(self.args).cuda()
        self.vln_bert2 = VLNBertCMT(self.args).cuda()
        # self.critic2 = Critic(self.args).cuda()
        self.vln_bert1 = model_PREVALENT.VLNBERT(feature_size=self.feature_size + self.args.angle_feat_size1).cuda()
        # self.critic1 = model_PREVALENT.Critic().cuda()
        self.speaker_encoder = model.SpeakerEncoder(self.feature_size+self.args.angle_feat_size1, self.args.rnn_dim, self.args.dropout1, bidirectional=self.args.bidir).cuda()
        self.speaker_decoder = model.SpeakerDecoder(self.tok.vocab_size, self.args.wemb, self.tok.vocab['[PAD]'],
                                            self.args.rnn_dim, self.args.dropout1).cuda()
        
    def _language_variable(self, obs):
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]
        
        seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
        mask = np.zeros((len(obs), max(seq_lengths)), dtype=bool)
        for i, ob in enumerate(obs):
            seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
            mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor)
        mask = torch.from_numpy(mask)
        return seq_tensor.long().cuda(), mask.cuda(), seq_lengths

    def _cand_pano_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        ob_cand_lens = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        ob_lens = []
        ob_img_fts, ob_ang_fts, ob_nav_types = [], [], []
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            cand_img_fts, cand_ang_fts, cand_nav_types = [], [], []
            cand_pointids = np.zeros((self.args.views, ), dtype=bool)
            for j, cc in enumerate(ob['candidate']):
                cand_img_fts.append(cc['feature'][:self.args.image_feat_size])
                cand_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                # print(cc['pointId'])
                cand_pointids[cc['pointId']] = True
                cand_nav_types.append(1)
            # add [STOP] feature
            cand_img_fts.append(np.zeros((self.args.image_feat_size, ), dtype=np.float32))
            cand_ang_fts.append(np.zeros((self.args.angle_feat_size, ), dtype=np.float32))
            cand_img_fts = np.vstack(cand_img_fts)
            cand_ang_fts = np.vstack(cand_ang_fts)
            cand_nav_types.append(2)
            # print(cand_pointids.sum(),len(cand_nav_types))
            # add pano context
            pano_fts = ob['feature'][~cand_pointids]
            cand_pano_img_fts = np.concatenate([cand_img_fts, pano_fts[:, :self.args.image_feat_size]], 0)
            cand_pano_ang_fts = np.concatenate([cand_ang_fts, pano_fts[:, self.args.image_feat_size:]], 0)
            cand_nav_types.extend([0] * (self.args.views - np.sum(cand_pointids)))
            # print(" " ,len(cand_nav_types),self.args.views)
            ob_lens.append(len(cand_nav_types))
            ob_img_fts.append(cand_pano_img_fts)
            ob_ang_fts.append(cand_pano_ang_fts)
            ob_nav_types.append(cand_nav_types)

        # pad features to max_len
        max_len = max(ob_lens)
        for i in range(len(obs)):
            num_pads = max_len - ob_lens[i]
            ob_img_fts[i] = np.concatenate([ob_img_fts[i], \
                np.zeros((num_pads, ob_img_fts[i].shape[1]), dtype=np.float32)], 0)
            ob_ang_fts[i] = np.concatenate([ob_ang_fts[i], \
                np.zeros((num_pads, ob_ang_fts[i].shape[1]), dtype=np.float32)], 0)
            ob_nav_types[i] = np.array(ob_nav_types[i] + [0] * num_pads)

        ob_img_fts = torch.from_numpy(np.stack(ob_img_fts, 0)).cuda()
        ob_ang_fts = torch.from_numpy(np.stack(ob_ang_fts, 0)).cuda()
        ob_nav_types = torch.from_numpy(np.stack(ob_nav_types, 0)).cuda()
        # print("----")
        # print(ob_img_fts[0].shape)
        # print(ob_img_fts[1].shape)
        # print(ob_lens)
        # print(ob_cand_lens[0])
        # print(ob_cand_lens[1])
        # print("----")
        return ob_img_fts, ob_ang_fts, ob_nav_types, ob_lens, ob_cand_lens,max(ob_cand_lens)

    def _candidate_variable(self, obs):
        cand_lens = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        max_len = max(cand_lens)
        cand_img_feats = np.zeros((len(obs), max_len, self.args.image_feat_size), dtype=np.float32)
        cand_ang_feats = np.zeros((len(obs), max_len, self.args.angle_feat_size), dtype=np.float32)
        cand_nav_types = np.zeros((len(obs), max_len), dtype=np.int64)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                cand_img_feats[i, j] = cc['feature'][:self.args.image_feat_size]
                cand_ang_feats[i, j] = cc['feature'][self.args.image_feat_size:]
                cand_nav_types[i, j] = 1
            cand_nav_types[i, cand_lens[i]-1] = 2

        cand_img_feats = torch.from_numpy(cand_img_feats).cuda()
        cand_ang_feats = torch.from_numpy(cand_ang_feats).cuda()
        cand_nav_types = torch.from_numpy(cand_nav_types).cuda()
        return cand_img_feats, cand_ang_feats, cand_nav_types, cand_lens

    def _history_variable(self, obs):
        hist_img_feats = np.zeros((len(obs), self.args.image_feat_size), np.float32)
        for i, ob in enumerate(obs):  
            hist_img_feats[i] = ob['feature'][ob['viewIndex'], :self.args.image_feat_size]
        hist_img_feats = torch.from_numpy(hist_img_feats).cuda()

        if self.args.hist_enc_pano:
            hist_pano_img_feats = np.zeros((len(obs), self.args.views, self.args.image_feat_size), np.float32)
            hist_pano_ang_feats = np.zeros((len(obs), self.args.views, self.args.angle_feat_size), np.float32)
            for i, ob in enumerate(obs):
                hist_pano_img_feats[i] = ob['feature'][:, :self.args.image_feat_size]
                hist_pano_ang_feats[i] = ob['feature'][:, self.args.image_feat_size:]
            hist_pano_img_feats = torch.from_numpy(hist_pano_img_feats).cuda()
            hist_pano_ang_feats = torch.from_numpy(hist_pano_ang_feats).cuda()
        else:
            hist_pano_img_feats, hist_pano_ang_feats = None, None

        return hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats

    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, obs,perm_obs,perm_idx, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, name):
            if type(name) is int:       # Go to the next view
                self.Env.sims[i].makeAction([name], [0], [0])
            else:                       # Adjust
                self.Env.sims[i].makeAction(*self.env_actions[name])
        # temp=torch.argsort(logit,axis=1)[:,-5:].cpu().numpy().tolist()
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                # print(i," ",len(ob['candidate'])," ",action)
                select_candidate = ob['candidate'][action]
                src_point = ob['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12  # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, 'down')
                    src_level -= 1
                while self.Env.sims[i].getState()[0].viewIndex != trg_point:    # Turn right until the target
                    take_action(i, 'right')
                assert select_candidate['viewpointId'] == \
                       self.Env.sims[i].getState()[0].navigableLocations[select_candidate['idx']].viewpointId
                # print(select_candidate['idx'])
                take_action(i, select_candidate['idx'])

                state = self.Env.sims[i].getState()[0]
                # temp_state=self.env1.env.sims[i].getState()[0]
                # assert state==temp_state,"two R2Rbatch's state is not same"
                # assert state==temp_state,print({k: state[k] for k in state if k in temp_state and state[k] != temp_state[k]},"\n",set(state.keys()) - set(temp_state.keys()))
                if traj is not None:
                    traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))

    def _sort_batch(self, obs):
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        # seq_lengths, perm_idx = seq_lengths.sort(0, True)  # True -> descending
        # sorted_tensor = seq_tensor[perm_idx]
        perm_idx=torch.from_numpy(np.arange(args.batch_size))
        sorted_tensor=seq_tensor
        mask = (sorted_tensor != padding_idx)

        token_type_ids = torch.zeros_like(mask)

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.long().cuda(), token_type_ids.long().cuda(), \
               list(seq_lengths), list(perm_idx)
            
    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), self.args.views, self.feature_size + self.args.angle_feat_size1), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']  # Image feat
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()
    
    def _candidate_variable1(self, obs):
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + self.args.angle_feat_size1), dtype=np.float32)
       
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = cc['feature']

        return torch.from_numpy(candidate_feat).cuda(), candidate_leng
    
    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), self.args.angle_feat_size1), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils1.angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).cuda()
        f_t = self._feature_variable(obs)      # Pano image features from obs
    
        candidate_feat, candidate_leng = self._candidate_variable1(obs) #YZ

        return input_a_t, f_t, candidate_feat, candidate_leng#YZ
    
    def masked_softmax(self, vec, mask, dim=1):
        masked_vec = vec * mask.float()
        max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
        exps = torch.exp(masked_vec-max_vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True)
        zeros=(masked_sums == 0)
        masked_sums += zeros.float()
        return masked_exps/masked_sums
    
    def teacher_forcing(self, train=True, features=None, obs=None, insts=None, for_listener=False, target=None, 
                        split_target=None, split_mask=None):
        (img_feats, can_feats), lengths = features
        
        ctx = self.speaker_encoder(can_feats, img_feats, lengths)
        
        batch_size = len(lengths)

        h_t = torch.zeros(1, batch_size, self.args.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, self.args.rnn_dim).cuda()
        ctx_mask = utils1.length2mask(lengths)

        # Decode
        logits, _, _, out_repr, split_logit = self.speaker_decoder(insts, ctx, ctx_mask, h_t, c_t)
        
        # Because the softmax_loss only allow dim-1 to be logit,
        # So permute the output (batch_size, length, logit) --> (batch_size, logit, length)
        logits = logits.permute(0, 2, 1).contiguous()
        split_logit = split_logit.permute(0, 2, 1).contiguous()

        loss = 0
        if target is not None:
            loss = self.softmax_loss(
                input  = logits[:, :, :-1],         # -1 for aligning
                target = target[:, 1:]               # "1:" to ignore the word <BOS>
            )

        split_loss = 0
        
        split_logit = split_logit.squeeze(dim=1)
        
        if split_target is not None:
            split_loss = self.split_loss(
                input  = split_logit[:, :-1],         # -1 for aligning
                target = split_target[:, 1:]            # "1:" to ignore the word <BOS>
            )
            split_loss *= split_mask[:, 1:]
            split_loss = split_loss.mean()
        

        soft_split = self.masked_softmax(split_logit, split_mask)
        masked_instr = soft_split.unsqueeze(-1)*insts

        if train:
            return loss, split_loss, out_repr, masked_instr
    
    def rollout(self, train_ml=None, train_rl=True, reset=True):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:  # Reset env
            # print("-------------")
            obs = self.env.reset(self.Env)#hamt
            obs1 = np.array(self.env1.reset(self.Env))#vln_trans
            # print("-------------")
        else:
            obs = self.env._get_obs(self.Env,t=0)

        batch_size = len(obs)

        # Language input：hamt模型
        txt_ids, txt_masks, txt_lens = self._language_variable(obs)

        ''' Language BERT '''
        language_inputs = {
            'mode': 'language',
            'txt_ids': txt_ids,
            'txt_masks': txt_masks,
        }
        txt_embeds2=self.vln_bert2(**language_inputs)
        txt_embeds = self.vln_bert(**language_inputs)

        # Language input:vln-trans模型
        sentence, language_attention_mask, token_type_ids, \
            seq_lengths, perm_idx = self._sort_batch(obs1)
        perm_obs = obs1[perm_idx]

        ''' Language BERT '''
        language_inputs = {'mode':        'language',
                        'sentence':       sentence,
                        'attention_mask': language_attention_mask,
                        'lang_mask':      language_attention_mask,
                        'token_type_ids': token_type_ids}
        h_t1, language_features = self.vln_bert1(**language_inputs)
            
        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
        } for ob in obs]

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']
            path_act = [vp[0] for vp in traj[i]['path']]
            last_ndtw[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

        # Initialization the tracking state
        ended = np.array([False] * batch_size)

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.

        # for backtrack
        visited = [set() for _ in range(batch_size)]
        hist_embeds2=[self.vln_bert2('history').expand(batch_size, -1)] 
        hist_embeds = [self.vln_bert('history').expand(batch_size, -1)]  # global embedding
        hist_lens = [1 for _ in range(batch_size)]
        initial_language_feat = language_features.clone()
        tmp_lanuage_mask = language_attention_mask.clone()
        for t in range(self.args.max_action_len):
        #hamt:
            if self.args.ob_type == 'pano':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens,candidate_len = self._cand_pano_feature_variable(obs)
                ob_masks = length2mask(ob_lens).logical_not()
            elif self.args.ob_type == 'cand':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(obs)
                ob_masks = length2mask(ob_cand_lens).logical_not()
            
            ''' Visual BERT '''
            visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,    # history before t step
                'hist_lens': hist_lens,
                'ob_img_feats': ob_img_feats,
                'ob_ang_feats': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
                'ob_masks': ob_masks,
                'return_states': True if self.feedback == 'sample' else False
            }      
            t_outputs = self.vln_bert(**visual_inputs)
            logit = t_outputs[0]
            # print(ob_cand_lens)
            # print(logit)
            visual_inputs1 = {
                'mode': 'visual',
                'txt_embeds': txt_embeds2,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds2,    # history before t step
                'hist_lens': hist_lens,
                'ob_img_feats': ob_img_feats,
                'ob_ang_feats': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
                'ob_masks': ob_masks,
                'return_states': True if self.feedback == 'sample' else False
            }
            t_outputs2 = self.vln_bert2(**visual_inputs1)
            logit2 = t_outputs2[0]
            
        #vln_trans:
            input_a_t, img_feats, candidate_feat, candidate_leng = self.get_input_feat(perm_obs) 

            features = (img_feats, candidate_feat.clone()), candidate_leng
            # split_target = None
            # sub_instr_target = None
            
            split_target = torch.zeros((len(obs), 80), dtype=torch.float32)
            sub_instr_target = []
            for id, ob in enumerate(perm_obs):
                indexes =  torch.tensor(ob['split_target'])
                start_index, end_index = indexes
                split_target[id][start_index+1: end_index+1] = 1
                sub_instr_target.append(torch.tensor(ob['sub_instr_target']))
            split_target = split_target.cuda()
            sub_instr_target = torch.stack(sub_instr_target, dim=0).cuda()
            
            speaker_loss, split_loss, speaker_repr, masked_repr = self.teacher_forcing(train=True, features=features, obs=perm_obs,
                                                                insts=initial_language_feat, target=sub_instr_target,
                                                                split_target=split_target, split_mask=tmp_lanuage_mask)
            # the first [CLS] token, initialized by the language BERT, serves
            # as the agent's state passing through time steps
            if t == 0:
                language_features = torch.cat((language_features, speaker_repr, masked_repr), dim = 1)
            if t >= 1 :
                language_features = torch.cat((h_t1.unsqueeze(1), language_features[:,1:,:]), dim=1)
                language_features[:,80:80*2,:] = speaker_repr
                language_features[:,80*2:, :] = masked_repr
            visual_temp_mask = (utils1.length2mask(candidate_leng) == 0).long()
            language_attention_mask = torch.cat((tmp_lanuage_mask, tmp_lanuage_mask, tmp_lanuage_mask), dim=-1)
            visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1)

            self.vln_bert1.vln_bert.config.directions = max(candidate_leng)
            ''' Visual BERT '''
            visual_inputs = {'mode':              'visual',
                            'sentence':           language_features,
                            'attention_mask':     visual_attention_mask,
                            'lang_mask':          language_attention_mask,
                            'vis_mask':           visual_temp_mask,
                            'token_type_ids':     token_type_ids,
                            'action_feats':       input_a_t,
                            # 'pano_feats':         f_t,
                            'cand_feats':         candidate_feat,
                            }
            h_t1, logit1  = self.vln_bert1(**visual_inputs)
            
            if self.feedback == 'sample':
                h_t = t_outputs[1]
                hidden_states.append(h_t)

            if train_ml is not None:
                # Supervised training 
                target = self._teacher_action(obs, ended)
                ml_loss += self.criterion(logit, target)
            
            # mask logit where the agent backtracks in observation in evaluation
            if self.args.no_cand_backtrack:#default:False
                bt_masks = torch.zeros(ob_nav_types.size()).bool()
                for ob_id, ob in enumerate(obs):
                    visited[ob_id].add(ob['viewpoint'])
                    for c_id, c in enumerate(ob['candidate']):
                        if c['viewpointId'] in visited[ob_id]:
                            bt_masks[ob_id][c_id] = True
                bt_masks = bt_masks.cuda()
                logit.masked_fill_(bt_masks, -float('inf'))

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                 # teacher forcing
            elif self.feedback == 'argmax':
                #使用最普通的集成的方法，直接把两个模型的得分相加；
                assert logit1.shape[1]==candidate_len,"vln-trans's candidate num is not consistent with the num in hamt"
                #logit1,logit2,logit是三个模型的输出，这里是把logit1和logit2叠加到logit上
                #将logit1融合到logit2中;
                # print(logit)
                # import pdb; pdb.set_trace()
                N = 2
                logit1_top5=torch.sort(logit1)[1][:,-N:]
                batchSize=logit1.shape[0]
                addTensor=torch.zeros((batchSize,logit.shape[1])).cuda()
                for i in range(batchSize):
                    if ob_cand_lens[i]<N:
                        num=ob_cand_lens[i]
                    else:
                        num=N;
                    averageScore=logit1[i][logit1_top5[i][-num:]].sum()/num
                    weight=logit1[i][logit1_top5[i][-1]]*0.8
                    addTensor[i][logit1_top5[i][-num:]]+=3
                logit2+=addTensor
                # import pdb; pdb.set_trace()
                # 将logit2融合到logit中；
                logit2_top5=torch.sort(logit2)[1][:,-N:]
                addTensor1=torch.zeros((batchSize,logit.shape[1])).cuda()
                for i in range(batchSize):
                    if ob_cand_lens[i]<N:
                        num=ob_cand_lens[i]
                    else:
                        num=N;
                    averageScore=logit2[i][logit2_top5[i][-num:]].sum()/num
                    weight=logit2[i][logit2_top5[i][-1]]*0.8
                    addTensor1[i][logit2_top5[i][-num:]]+=3 #averageScore*weight
                logit+=addTensor1
                # import pdb; pdb.set_trace()
                #后面对进行叠加后的Logit进行softmax的操作获得有最高得分的场景
                # tmp=logit.shape[1]-logit1.shape[1];
                # logit1=torch.cat([logit1,torch.zeros(logit.shape[0],tmp,dtype=torch.float32).cuda()],1)
                # logit=logit+logit2+logit1
                _, a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (ob_cand_lens[i]-1) or next_id == self.args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # get history input embeddings
            if train_rl or ((not np.logical_or(ended, (cpu_a_t == -1)).all()) and (t != self.args.max_action_len-1)):
                # DDP error: RuntimeError: Expected to mark a variable ready only once.
                # It seems that every output from DDP should be used in order to perform correctly
                hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats = self._history_variable(obs)
                prev_act_angle = np.zeros((batch_size, self.args.angle_feat_size), np.float32)
                for i, next_id in enumerate(cpu_a_t):
                    if next_id != -1:
                        # print(next_id)
                        prev_act_angle[i] = obs[i]['candidate'][next_id]['feature'][-self.args.angle_feat_size:]
                prev_act_angle = torch.from_numpy(prev_act_angle).cuda()

                t_hist_inputs = {
                    'mode': 'history',
                    'hist_img_feats': hist_img_feats,
                    'hist_ang_feats': prev_act_angle,
                    'hist_pano_img_feats': hist_pano_img_feats,
                    'hist_pano_ang_feats': hist_pano_ang_feats,
                    'ob_step': t,
                }
                t_hist_embeds2=self.vln_bert2(**t_hist_inputs)
                t_hist_embeds = self.vln_bert(**t_hist_inputs)
                hist_embeds.append(t_hist_embeds)
                hist_embeds2.append(t_hist_embeds2)
                
                for i, i_ended in enumerate(ended):
                    if not i_ended:
                        hist_lens[i] += 1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, obs, perm_obs,perm_idx,traj)
            obs = self.env._get_obs(self.Env,t=t+1)
            obs1 = np.array(self.env1._get_obs(self.Env))
            perm_obs = obs1[perm_idx]  
            
            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                ndtw_score = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(obs):
                    dist[i] = ob['distance']
                    path_act = [vp[0] for vp in traj[i]['path']]
                    ndtw_score[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        # Target reward
                        if action_idx == -1:                              # If the action now is end
                            if dist[i] < 3.0:                             # Correct
                                reward[i] = 2.0 + ndtw_score[i] * 2.0
                            else:                                         # Incorrect
                                reward[i] = -2.0
                        else:                                             # The action is not end
                            # Path fidelity rewards (distance & nDTW)
                            reward[i] = - (dist[i] - last_dist[i])  # this distance is not normalized
                            ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            if reward[i] > 0.0:                           # Quantification
                                reward[i] = 1.0 + ndtw_reward
                            elif reward[i] < 0.0:
                                reward[i] = -1.0 + ndtw_reward
                            else:
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i]-last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score

            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all endedf
            if ended.all():
                break

        if train_rl:
            if self.args.ob_type == 'pano':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens,candidate_len = self._cand_pano_feature_variable(obs)
                ob_masks = length2mask(ob_lens).logical_not()
            elif self.args.ob_type == 'cand':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(obs)
                ob_masks = length2mask(ob_cand_lens).logical_not()

            ''' Visual BERT '''
            visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,
                'hist_lens': hist_lens,
                'ob_img_feats': ob_img_feats,
                'ob_ang_feats': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
                'ob_masks': ob_masks,
                'return_states': True
            }
            _, last_h_ = self.vln_bert(**visual_inputs)
            
            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()        # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * self.args.gamma + rewards[t]  # If it ended, the reward will be 0
                mask_ = torch.from_numpy(masks[t]).cuda()
                clip_reward = discount_reward.copy()
                r_ = torch.from_numpy(clip_reward).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                t_policy_loss = (-policy_log_probs[t] * a_ * mask_).sum()
                t_critic_loss = (((r_ - v_) ** 2) * mask_).sum() * 0.5 # 1/2 L2 loss

                rl_loss += t_policy_loss + t_critic_loss
                if self.feedback == 'sample':
                    rl_loss += (- self.args.entropy_loss_weight * entropys[t] * mask_).sum()

                self.logs['critic_loss'].append(t_critic_loss.item())
                self.logs['policy_loss'].append(t_policy_loss.item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if self.args.normalize_loss == 'total':
                rl_loss /= total
            elif self.args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert self.args.normalize_loss == 'none'

            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item()) # critic loss + policy loss + entropy loss

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())
        #损失函数有两个部分组成：1.强化学习损失 2.输出的决策向量和真值之间的损失
        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.args.max_action_len)  # This argument is useless.

        return traj
    

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
            self.vln_bert1.eval();
            # self.critic1.eval()
            self.speaker_encoder.eval()
            self.speaker_decoder.eval()
            self.vln_bert2.eval();
            # self.critic2.eval()
        super().test(iters=iters)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=self.args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=self.args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

        self.vln_bert_optimizer.step()
        self.critic_optimizer.step()

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.vln_bert.train()
        self.critic.train()

        self.losses = []
        for iter in range(1, n_iters + 1):

            self.vln_bert_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            self.loss = 0

            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.rollout(train_ml=self.args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample':  # agents in IL and RL separately
                if self.args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(train_ml=self.args.ml_weight, train_rl=False, **kwargs)
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, **kwargs)
            else:
                assert False

            #print(self.rank, iter, self.loss)
            self.loss.backward()

            torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

            self.vln_bert_optimizer.step()
            self.critic_optimizer.step()

            if self.args.aug is None:
                print_progress(iter, n_iters+1, prefix='Progress:', suffix='Complete', bar_length=50)

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            state_dict = states[name]['state_dict']
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
                if not list(model_keys)[0].startswith('module.') and list(load_keys)[0].startswith('module.'):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            state.update(state_dict)
            model.load_state_dict(state)
            if self.args.resume_optimizer:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1
    
    def load2(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            state_dict = states[name]['state_dict']
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
                if not list(model_keys)[0].startswith('module.') and list(load_keys)[0].startswith('module.'):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            state.update(state_dict)
            model.load_state_dict(state)
            if self.args.resume_optimizer:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("vln_bert", self.vln_bert2, self.vln_bert_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1
    
    def load1(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            state_dict = states[name]['state_dict']
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
                if not list(model_keys)[0].startswith('module.') and list(load_keys)[0].startswith('module.'):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            state.update(state_dict)
            model.load_state_dict(state)
            if self.args.resume_optimizer:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("vln_bert", self.vln_bert1, None),
                     ("speaker_encoder", self.speaker_encoder, None),
                     ("speaker_decoder", self.speaker_decoder, None)]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1
