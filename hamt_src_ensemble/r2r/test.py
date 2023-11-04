import sys
sys.path.append("..")
sys.path.append(".")
import time;
import torch;
import os;
import json;
from utils.misc import set_random_seed
from collections import defaultdict
from collections import OrderedDict
from models.vlnbert_init import get_tokenizer
from r2r.data_utils import ImageFeaturesDB, construct_instrs
from r2r.agent_cmt import Seq2SeqCMTAgent
from r2r.env import R2RBatch
from utils1 import read_img_features
import utils1
from eval import Evaluation
from vlnbert.vlnbert_init import get_tokenizer1
from env1 import R2RBatch1
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results
from parser import args;
from env import EnvBatch

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
PLACE365_FEATURES = '/root/mount/Matterport3DSimulator/vln_data/ResNet-152-places365.tsv'
result_path = "/root/mount/Matterport3DSimulator/VLN-speaker/result/"
if args.features1 == 'imagenet':
    features = IMAGENET_FEATURES
elif args.features1 == 'places365':
    features = PLACE365_FEATURES
    
def build_dataset(rank=0, is_test=False):
    # val_env_names = ['val_seen', 'val_unseen']
    val_env_names = ['val_unseen']
    
    #hamt：生成式的模型；
    tok = get_tokenizer(args)
    feat_db_val = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
    feat_db_train = ImageFeaturesDB(args.img_ft_file, args.image_feat_size, args.img_aug_ft_file)
    dataset_class = R2RBatch
    train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, ['val_seen'], tokenizer=tok, max_instr_len=args.max_instr_len
    )
    train_env = dataset_class(
        train_instr_data, args.connectivity_dir, batch_size=args.batch_size,
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        sel_data_idxs=None, name='train'
    )
    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split], tokenizer=tok, max_instr_len=args.max_instr_len
        )
        val_env = dataset_class(
            val_instr_data, args.connectivity_dir, batch_size=args.batch_size,
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split
        )
        val_envs[split] = val_env
    
    #vln_trans:基于lstm的模型；
    tok1 = get_tokenizer1(args)
    feat_dict = read_img_features(features,test_only=False)
    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
    val_envs1 = OrderedDict(
        (
            (split,R2RBatch1(batch_size=args.batch_size, splits=[split], tokenizer=tok1)) for split in val_env_names
        )
    )
    Env=EnvBatch(args.connectivity_dir,feature_store=feat_dict,feat_db=feat_db_val,batch_size=args.batch_size)
    return train_env,val_envs,val_envs1,tok1,Env;

def valid(Env,train_env,val_envs,val_envs1,tok,rank):
    agent = Seq2SeqCMTAgent(args, train_env,train_env,Env,tok, rank=rank)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (agent.load(args.resume_file), args.resume_file))
        print("Loaded the listener model at iter %d from %s" % (agent.load2(args.resume_file1), args.resume_file1))
        agent.load1(args.load1)
        print("load model vln-trans's parameter:%s"%(args.load1))
        
        
    with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
        json.dump(vars(args), outf, indent=4)
    record_file = os.path.join(args.log_dir, 'valid.txt')
    write_to_record_file(str(args) + '\n\n', record_file)

    for env_name, env in val_envs.items():
        print("env_name :",env_name)
        # if os.path.exists(os.path.join(args.pred_dir, "submit_%s.json" % env_name)):
        #     continue
        agent.logs = defaultdict(list)
        agent.env = env
        agent.env1=val_envs1[env_name]
        iters = None
        start_time = time.time()
        agent.test(use_dropout=False, feedback='argmax', iters=iters)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results()
        if 'test' not in env_name:
            score_summary, _ = env.eval_metrics(preds)
            loss_str = "Env name: %s" % env_name
            for metric, val in score_summary.items():
                loss_str += ', %s: %.2f' % (metric, val)
            write_to_record_file(loss_str+'\n', record_file)

        if args.submit:
            json.dump(
                preds,
                open(os.path.join(args.pred_dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )

def main():
    rank = 0
    torch.set_grad_enabled(False)
    set_random_seed(args.seed + rank)
    train_env,val_envs,val_envs1,tok1,Env= build_dataset( rank=rank)
    valid(Env,train_env,val_envs,val_envs1,tok1, rank=rank)

if __name__ == '__main__':
    main()