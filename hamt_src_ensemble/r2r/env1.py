''' Batched Room-to-Room navigation environment '''

from curses import delay_output
import sys
sys.path.append('buildpy36')
sys.path.append('Matterport_Simulator/build/')
import MatterSim
import csv
import numpy as np
import math
import base64
import utils1
import json
import os
import random
import networkx as nx
from tqdm import tqdm
from r2r.data_utils import new_simulator
from r2r.data_utils import load_instr_datasets
from utils1 import load_datasets, load_nav_graphs, pad_instr_tokens

csv.field_size_limit(sys.maxsize)
from parser import args;
from env import EnvBatch;

class R2RBatch1():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, batch_size=100, seed=0, splits=['train'], tokenizer=None,
                 name=None,feature_store=None): 
        # self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        
        self.feature_size = 2048
        self.data = []
        if tokenizer:
            self.tok = tokenizer
        scans = []
        for split in splits:
            sub_instr_dict = {}
            for i_item, item in enumerate(tqdm(load_instr_datasets(args.anno_dir,'r2r',[split]))):
                if args.test_only and i_item == 64:
                    break
                if "/" in split:
                    new_item = dict(item)
                    new_item['instr_id'] = item['path_id']
                    new_item['instructions'] = item['instructions']
                    new_item['instr_encoding'] = item['instr_enc']

                    new_item['split_target'] = item['split_target']
                    new_item['sub_instr_target'] = item['sub_instr_target']
                    if new_item['instr_encoding'] is not None:  # Filter the wrong data
                        self.data.append(new_item)
                        scans.append(item['scan'])

                else:
                    # Split multiple instructions into separate entries
                    for j, instr in enumerate(item['instructions']):
                        try:
                            new_item = dict(item)
                            new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                            new_item['instructions'] = instr

                            #BERT tokenizer
                            instr_tokens = tokenizer.tokenize(instr)
                            if len(instr_tokens) <= 2:
                                instr_tokens +=['.']*(3-len(instr_tokens))
                            padded_instr_tokens, num_words = pad_instr_tokens(instr_tokens, args.maxInput)
                            new_item['instr_encoding'] = tokenizer.convert_tokens_to_ids(padded_instr_tokens)

                            #Split target and sub-instr target
                            split_dict = {}
                            sub_instr_dict = {}
                            
                            if 'chunk_view' in new_item:
                                for view_id, each_view in enumerate(new_item['chunk_view'][j]):
                                    start  = each_view[0]-1
                                    end = each_view[1] - 1
                                    split_index = new_item['split_index'][j]
                                    sub_instr = new_item['sub_instr'][j]
                                    for viewpoint in new_item['path'][start:end+1]:
                                        split_dict[viewpoint] = split_index[view_id]
                                        sub_instr_dict[viewpoint] = sub_instr[view_id]
                                new_item['split_target'] = split_dict
                                new_item['sub_instr_target'] = sub_instr_dict
                                assert len(split_dict) == len(new_item['path'])
                           
                            if new_item['instr_encoding'] is not None:  # Filter the wrong data
                                self.data.append(new_item)
                                scans.append(item['scan'])
                        except IndexError:
                            # sometimes there are cases that more than 3 sentences
                            continue
       
        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        # random.seed(self.seed)
        # random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        
        self.sim = new_simulator(args.connectivity_dir)
        self.angle_feature = utils1.get_all_point_angle_feature(self.sim)
        self.buffered_state_dict = {}


        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        print('R2RBatch1 loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            # print("-------------------")
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                # random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                # random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        # if shuffle:
            # random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    self.sim.makeAction([0], [1.0], [1.0])
                else:
                    self.sim.makeAction([0], [1.0], [0])

                state = self.sim.getState()[0]
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = utils1.angle_feature(loc_heading, loc_elevation)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            'scanId':scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1),
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                normalized_heading = c_new['normalized_heading']
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                angle_feat = utils1.angle_feature(c_new['heading'], c_new['elevation'])
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self,env):
        obs = []
        for i, (feature, state) in enumerate(env.getStates1()):
            item = self.batch[i]
            base_view_id = state.viewIndex

            if feature is None:
                feature = np.zeros((36, 2048))

            # Full features
            # print(i,state.scanId, state.location.viewpointId)
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)
            # [visual_feature, angle_feature] for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)

            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instructions' : item['instructions'],
                'teacher' : self._shortest_path_action(state, item['path'][-1]),
                'gt_path' : item['path'],
                'path_id' : item['path_id']
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
                try:
                    obs[-1]['split_target'] = item['split_target'][state.location.viewpointId]
                    obs[-1]['sub_instr_target'] = item['sub_instr_target'][state.location.viewpointId]
                except KeyError:
                    obs[-1]['split_target'] = [0, 79]
                    obs[-1]['sub_instr_target'] = item['instr_encoding']
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return obs

    def reset(self, env,batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        env.newEpisodes(scanIds, viewpointIds, headings)
        # print(scanIds)
        # print(viewpointIds)
        # print(headings)
        return self._get_obs(env)

    def step(self,env, actions):
        ''' Take action (same interface as makeActions) '''
        env.makeActions(actions)
        return self._get_obs(env)

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats