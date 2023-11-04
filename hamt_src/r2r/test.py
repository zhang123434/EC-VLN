import sys
sys.path.append("..")
sys.path.append(".")
from r2r.parser import parse_args
import time;
import torch;
import os;
import json;
from utils.misc import set_random_seed
from collections import defaultdict;
from models.vlnbert_init import get_tokenizer
from r2r.data_utils import ImageFeaturesDB, construct_instrs
from r2r.agent_cmt import Seq2SeqCMTAgent
from r2r.env import R2RBatch

def build_dataset(args, rank=0, is_test=False):
    tok = get_tokenizer(args)
    feat_db_train = ImageFeaturesDB(args.img_ft_file, args.image_feat_size, args.img_aug_ft_file)
    feat_db_val = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
    dataset_class = R2RBatch
    path="../datasets/R2R_test.json"
    val_instr_data = construct_instrs(
        args.anno_dir,args.dataset , [path], tokenizer=tok, max_instr_len=args.max_instr_len
    )
    val_env = dataset_class(
        feat_db_val, val_instr_data, args.connectivity_dir, batch_size=args.batch_size,
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name='test'
    )
    return val_env
def valid(args,val_env, rank=-1):#为啥测试的时候需要train_env???
    agent =Seq2SeqCMTAgent(args,val_env, rank=rank)
    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (agent.load(args.resume_file), args.resume_file))
    agent.logs = defaultdict(list)
    iters = None
    start_time = time.time()
    agent.test(use_dropout=False, feedback='argmax', iters=iters)
    print('test', 'cost time: %.2fs' % (time.time() - start_time))
    preds = agent.get_results()
    # preds = merge_dist_results(all_gather(preds))#单卡训练的时候，这一步可以不执行吧???
    json.dump(
        preds,
        open(os.path.join(args.pred_dir, "submit_test1.json"), 'w'),
        sort_keys=True, indent=4, separators=(',', ': ')
    )

def main():
    args = parse_args()
    rank = 0
    torch.set_grad_enabled(False)#could accelate the forward process
    set_random_seed(args.seed + rank)
    val_env= build_dataset(args, rank=rank)
    valid(args, val_env, rank=rank)
    with open(os.path.join(args.pred_dir, "submit_test1.json"), 'r') as f:
        a=json.load(f)
    with open(os.path.join(args.pred_dir, "submit_test.json"), 'r') as f1:
        b=json.load(f1)
    print(type(b))
    print(len(b)*3)
    print(a==b)

if __name__ == '__main__':
    main()