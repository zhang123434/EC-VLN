import torch


def get_tokenizer(args):
    from transformers import AutoTokenizer
    if args.dataset == 'rxr' or args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
    else:
        cfg_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained("/root/mount/Matterport3DSimulator/EnvEdit-main/hamt_src_ensemble/pretrain_bin/")
    return tokenizer

def get_vlnbert_models(args, config=None):
    
    from transformers import PretrainedConfig
    from models.vilmodel_cmt import NavCMT

    model_class = NavCMT

    model_name_or_path = args.bert_ckpt_file
    new_ckpt_weights = {}
    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path)
        for k, v in ckpt_weights.items():
            # print(k);
            if k.startswith('module'):
                print(k)
                new_ckpt_weights[k[7:]] = v
            else:
                # add next_action in weights
                if k.startswith('next_action'):
                    print(k)
                    k = 'bert.' + k
                new_ckpt_weights[k] = v
    print()
    if args.dataset == 'rxr' or args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
    else:
        cfg_name = 'bert-base-uncased'
    vis_config = PretrainedConfig.from_pretrained(cfg_name)

    if args.dataset == 'rxr' or args.tokenizer == 'xlm':
        vis_config.type_vocab_size = 2
    
    vis_config.max_action_steps = 100
    vis_config.image_feat_size = args.image_feat_size
    vis_config.angle_feat_size = args.angle_feat_size
    vis_config.num_l_layers = args.num_l_layers
    vis_config.num_r_layers = 0
    vis_config.num_h_layers = args.num_h_layers
    vis_config.num_x_layers = args.num_x_layers
    vis_config.hist_enc_pano = args.hist_enc_pano
    vis_config.num_h_pano_layers = args.hist_pano_num_layers
    
    vis_config.fix_lang_embedding = args.fix_lang_embedding
    vis_config.fix_hist_embedding = args.fix_hist_embedding
    vis_config.fix_obs_embedding = args.fix_obs_embedding

    vis_config.update_lang_bert = not args.fix_lang_embedding
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1

    vis_config.no_lang_ca = args.no_lang_ca
    vis_config.act_pred_token = args.act_pred_token
    vis_config.max_action_steps = 50 
    vis_config.max_action_steps = 50
    vis_config.d_model=args.d_model;
    vis_config.factor=args.factor;
    vis_config.n_heads=args.n_heads;
    vis_config.d_ff=args.d_ff;
    vis_config.Inform_dropout=args.Inform_dropout;
    vis_config.activation=args.activation
    vis_config.output_attention=args.output_attention
    vis_config.mix=args.mix;
    vis_config.e_layers=args.e_layers;
    vis_config.s_layers=args.s_layers;
    vis_config.distill_layers=args.distill_layers
    # print("max_action_steps:",vis_config.max_action_steps)
    # print("config.hidden size",vis_config.hidden_size)
    # print("config.n_head",vis_config.num_attention_heads)
    print("output_attention:",vis_config.output_attentions)
    visual_model = model_class.from_pretrained(
        pretrained_model_name_or_path=None, 
        config=vis_config, 
        state_dict=new_ckpt_weights)
        
    return visual_model
