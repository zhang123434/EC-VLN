features=vitbase_r2rfte2e
ft_dim=768
feedback=sample
ngpus=1


flag="--dataset r2r_last
      --ngpus ${ngpus}
      
      --hist_enc_pano
      --hist_pano_num_layers 2

      --fix_lang_embedding
      --fix_hist_embedding

      --features ${features}
      --feedback ${feedback}

      --max_action_len 15
      --batch_size 8
      --image_feat_size ${ft_dim}

      --lr 1e-5
      --iters 300000
      --log_every 1000
      --optim adamW

      --ml_weight 0.2
      --max_instr_len 60
      --angle_feat_size 4
      --feat_dropout 0.4
      --dropout 0.5"
#cktps
CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag \
      --output_dir ../datasets/R2R/exprs_r2rlast/ \
      --bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain-e2e/model_step_22000.pt \
      --features_aug vit-16-st-samefilter-768-e2e