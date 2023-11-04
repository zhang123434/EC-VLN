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

CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag \
     --resume_file  ../datasets/R2R/expr_r2rlast/best_second \
     --test --submit