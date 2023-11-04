features=vitbase_r2rfte2e
ft_dim=768
feedback=sample
ngpus=1

flag="--dataset r4r

      --seed 0
      --ngpus ${ngpus}

      --no_lang_ca
      --ob_type pano
      --hist_enc_pano
      --hist_pano_num_layers 2

      --features ${features}
      --feedback ${feedback}

      --max_instr_len 100
      --max_action_len 30
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-5
      --iters 300000
      --log_every 1000
      --optim adamW

      --ml_weight 0.2
      
      --batch_size 4
      --feat_dropout 0.4
      --dropout 0.5"

# train
#ckpts1:
CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag \
      --output_dir ../datasets/R2R/exprs_r4r/ \
      --bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain-e2e/model_step_22000.pt \
      --resume_file ../datasets/R2R/exprs_r4r/ckpts1/latest_dict \
      --features_aug vit-16-st-samefilter-768-e2e