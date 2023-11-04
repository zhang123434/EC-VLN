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

      --maxInput 100
      --maxAction 30
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-5
      --iters 300000
      --log_every 1000
      --optim adamW

      --mlWeight 0.2
      
      --batchSize 1400
      --featdropout 0.4
      --dropout 0.5"

# train
#ckpts
# CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag \
#       --output_dir ../datasets/R2R/exprs_r4r/ \
#       --bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain-e2e/model_step_22000.pt \
#       --features_aug vit-16-spade-original-768-e2e

CUDA_VISIBLE_DEVICES='0' python r2r/test.py $flag \
     --resume_file  ../datasets/R2R/exprs_r4r/ckpts/best_val_unseen_sampled \
     --resume_file1  ../datasets/R2R/exprs_r4r/ckpts1/best_val_unseen_sampled \
     --test --submit