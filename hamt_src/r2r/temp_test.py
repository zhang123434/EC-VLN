import torch;
model=torch.load("/root/mount/Matterport3DSimulator/EnvEdit-main/datasets/R2R/trained_models/vitbase-6tasks-pretrain/model_step_130000.pt")
model1=torch.load("/root/mount/Matterport3DSimulator/EnvEdit-main/datasets/R2R/exprs/pretrain/cmt-vitbase-6tasks/ckpts/model_step_40000.pt")
for i in model.keys():
    # print(type(model[i]))
    if model[i].shape!=model1[i].shape:
        print("error!!")