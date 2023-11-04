import torch;
a=torch.tensor([[1,2]])
b=torch.tensor([[3,4]]);
print(a.shape)
print(torch.cat([a,b],0).shape)