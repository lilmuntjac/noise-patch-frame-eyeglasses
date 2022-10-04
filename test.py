import torch

a = torch.tensor([[1,2,3,4,5],[1,2,3,5,4],[5,4,3,2,1]])
b = torch.tensor([1,0,1,0,2])
c = torch.tensor([1,4,6,4,1])

print(b*c)
print(torch.mean(a.to(torch.float32)))

print(a.shape)
print(b.shape)