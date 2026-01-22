import torch

a = torch.tensor([1, 2, 3])

b = torch.tensor([0, 1], dtype=torch.int32)
print(a[b])

c = torch.tensor([], dtype=torch.int32)

print(c.numel())

print(len(b))