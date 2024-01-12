import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6], device=device)
print(a)
print(b)
