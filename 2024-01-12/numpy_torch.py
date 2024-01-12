import torch
import numpy as np

a = torch.tensor([1,2,3], dtype=torch.float64)
print(a.numpy())

b = np.array([1,2,3])
print(torch.from_numpy(b))