from time import time

import torch

def count():
    points = array.uniform_(0, 1)
    d = (points[0].square()+points[1].square()).sqrt()
    idx = torch.where(d<=1)
    return idx[0].shape[0]

batch = 1_000_000
epochs = 200
device = "cuda" if torch.cuda.is_available() else "cpu"
array = torch.zeros([2, batch], dtype=torch.float32).to(device)
incircle = 0
for epoch in range(epochs):
    t1 = time()
    incircle += count()
    area = incircle / ((epoch+1)*batch)
    pi = area * 4
    t2 = time()
    print(f"Epoch: {epoch+1}, Time: {t2-t1} seconds, PI: {pi}")