import numpy as np
from torch.utils.data import DataLoader,BatchSampler

a=[1,2,3,4,5,6,7,8,9,0]

d=DataLoader(a,3,shuffle=True)

for i in d:
    print(i)