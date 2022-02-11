from queue import Queue
import numpy as np

a=[Queue(100),Queue(100),Queue(100),Queue(100)]
a[0].put(1)

a=[]
temp=[]

for i in range(5):
    temp.append(i)
a.append(temp)
temp=[]

for i in range(1,6):
    temp.append(i)
a.append(temp)
print(a)

s=np.array(a).T
print(s)