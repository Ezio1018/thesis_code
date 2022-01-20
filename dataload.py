from asyncio import trsock
import imp
import os
from random import shuffle
import pandas as pd
import torch 
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms,models

from const import DEVICE

data_dir = "./mvtec_anomaly_detection/bottle"
train_dir = data_dir + "/train"
test_dir = data_dir + "/test"

data_transoforms = {
    'train':transforms.Compose([
        transforms.Resize([256,256]),
        transforms.ToTensor(),
        ]
    ),
    'test':transforms.Compose([
        transforms.Resize([256,256]),
        transforms.ToTensor(),
        ]
    ),
}

batch_size = 5
image_datasets = {x:torchvision.datasets.ImageFolder(os.path.join(data_dir,x),data_transoforms[x]) for x in ['train','test']}
dataloaders =  {x:torch.utils.data.DataLoader(image_datasets[x],batch_size=batch_size,shuffle=True) for x in ['train','test']}

class_ids = image_datasets['test'].classes

for image_batch,_ in dataloaders['test']:
    # image_batch=image_batch.to(DEVICE)
    print(image_batch)
    