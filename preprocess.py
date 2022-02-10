from operator import ge
from random import random
from torch.utils.data import DataLoader,RandomSampler
from torchvision import transforms  
import matplotlib.pyplot as plt
from customdatasets import CustomVisionDataset

def getDL():
    transform = transforms.Compose([
        transforms.Resize([256,256]),
        transforms.ToTensor(),
    ])
    
    gt_transform=transforms.Compose([
        transforms.Resize([256,256]),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    labeledSet = CustomVisionDataset(img_root='./mvtec_anomaly_detection/bottle/test',gt_root="./mvtec_anomaly_detection/bottle/ground_truth", img_transform=transform, gt_transform=gt_transform)
    unlabeledSet =  CustomVisionDataset(img_root='./mvtec_anomaly_detection/bottle/test',gt_root="./mvtec_anomaly_detection/bottle/ground_truth", img_transform=transform, gt_transform=gt_transform)
    return labeledSet,unlabeledSet
getDL()