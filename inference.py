import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np 
import matplotlib.pyplot as plt
import os
import time
from data import TinyIN, cifar10train,cifar10test,cifar100, svhn
from tqdm import tqdm
import warnings 
warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    
    device = 'cuda'
else :
    device = 'cpu'

def student_inference(student,test_loader):
    
    student.eval()
    with torch.no_grad():
        n_samples = 0
        n_correct = 0
        for  i,(samples,labels) in enumerate(test_loader):
            samples = samples.to(device)
            labels = labels.to(device)
            predictions = student.forward(samples)
            n_samples+= samples.shape[0]
            _,y_hat  =  torch.max(predictions,1)
            n_correct+= y_hat.eq(labels).sum().item()
    acc = (n_correct/n_samples)
    return acc,n_correct,n_samples