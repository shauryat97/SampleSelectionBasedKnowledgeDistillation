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


# Training Teacher Model on Target Dataset - CIFAR10
def train_teacher(cifar10train_loader):

    teacher = torchvision.models.resnet34(pretrained=True)
    num_ftrs = teacher.fc.in_features
    teacher.fc = nn.Linear(num_ftrs,10)
    teacher.to(device)
    n_total_steps = len(cifar10train_loader)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(teacher.parameters(),lr=0.1)
    num_epochs = 20
    for epoch in range(num_epochs):
        for i,(image,label) in enumerate(cifar10train_loader):
            image = image.to(device)
            label = label.to(device)
            # forward
            output = teacher(image)
            loss = criterion(output,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if  (i+1)%64==0:
                print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss: {loss.item():.4f}')

    print('#######################  Teacher Training done  ##########################')
    torch.save(teacher,'teacher.pth')