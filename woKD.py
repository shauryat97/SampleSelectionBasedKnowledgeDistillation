import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.datasets as datasets
import numpy as np
from inference import *
from tqdm import tqdm
import warnings 
warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    
    device = 'cuda'
else :
    device = 'cpu'

cifar10train_loader,cifar10confidence_loader,n_class,n_training = cifar10train()
cifar10test_loader = cifar10test()
student_untrained = torchvision.models.resnet18(pretrained=True) # Just trained on ImageNet
num_ftrs = student_untrained.fc.in_features
student_untrained.fc = nn.Linear(num_ftrs,n_class)
student_untrained.to(device)
def train_student(cifar10train_loader,n_class):

    student = torchvision.models.resnet18(pretrained=True) # train later on Cifar10
    num_ftrs = student.fc.in_features
    student.fc = nn.Linear(num_ftrs,n_class)
    student.to(device)
    n_total_steps = len(cifar10train_loader)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(student.parameters(),lr=0.1)
    num_epochs = 20
    for epoch in range(num_epochs):
        for i,(image,label) in enumerate(cifar10train_loader):
            image = image.to(device)
            label = label.to(device)
            # forward
            output = student(image)
            loss = criterion(output,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if  (i+1)%64==0:
                print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss: {loss.item():.4f}')
    return student
student_trained = train_student(cifar10train_loader,n_class)
acc_trained,n_correct,n_samples  = student_inference(student_trained,cifar10test_loader)
acc_untrained,n_correct,n_samples  = student_inference(student_untrained,cifar10test_loader)
print('Trained Accuracy = ',acc_trained)
print('Untrained Accuracy = ',acc_untrained)
