import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings 
warnings.filterwarnings('ignore')



if torch.cuda.is_available():
    device = 'cuda'
else :
    device = 'cpu'

def softmax_mod(x,T):
    x = torch.div(x,T)
    return torch.nn.Softmax(dim=1)(x)

def loss_fn_kd(student_outputs,  teacher_outputs,T):
    KD_loss = nn.KLDivLoss()(F.log_softmax(student_outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * ( T * T)
    return KD_loss


def train_student(teacher,student,transfer_loader,n_class):

    class Distiller(nn.Module):
        def __init__(self,teacher,student):
            super(Distiller,self).__init__()
            self.teacher = teacher
            self.student = student
        def forward(self,x):
            teacher_logits = self.teacher(x)
            student_logits = self.student(x)
            return teacher_logits,student_logits
    distiller = Distiller(teacher,student)

    num_ftrs = student.fc.in_features
    num_epochs = 20
    student.fc = nn.Linear(num_ftrs,n_class)
    student.to(device)
    optimizer = torch.optim.SGD(student.parameters(),lr=0.1)
    n_total_steps = len(transfer_loader)
    temperature = 10
    for epoch in tqdm(range(num_epochs)):
        for i,(image,labels) in enumerate(transfer_loader):
            image = image.to(device)
            labels = labels.to(device)
            teacher_logits,student_logits= distiller.forward(image)
            loss = loss_fn_kd(student_logits,teacher_logits,temperature) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1)%64==0:
                print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss: {loss.item():.4f}')
    torch.save(student,'student.pth')
    return student