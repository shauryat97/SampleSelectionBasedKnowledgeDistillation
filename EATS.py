import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from inference import *
from data import *
from tqdm import tqdm
import warnings 
warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    
    device = 'cuda'
else :
    device = 'cpu'

tinytrain_loader = TinyIN()
svhn_loader = svhn()
teacher = torch.load('teacher.pth')

def compute_loader(n_class,loader,teacher,bucket_size,dct):
    lst = []
    num_values = 0
    soft = nn.Softmax()
    for i,(image,label) in enumerate(loader):
        batch_size = image.shape[0]
        image = image.to(device)
        with torch.no_grad():
            batch_pred = teacher(image)
            for j in range(batch_size):
                logits = batch_pred[j]
                final_pred = soft(logits)
                topk_vals, pred_classes= final_pred.topk(2)
                top_class = pred_classes[0].item()
                if (dct[top_class] < bucket_size):
                    dct[top_class] +=1
                    img = image[j].cpu().data.numpy()
                    lst.append(img)
    for i in range(n_class):
        num_values+=dct[i]
    return lst,num_values 

def transfer_set(n_training,n_class):
    
    dct = {}
    transfer_lst = []
    for i in range(n_class):
        dct[i]=0
    bucket_size = int(n_training/n_class)
    teacher = torch.load('teacher.pth')
    teacher.eval()
    sum_values = 0
    loader_lst = [svhn_loader,tinytrain_loader]
    i = 0
    temp_lst = []
    while(sum_values!=(bucket_size*n_class) and i<len(loader_lst) ) :
        temp_lst,sum_values = compute_loader(n_class,loader_lst[i],teacher,bucket_size,dct)
        transfer_lst.extend(temp_lst)
        i+=1
    transfer = np.array(transfer_lst)
    transfer = torch.Tensor(transfer)
    total = len(transfer_lst)
    y = torch.rand(total)
    my_dataset = TensorDataset(transfer,y)
    transfer_loader  = DataLoader(my_dataset,batch_size=64,shuffle=True)
    my_data_numpy = np.array(transfer_lst)
    np.save('my_data_numpy',my_data_numpy)
    return transfer_loader,dct,i

cifar10train_loader,cifar10confidence_loader,n_class,n_training = cifar10train()
cifar10test_loader = cifar10test()
def softmax_mod(x,T):
    x = torch.div(x,T)
    return torch.nn.Softmax(dim=1)(x)

def loss_fn_kd(student_outputs,  teacher_outputs,T):
    KD_loss = nn.KLDivLoss()(F.log_softmax(student_outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * ( T * T)
    return KD_loss
def train_student_old(teacher,student,transfer_loader,n_class):

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
def train_student( cifar10train_loader,n_class,student =torchvision.models.resnet18(pretrained=True)):

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
            output = student(image).long()
            # output
            loss = criterion(output,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if  (i+1)%64==0:
                print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss: {loss.item():.4f}')
    return student
student_trained = train_student(cifar10train_loader,n_class)
acc_trained,n_correct,n_samples  = student_inference(student_trained,cifar10test_loader)
print('Trained Accuracy = ',acc_trained)
transfer_loader,dct,num_datasets = transfer_set(n_training,n_class)
student_KD = torchvision.models.resnet18(pretrained=False)
student_KD = train_student_old(teacher,student_KD,transfer_loader,n_class)
acc_KD,n_correct,n_samples  = student_inference(student_KD,cifar10test_loader)
print('KD Accuracy = ',acc_KD)
