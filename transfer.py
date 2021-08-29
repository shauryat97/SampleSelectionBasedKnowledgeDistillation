
import torch
import numpy as np
from torch._C import device
from mms_threshold import calc_mms_threshold
from data import *
import torchvision 
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset, DataLoader
import time

start = time.time()

if torch.cuda.is_available():
    
    device = 'cuda'
else :
    device = 'cpu'

# Datasets

tinytrain_loader = TinyIN()
svhn_loader = svhn()
cifar10train_loader,cifar10confidence_loader,n_class,n_training = cifar10train()
cifar100train_loader = cifar100()
cal101_loader = caltech101()
cal256_loader = caltech256()

def compute_loader(n_class,loader,teacher,bucket_size,dct,mms_avg):
    
    lst = []
    num_values = 0
    for i,(image,label) in enumerate(loader):
        batch_size = image.shape[0]
        image = image.to(device)
        with torch.no_grad():
            batch_pred = teacher(image)
            for j in range(batch_size):
                logits= batch_pred[j]
                topk_vals, pred_classes = logits.topk(2)
                top_class = pred_classes[0].item()
                w1 = teacher.fc.weight.index_select(0, pred_classes[0])
                w2 = teacher.fc.weight.index_select(0, pred_classes[1])
                mms_score = (topk_vals[0] - topk_vals[1]) / (w1 - w2).norm(dim=-1) 
                if (dct[top_class] <bucket_size) and mms_score>=mms_avg[top_class] :
                    
                    dct[top_class] +=1
                    img = image[j].cpu().data.numpy()
                    lst.append(img)
    for i in range(n_class):
        num_values+=dct[i]
    return lst,num_values

def transfer_set(threshold,n_training,n_class):
    
    dct = {}
    transfer_lst = []
    for i in range(n_class):
        dct[i]=0
    bucket_size = int(n_training/n_class)
    teacher = torch.load('teacher.pth')
    teacher.eval()
    sum_values = 0
    loader_lst = [svhn_loader,tinytrain_loader,cifar100train_loader,cal101_loader,cal256_loader]
    i = 0
    temp_lst = []
    while(sum_values!=(bucket_size*n_class) and i<len(loader_lst) ) :
        temp_lst,sum_values = compute_loader(n_class,loader_lst[i],teacher,bucket_size,dct,threshold)
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
    return transfer_loader




