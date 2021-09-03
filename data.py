import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import Counter

# Transforms 

svhn_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
])

def cifar10train():

    cifar10train = torchvision.datasets.CIFAR10('/home/cds/Shaurya/Datasets',
                                              download=True,
                                              train=True,
                                              transform=transform)
    n_training = len(cifar10train)
    n_class = len(dict(Counter(cifar10train.targets)))
    cifar10train, confidence_set = torch.utils.data.random_split(cifar10train, [40000,10000])

    cifar10train_loader = torch.utils.data.DataLoader(cifar10train,
                                         batch_size = 64,
                                         shuffle = True
                                         )
    cifar10confidence_loader = torch.utils.data.DataLoader(confidence_set,
                                         batch_size = 64,
                                         shuffle = True
                                         )
    return cifar10train_loader,cifar10confidence_loader,n_class,n_training

def cifar10test():
    cifar10test = torchvision.datasets.CIFAR10('/home/cds/Shaurya/Datasets',download=True,
                                              train=False,
                                              transform=transform)
    cifar10test_loader = torch.utils.data.DataLoader(cifar10test,
                                         batch_size = 64,
                                         shuffle = False
                                         )
    return cifar10test_loader

def cifar100():
    cifar100train = torchvision.datasets.CIFAR100('/home/cds/Shaurya/Datasets',download=True,
                                              train=True,
                                              transform=transform)

    cifar100train_loader = torch.utils.data.DataLoader(cifar100train,
                                         batch_size = 64,
                                         shuffle = True
                                         )
    return cifar100train_loader


def svhn():

    svhn = torchvision.datasets.SVHN('/home/cds/Shaurya/Datasets',
                                download=True,
                                split='train',
                                transform=svhn_transform)

    svhn_loader = torch.utils.data.DataLoader(svhn,
                                         batch_size = 64,
                                         shuffle = True
                                         )
    return svhn_loader

def TinyIN():

    tinytrain_dir =  '/home/cds/Shaurya/tiny-imagenet-200/train'
    tiny_dataset = torchvision.datasets.ImageFolder(tinytrain_dir,transform=svhn_transform)
    tinytrain_loader = torch.utils.data.DataLoader(tiny_dataset,
                                              batch_size=64,shuffle=True
                                              )
    return tinytrain_loader

def caltech256():
    
    caltech256_dir = '/home/cds/Shaurya/Datasets/256_ObjectCategories'
    caltech256_dataset = torchvision.datasets.ImageFolder(caltech256_dir,transform=svhn_transform)
    caltech256_loader = torch.utils.data.DataLoader(caltech256_dataset,
                                              batch_size=64,shuffle=True
                                              )
    return caltech256_loader

def caltech101():
    caltech_dir = '/home/cds/Shaurya/Datasets/101_ObjectCategories'
    caltech_dataset = torchvision.datasets.ImageFolder(caltech_dir,transform=svhn_transform)
    caltech101_loader = torch.utils.data.DataLoader(caltech_dataset,
                                              batch_size=64,shuffle=True
                                              )
    return caltech101_loader



def stl10():
    stl10 = torchvision.datasets.STL10('/home/cds/Shaurya/Datasets',
                                 split  = 'train',
                                 transform = svhn_transform, 
                                 download = True)
    stl10_loader = torch.utils.data.DataLoader(stl10,
                                         batch_size = 64,
                                         shuffle = True
                                         )
    return stl10_loader

def places365():

    places365 = torchvision.datasets.Places365('/home/cds/Shaurya/Datasets',
                                                split = 'val',
                                                small = True,
                                                download = False, 
                                                transform = svhn_transform)
    places365_loader = torch.utils.data.DataLoader(places365,
                                         batch_size = 64,
                                         shuffle = True
                                         )
    return places365_loader