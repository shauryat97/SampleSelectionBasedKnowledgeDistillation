import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import torchvision.datasets as datasets
import numpy as np
from Distill import *
from TeacherTrain import *
from mms_threshold import *
from transfer import *
from inference import *


# Train Teacher on Cifar 10 
# Target Dataset
cifar10train_loader,cifar10confidence_loader,n_class,n_training = cifar10train()
# teacher = train_teacher(cifar10train_loader)
teacher = torch.load('teacher.pth')

# Make Transfer set using Trained Teacher and MMS_scores

teacher.eval()
threshold = calc_mms_threshold(teacher,cifar10confidence_loader,n_class)
transfer_loader = transfer_set(threshold,n_training,n_class)


# Train student on Transfer set (Distillation)

student = torchvision.models.resnet18(pretrained=False)
student = train_student(teacher,student,transfer_loader,n_class)



# Inference on Test data to calculate accuarcy of student 
cifar10test_loader = cifar10test()
acc,n_correct,n_samples  = student_inference(student,cifar10test_loader)

print('Accuracy = ',acc)

