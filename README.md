# Sample Selection Based Knowledge Distillation

Knowledge distillation is an effective approach for compressing deep neural networks by distilling the generalization capability (“dark knowledge”) of a complex network (teacher) to a simpler network (student).
The core idea of KD was given by Geoffrey Hinton et al. in 2014.[1].

# Task 
To introduce a metric for `selective sampling` from arbitrary datasets in order to  increase the distillation performance of the teacher model.

# Baseline
My baseline was based on the [paper](https://arxiv.org/abs/2011.09113) and work done by Gaurav et al on Data Free Knowledge Distillation.


# Models and Datasets
| Target Dataset        |Teacher Model          | Student Model  |
| ------------- |:-------------:| -----:   |
| `CIFAR10`   | `Resnet-34(pretrained on ImageNet)`| `Resnet-18` |
| `CIFAR100`   | `Inception-V3(pretrained on ImageNet)`| `Resnet-18` |

# Code files explained
- compile.py : is the main code which contains step-wise implementation of the algorithm.
- TeacherTrain.py is the code responsible for training the teacher model on the target dataset.
- data.py contains code for downloading all datasets.
- distill.py is the code responsible for distilling the knowledge from trained teacher model to the student model.
- mms_threshold.py and entropy.py contain code for calculating threshold metric for selective samplling.
- transfer.py is the code for constructing the transfer set.



