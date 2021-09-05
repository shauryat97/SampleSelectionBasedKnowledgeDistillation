# Sample Selection Based Knowledge Distillation

![KD_Diagram-2](https://user-images.githubusercontent.com/73641247/132132493-0d22c4c2-2450-4303-b2fd-f6f809d7f917.png)


Knowledge distillation is an effective approach for compressing deep neural networks by distilling the generalization capability (“dark knowledge”) of a complex network (teacher) to a simpler network (student).
The core idea of KD was given by Geoffrey Hinton et al. in 2014.[1].

# Task 
To introduce a metric for `selective sampling` from arbitrary datasets in order to  increase the distillation performance of the teacher model.

# Baseline
My baseline was based on the [paper](https://arxiv.org/abs/2011.09113) and work done by Gaurav et al on Data Free Knowledge Distillation.


# Models
| Target Dataset        |Teacher Model          | Student Model  |
| ------------- |:-------------:| -----:   |
| `CIFAR10`   | `Resnet-34(pretrained on ImageNet)`| `Resnet-18` |
| `CIFAR100`   | `Inception-V3(pretrained on ImageNet)`| `Resnet-18` |

# Datasets used for constructing transfer sets.
| Target Dataset        |Arbitrary Datasets used          |
| ------------- |:-------------:|
| CIFAR10   | SVHN , TinyImageNet , Places365 , Cifar100 , STL10 , Caltech256 , Caltech101 |
| CIFAR100  | SVHN , TinyImageNet , Places365 , Cifar10 , STL10 , Caltech256 , Caltech101|

#  Minimal Margin Score(MMS) as a sample selection metric
MMS was initially used by Berry et al for [selective sampling to accelerate the training of deep neural networks](https://arxiv.org/pdf/1911.06996.pdf) by selecting samples which have low margin distance from the decision boundary.This concept was motivated from Support Vector Machines , where samples far from the decision boundary were the ones on which the model had high confidence and samples close to the decision boundary were the ones on which model had low confidence.In order to train the model better, we had to focus on samples on which model had less confidence and we could ignore high confodence samples and thus saving the compute cost and time.
I used the same concept of confidence as a sample selection metric. I select those samples on which the teacher model has high confidence. This knowledge of teacher is further distilled through the process of Knowledge Distillation to the student model.

# Results

1. For CIFAR10 as target dataset and Resnet-18 as Student : 

|   Approach    |Accuracy(%)      |
| ------------- |:-------------:|
| Student pretrained on ImageNet without KD  | 79.65|
| Student untrained with KD using SVHN+TinyIN Balanced  | 60.67|
| `Student untrained with KD+MMS Score`   | `68.02`|

# Code files explained
- compile.py : is the main code which contains step-wise implementation of the algorithm.
- TeacherTrain.py is the code responsible for training the teacher model on the target dataset.
- data.py contains code for downloading all datasets.
- distill.py is the code responsible for distilling the knowledge from trained teacher model to the student model.
- mms_threshold.py and entropy.py contain code for calculating threshold metric for selective samplling.
- transfer.py is the code for constructing the transfer set.



