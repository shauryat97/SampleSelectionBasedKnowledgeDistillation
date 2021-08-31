import torch
import numpy as np
import torch.nn as nn
import warnings 
warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    device = 'cuda'
else :
    device = 'cpu'


def entropy_threshold(teacher,confidence_loader,n_class):
    dct = {}
    sample_entropy = {}
    soft = nn.Softmax()
    for i in range(n_class):
        dct[i] = dct.get(i,0)
        sample_entropy[i] = sample_entropy.get(i,[])    
    for i,(image,label) in enumerate(confidence_loader):
        batch_size = image.shape[0]
        image = image.to(device) 
        with torch.no_grad():
            batch_pred = teacher(image) 
            for j in range(batch_size):
                logits= batch_pred[j]
                final_pred = soft(logits)
                topk_vals, pred_classes= final_pred.topk(2)
                top_class = pred_classes[0].item()
                entropy = 0
                final_pred = final_pred.cpu().numpy()
                for k in range(n_class):
                    entropy+=(-1*final_pred[k]*np.log(final_pred[k]))
                sample_entropy[top_class].append(entropy)
    top_entropy = {}
    for i in range(n_class):
        temp_lst = sample_entropy[i]
        temp_lst.sort()
        top_entropy[i] = temp_lst[:100] # taking top 10 % into account 
    new_entropy={}
    for i in range(n_class):
        new_entropy[i] = sum(top_entropy[i])/100
    return new_entropy