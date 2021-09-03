import torch
from data import *
import warnings 
warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    device = 'cuda'
else :
    device = 'cpu'

def calc_mms_threshold(teacher,confidence_loader,n_class):
    
    dct = {}
    top_mms = {}
    mms_avg = {}
    sample_confidence = {}
    for i in range(n_class):
        dct[i] = dct.get(i,0)
        sample_confidence[i] = sample_confidence.get(i,[])
        mms_avg[i] = mms_avg.get(i,0)
    for i,(image,labels) in  enumerate(confidence_loader):
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
                sample_confidence[top_class].append(mms_score.item())
    for i in range(n_class):
        temp_lst = sample_confidence[i]
        temp_lst.sort(reverse=True)
        top_mms[i] = temp_lst[:100] # taking top 10 % into account 
    new_mms={}
    for i in range(n_class):
        new_mms[i] = sum(top_mms[i])/100
    for i in range(n_class):
        mms_avg[i] = sum(sample_confidence[i])/len(sample_confidence[i])
    # return mms_avg
    return mms_avg

