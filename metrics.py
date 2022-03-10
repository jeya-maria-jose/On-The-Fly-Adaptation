import numpy as np
import torch
import torch.nn.functional as F
import pdb

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    # pdb.set_trace()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    return iou, dice

def iou_score_m(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    

    output_ = output[:,:,:,:] > 0.5
    target_ = target[:,:,:,:] > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou1 = (intersection + smooth) / (union + smooth)
    wt = (2* iou1) / (iou1+1)

    output_ = output[:,0:1,:,:] > 0.5
    target_ = target[:,0:1,:,:] > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou2 = (intersection + smooth) / (union + smooth)
    tc = (2* iou2) / (iou2+1)

    output_ = output[:,2,:,:] > 0.5
    target_ = target[:,2,:,:] > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou3 = (intersection + smooth) / (union + smooth)
    et = (2* iou3) / (iou3+1)

    return wt,et,tc

def dice_brats(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
        
    output_ = output[:,:,:,:] > 0.5
    target_ = target[:,:,:,:] > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou1 = (intersection + smooth) / (union + smooth)
    wt = (2* iou1) / (iou1+1)

    output_ = output[:,0:1,:,:] > 0.5
    target_ = target[:,0:1,:,:] > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou2 = (intersection + smooth) / (union + smooth)
    tc = (2* iou2) / (iou2+1)

    output_ = output[:,2,:,:] > 0.5
    target_ = target[:,2,:,:] > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou3 = (intersection + smooth) / (union + smooth)
    et = (2* iou3) / (iou3+1)

    return wt,et,tc

def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
