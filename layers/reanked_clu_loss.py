# encoding: utf-8
"""
@author:  zzg
@contact: xhx1247786632@gmail.com
"""
import numpy as np
import torch
from torch import nn
from scipy.cluster.vq import vq,kmeans

def normalize_rank(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def euclidean_dist_rank(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def necrank_loss(dist_mat, labels, margin,alpha,tval):
    """
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    total_loss = 0.0
    for ind in range(N):
        is_neg = labels.ne(labels[ind])
        dist_an = dist_mat[ind][is_neg]
        an_is_pos = torch.lt(dist_an,alpha)
        an_less_alpha = dist_an[an_is_pos]
        an_weight = torch.exp(tval*(-1*an_less_alpha+alpha))
        an_weight_sum = torch.sum(an_weight) + 1e-5
        an_dist_lm = alpha - an_less_alpha
        an_ln_sum = torch.sum(torch.mul(an_dist_lm,an_weight))
        loss_an = torch.div(an_ln_sum,an_weight_sum)

        total_loss = total_loss+loss_an
    total_loss = total_loss*1.0/N
    return total_loss

def pocrank_loss(dist_mat, margin,alpha,tval):
    """
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    
    total_loss = 0.0
    for ind in range(N):
        dist_ap = dist_mat[ind]

        ap_is_pos = torch.clamp(torch.add(dist_ap,margin-alpha),min=0.0)
        ap_pos_num = ap_is_pos.size(0) +1e-5
        ap_pos_val_sum = torch.sum(ap_is_pos) 
        loss_ap = torch.div(ap_pos_val_sum,float(ap_pos_num)) 
        
        total_loss = total_loss+loss_ap
    total_loss = total_loss*1.0/N
    return total_loss

class CRankedLoss(object):
    def __init__(self, margin=None, alpha=None, tval=None):
        self.margin = margin
        self.alpha = alpha
        self.tval = tval
        
    def __call__(self, global_feat, labels, normalize_feature=True):
        if normalize_feature:
            global_feat = normalize_rank(global_feat, axis=-1)

        clu_features = global_feat.clone()
        clu_features = clu_features.detach().cpu().numpy()
        
        centroid = kmeans(clu_features, 8 ,iter=10)[0]
        appearanceGroups = vq(clu_features,centroid)[0] 
        
        allGroups = np.unique(appearanceGroups)
        
        total_loss = 0
        for group in allGroups:
            indices = np.where(appearanceGroups == group)
            groupfeatures = global_feat[indices]
            grouplabel = labels[indices]
            
            if(grouplabel.size()[0] >3):
                nedist_mat = euclidean_dist_rank(groupfeatures, groupfeatures)
                ne_loss = necrank_loss(nedist_mat,grouplabel,self.margin,self.alpha,self.tval)

                total_loss += ne_loss

        labelgp = labels.clone().cpu().numpy()
        IDGroup = np.unique(labelgp)
        for ind in IDGroup:
            is_pos = np.where(labelgp == ind)
            posfeatures = global_feat[is_pos]
            
            posdist_mat = euclidean_dist_rank(posfeatures, posfeatures)
            
            pos_loss = pocrank_loss(posdist_mat, self.margin,self.alpha,self.tval)
            total_loss += pos_loss
            
        return total_loss




