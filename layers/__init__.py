# encoding: utf-8

import torch.nn.functional as F

from .reanked_loss import RankedLoss, CrossEntropyLabelSmooth
from .reanked_clu_loss import CRankedLoss

def make_loss(cfg, num_classes):
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'ranked_loss':
        ranked_loss = RankedLoss(cfg.SOLVER.MARGIN_RANK,cfg.SOLVER.ALPHA,cfg.SOLVER.TVAL) # ranked_loss
        
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'cranked_loss':
        cranked_loss = CRankedLoss(cfg.SOLVER.MARGIN_RANK,cfg.SOLVER.ALPHA,cfg.SOLVER.TVAL) # cranked_loss
        
    else:
        print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'ranked_loss':
        def loss_func(score, feat, target):
            return ranked_loss(feat, target)[0] 
    elif cfg.DATALOADER.SAMPLER == 'cranked_loss':
        def loss_func(score, feat, target):
            return cranked_loss(feat, target)[0] 
    elif cfg.DATALOADER.SAMPLER == 'softmax_rank':
        def loss_func(score, feat, target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'ranked_loss':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return  xent(score, target) + 0.4*ranked_loss(feat, target)[0] # new add by zzg, open label smooth
                else:
                    return F.cross_entropy(score, target) + ranked_loss(feat, target)[0]    # new add by zzg, no label smooth

            elif cfg.MODEL.METRIC_LOSS_TYPE == 'cranked_loss':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return  xent(score, target) + 0.4*cranked_loss(feat, target)[0] # new add by zzg, open label smooth
                else:
                    return F.cross_entropy(score, target) + cranked_loss(feat, target)[0]    # new add by zzg, no label smooth
            else:
                print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster，'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    else:
        print('expected sampler should be softmax, ranked_loss or cranked_loss, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


