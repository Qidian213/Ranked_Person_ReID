# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline

def build_model(cfg, num_classes):
    model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.STN, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    return model
