# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by Bercy
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging
import numpy as np
import configparser

from lifelines.utils import concordance_index



def c_index(risk_pred, y, e):
    ''' Performs calculating c-index

    :param risk_pred: (np.ndarray or torch.Tensor) model prediction
    :param y: (np.ndarray or torch.Tensor) the times of event e
    :param e: (np.ndarray or torch.Tensor) flag that records whether the event occurs
    :return c_index: the c_index is calculated by (risk_pred, y, e)
    '''
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    return concordance_index(y, risk_pred, e)

def adjust_learning_rate(optimizer, epoch, lr, lr_decay_rate, warm_epoch, warmup):
    ''' Adjusts learning rate according to (epoch, lr and lr_decay_rate)

    :param optimizer: (torch.optim object)
    :param epoch: (int)
    :param lr: (float) the initial learning rate
    :param lr_decay_rate: (float) learning rate decay rate
    :return lr_: (float) updated learning rate
    '''
    if warmup:
        if epoch < warm_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr / (warm_epoch-epoch)
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr / (1+(epoch-warm_epoch)*lr_decay_rate)
        #print(lr / (warm_epoch-epoch))
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr / (1+epoch*lr_decay_rate)
    return optimizer.param_groups[0]['lr']

