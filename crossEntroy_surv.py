import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time

class CrossEntropyLoss_surv(nn.Module):

    def __init__(self,s0 ,s ,m ,cutoff, size_loss='average'):
        super(CrossEntropyLoss_surv, self).__init__()
        self.size_loss = size_loss
        self.s0 = torch.exp(torch.Tensor([s0]))[0]
        self.cutoff = cutoff
        self.oaf = 1e-10
        self.s = s
        self.m = m
    
    def _get_label(self,y,e):
        count = torch.zeros(len(self.cutoff)-1)
        f = 0
        for i in range(len(self.cutoff)-1):
            if f == 1 and e == 0:
                count[i] = 1
            if y >= self.cutoff[i] and y < self.cutoff[i+1]:
                count[i] = 1
                f = 1
                if e == 0:
                    continue
                else:
                    break
        return count     


    def forward(self, input, y, e, device):
        batch_loss = 0.
        label = torch.Tensor([]).to(device)
        #计算batch中患者的重要性
        for i in range(input.shape[0]):
            count = self._get_label(y[i],e[i])
            count_remain = self._get_label(y[i],e[i])
            for j in range(count.shape[0]):
                if count[j] == torch.min(count):
                    input[i,j] += self.m
            if self.s != 1:
                input[i,:] *= self.s

            count = count.to(device)
            #损失计算
            count_reversal = torch.abs(count-1)
            #if e[i] == 1:
            if torch.sum(count_remain) == 1:
                prob_e = torch.exp(-1*input[i,:])
                prob_e_neg = torch.exp(input[i,:])
                correct = torch.sum(torch.mul(prob_e,count))+self.s0
                false = torch.sum(torch.mul(prob_e_neg,count_reversal))+self.s0

                loss = torch.log(correct) + torch.log(false)
            else:
                prob_e = torch.exp(-1*input[i,:])
                prob_e_neg = torch.exp(input[i,:])
                correct = torch.sum(torch.mul(prob_e,count))
                false = torch.sum(torch.mul(prob_e,count_reversal))
                loss = torch.log(1+correct*false+false*self.s0**(-1))

            if i == 0:
                label = torch.unsqueeze(count,0).clone()
            else:
                label = torch.cat([label,torch.unsqueeze(count,0)],0)
            # 损失累加
            batch_loss += loss

        # 整个 batch 的总损失是否要求平均
        if self.size_loss == 'average':
            batch_loss /= input.shape[0]
        label[label<0.5] = 0
        label[label>0.5] = 1
        return batch_loss,label.int()