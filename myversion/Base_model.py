import sys
import math
import random
import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from utils import get_opt
from trainer import Trainer

class SelfAttention(nn.Module):
    '''
    若X: T*F Mask应为 T*T
    '''
    def __init__(self, input_size, output_size,drop_ratio = 0):
        super(SelfAttention, self).__init__()
        self.Dropout = nn.Dropout(drop_ratio)

        self.wq = nn.Linear(in_features=input_size,
                            out_features=output_size, bias=False)
        self.wk = nn.Linear(in_features=input_size,
                            out_features=output_size, bias=False)
        self.wv = nn.Linear(in_features=input_size,
                            out_features=output_size, bias=False)

    def forward(self, X, Mask = None):
        #X : B * T * input_size
        #Mask:T*T
        #qkv:B*T*output_size
        q = self.wq(X)
        k = self.wk(X)
        v = self.wv(X)

        dk = q.size(-1)
        #z: B*T*T
        z = torch.bmm(q, k.permute(0,2,1)) / math.sqrt(dk)
        if Mask:
            z = z.masked_fill_(Mask>0,-np.inf)
        beta = F.softmax(z, dim=2)
        beta = self.Dropout(beta)

        res = torch.bmm(beta, v)
        return res

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])