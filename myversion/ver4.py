#!/usr/bin/env python
# encoding: utf-8
'''
测试卷积的self-attention结构
'''
import sys
import math
import random
import argparse

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable

from utils import get_opt
from trainer import Trainer

class TA(nn.Module):
    def __init__(self,v,L,C,C_alpha):
        
        super(TA,self).__init__()
        self.Q_weights = Parameter(torch.randn(v,C,C_alpha), requires_grad=True)
        self.K_weights = Parameter(torch.randn(v,C,C_alpha), requires_grad=True)
        self.V_weights = Parameter(torch.randn(v,C,C_alpha), requires_grad=True)
        self.Q_bais = Parameter(torch.randn(C_alpha), requires_grad=True)
        self.K_bais = Parameter(torch.randn(C_alpha), requires_grad=True)
        self.V_bais = Parameter(torch.randn(C_alpha), requires_grad=True)
        self.gama = Parameter(torch.randn(1), requires_grad=True)
        self.fc = nn.Linear(C_alpha,C)
        self.mask = torch.triu(torch.ones((L, L)),diagonal=1)*10**10
        self.mask = Variable(self.mask.type(torch.FloatTensor)).to("cuda:0")
    def forward(self,X):
        #X: B*V*L*C
        #QKV: B*V*L*C_alpha
        Q = torch.matmul(X,self.Q_weights) + self.Q_bais
        K = torch.matmul(X,self.K_weights) + self.K_bais
        V = torch.matmul(X,self.V_weights) + self.V_bais
        #alpha: B*V*L*L
        #self.mask.to('cuda:0')
        alpha = F.softmax(torch.matmul(Q,K.permute(0,1,3,2))-self.mask,dim = 3)
        #V_alpha:B*V*L*C_alpha
        V_alpha = torch.matmul(alpha,V)
        X = X + torch.mul(self.fc(V_alpha),self.gama)
        return X

class VA(nn.Module):
    def __init__(self,L,C,C_alpha):
        #论文中 L:T  C:input_size C_alpha:hidden_size
        super(VA,self).__init__()
        self.Q_weights = Parameter(torch.randn(L,C,C_alpha), requires_grad=True)
        self.K_weights = Parameter(torch.randn(L,C,C_alpha), requires_grad=True)
        self.V_weights = Parameter(torch.randn(L,C,C_alpha), requires_grad=True)
        self.Q_bais = Parameter(torch.randn(C_alpha), requires_grad=True)
        self.K_bais = Parameter(torch.randn(C_alpha), requires_grad=True)
        self.V_bais = Parameter(torch.randn(C_alpha), requires_grad=True)
        self.gama = Parameter(torch.randn(1), requires_grad=True)
        self.fc = nn.Linear(C_alpha,C)
    def forward(self,Y):
        #B*V*L*C ---> B*L*V*C
        Y = Y.permute(0,2,1,3)
        
        #QKV: B*L*V*C_alpha
        Q = torch.matmul(Y,self.Q_weights) + self.Q_bais
        K = torch.matmul(Y,self.K_weights) + self.K_bais
        V = torch.matmul(Y,self.V_weights) + self.V_bais
        #alpha: B*L*V*V
        bate = F.softmax(torch.matmul(Q,K.permute(0,1,3,2)),dim = 3)
        #V_bate:B*L*V*C_alpha
        V_bate = torch.matmul(bate,V)
        Y = Y + torch.mul(self.fc(V_bate),self.gama)
        Y = Y.permute(0,2,1,3)
        return Y


class CNN_self_attention(nn.Module):
    """encoder in DA_RNN."""

    def __init__(self, input_size,
                 hidden_size,
                 time_step,
                 drop_ratio):
        """Initialize an encoder in DA_RNN."""
        super(CNN_self_attention, self).__init__()
        self.num_hidden = hidden_size
        self.input_size = input_size
        self.T = time_step

        self.CNN_sequential = nn.Sequential(
            #B*1*F*T
            nn.Conv2d(in_channels = 1,out_channels = self.num_hidden,kernel_size=(9,1),padding=(4,0) ),
            nn.Conv2d(in_channels = self.num_hidden,out_channels = 2*self.num_hidden,kernel_size=(5,1),padding=(2,0)),
            nn.Conv2d(in_channels = 2*self.num_hidden,out_channels = self.num_hidden,kernel_size=(3,1),padding=(1,0))
            #B*128*F*T (B*C*V*L)
        )
        self.TA = TA(v =self.input_size, L = self.T, C = self.num_hidden, C_alpha = self.num_hidden)
        self.VA = VA(L = self.T, C = self.num_hidden, C_alpha = self.num_hidden)
        
        self.global_pool = nn.Sequential(
            nn.Conv2d(in_channels = self.num_hidden,out_channels = 1,kernel_size = 1),
            nn.AvgPool2d(3,stride=1,padding=1)
        )
        self.loss_func = nn.BCELoss()
        self.test = nn.Linear(input_size*time_step,1)
    def forward(self, X ,*args):
        
        #X:B*T*F --> B*F*T
        X = X.permute(0,2,1)
        #X:B*1*F*T
        X = X.unsqueeze(1)

        #B*128*F*T (B*C*V*L)
        
        X = self.CNN_sequential(X)

        #X:B*V*L*C
        X = X.permute(0,2,3,1)
        
        Y = self.TA(X)
        Y = self.VA(Y)
        #Y:B*C*L*V 
        Y = Y.permute(0,3,2,1)

        #Y:B*1*L*V
        Y = self.global_pool(Y)
        #Y:B*L*V
        Y = Y.squeeze(1)
        #-->B*(L*V)
        Y = Y.view(Y.size(0),-1)
        #B
        out = self.test(Y).squeeze(1)
        return torch.sigmoid(out)


if __name__ == '__main__':
    random.seed(0)
    parser = argparse.ArgumentParser(description='输入参数yml文件')
    parser.add_argument('-opt', type=str, default='./train.yml',
                        help='Path to option YAML file.')
    args = parser.parse_args()
    opt = get_opt(args.opt)[__file__[:-3]]
    Train = Trainer(CNN_self_attention,opt['model_conf'], opt['data_conf'], opt['train_conf'],__file__[:-3])
    Train.run()
