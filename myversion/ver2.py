#!/usr/bin/env python
# encoding: utf-8
'''
只使用LSTM 
'''
import sys
import math
import random
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import get_opt
from trainer import Trainer


class myLSTM(nn.Module):
    """encoder in DA_RNN."""

    def __init__(self, input_size,
                 hidden_size,
                 time_step,
                 drop_ratio):
        """Initialize an encoder in DA_RNN."""
        super(myLSTM, self).__init__()
        self.encoder_num_hidden = hidden_size
        self.input_size = input_size
        self.T = time_step

        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.encoder_num_hidden,
            num_layers=1
        )

        self.fc = nn.Sequential(
            nn.Linear(self.encoder_num_hidden, 1),
            nn.Sigmoid()
        )
        self.loss_func = nn.BCELoss()

    def forward(self, X,y):
        #X B*T*F
        X_encoded = Variable(X.data.new(
            X.size(0), self.T, self.encoder_num_hidden).zero_())
        h_n = self._init_states(X)
        s_n = self._init_states(X)
        self.encoder_lstm.flatten_parameters()
        _, final_state = self.encoder_lstm(
                X.permute(1,0,2), (h_n, s_n))
        #h 1*B*H
        h_n = final_state[0]
        out = self.fc(h_n[0])
        
        return out.squeeze(1)

    def _init_states(self, X):
        return Variable(X.data.new(1, X.size(0), self.encoder_num_hidden).zero_())



random.seed(0)

if __name__ == '__main__':
    random.seed(0)
    parser = argparse.ArgumentParser(description='输入参数yml文件')
    parser.add_argument('-opt', type=str, default='./train.yml',
                        help='Path to option YAML file.')
    args = parser.parse_args()
    opt = get_opt(args.opt)[__file__[:-3]]
    Train = Trainer(myLSTM,opt['model_conf'], opt['data_conf'], opt['train_conf'],__file__[:-3])
    Train.run()