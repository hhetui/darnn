#!/usr/bin/env python
# encoding: utf-8
'''
基础 DARNN+self-attention 原始参数
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

class RCNN(nn.Module):
    def __init__(self,D):
        super(RCNN,self).__init__()
        #B*1*T*D
        self.rcnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,kernel_size = (1,D)),
            nn.Conv2d(in_channels=16, out_channels=32,kernel_size = (3,1)),
            nn.MaxPool2d(kernel_size = (2,1)),
            nn.Conv2d(in_channels=32, out_channels=32,kernel_size = (3,1)),
            nn.MaxPool2d(kernel_size = (2,1))
        )
        #B*32*T'*1
    def forward(self,X):
        return self.rcnn(X)


class CNN_downsampling(nn.Module):
    def __init__(self,input_size,
                    time_step):
        super(CNN_downsampling,self).__init__()
        self.D = input_size
        self.T = time_step
        
        self.S = int((self.T-1)/9)
        self.RCNN_list = nn.ModuleList([RCNN(self.D) for s in range(1,self.S+1)])

    def forward(self,X):
        #B*T*D
        for s in range(1,self.S+1):
            x_s = X[:,list(range(0,self.T,s)),:] #B*T_s*D
            x_s = x_s.unsqueeze(1) #B*1*T_s*D
            x_s = self.RCNN_list[s-1](x_s) #B*32*v_num*1
            x_s = x_s.squeeze(3).permute(0,2,1)#B*v_num*32
           
            if s == 1:
                res = x_s #B*v_num*32 第一个v_num最长
                self.MAX_len = x_s.size(1)
            else:
                x_s = torch.cat((torch.zeros(x_s.size(0),self.MAX_len-x_s.size(1),x_s.size(2)).to(x_s.device),x_s),dim = 1) #B*T*32在前面补0
                res = torch.cat((res,x_s),dim = 2) #B*T*(32xs)
        
        return res #B*T*(32xS)


class Encoder(nn.Module):
    """encoder in DA_RNN."""

    def __init__(self, input_size,
                 hidden_size,
                 time_step,
                 drop_ratio):
        """Initialize an encoder in DA_RNN."""
        super(Encoder, self).__init__()
        self.encoder_num_hidden = hidden_size
        
        self.T = time_step

        self.CNN_downsampling = CNN_downsampling(input_size=input_size,
                                                time_step = self.T)
        self.S = int((self.T-1)/9)
        self.num_hidden = 32*self.S
        self.encoder_lstm = nn.LSTM(
            input_size=self.num_hidden,
            hidden_size=self.encoder_num_hidden,
            num_layers=1
        )

    def forward(self, X):
        #B*T*F -- > B*不知道多少*(32xS)
        X = self.CNN_downsampling(X)
        self.T = X.size(1)
        X_encoded = Variable(X.data.new(
            X.size(0), self.T, self.encoder_num_hidden).zero_())
        h_n = self._init_states(X)
        s_n = self._init_states(X)
        for t in range(self.T):
            
            self.encoder_lstm.flatten_parameters()

            _, final_state = self.encoder_lstm(
                X[:,t,:].unsqueeze(0), (h_n, s_n))
            h_n = final_state[0]
            s_n = final_state[1]

            X_encoded[:, t, :] = h_n
        return X_encoded

    def _init_states(self, X):
        return Variable(X.data.new(1, X.size(0), self.encoder_num_hidden).zero_())


class Decoder(nn.Module):
    """decoder in DA_RNN."""

    def __init__(self, encoder_num_hidden,
                 hidden_size,
                 time_step
                 ):
        """Initialize a decoder in DA_RNN."""
        super(Decoder, self).__init__()
        self.decoder_num_hidden = hidden_size
        self.encoder_num_hidden = encoder_num_hidden
        self.T = time_step
        self.v_num = int((int((time_step-2)/2)-2)/2)
        self.S = int((self.v_num-1)/9)
        
        self.lstm_layer = nn.LSTM(
            input_size=self.encoder_num_hidden+1,
            hidden_size=self.decoder_num_hidden
        )
        #self.fc = nn.Linear(self.num_hidden, 1)
        #self.fc.weight.data.normal_()

    def forward(self, X_encoded, y_prev):
        y_prev = y_prev[:,list(range(self.T-1,0,-int(self.T/X_encoded.size(1))))[::-1]]
        #X_encoded B*T_new*encoder_num_hidden ->B*T_new*encoder_num_hidden+1
        X_encoded = torch.cat((X_encoded,y_prev.unsqueeze(2)),dim = 2)
        d_n = self._init_states(X_encoded)
        c_n = self._init_states(X_encoded)

        for t in range( X_encoded.size(1)):
            #y_tilde = self.fc(X_encoded[:,t,:])
            _, final_states = self.lstm_layer(
                X_encoded[:,t,:].unsqueeze(0), (d_n, c_n))

            d_n = final_states[0]  # 1 * batch_size * decoder_num_hidden
            c_n = final_states[1]  # 1 * batch_size * decoder_num_hidden
        return d_n.squeeze(0)

    def _init_states(self, X):
        return Variable(X.data.new(1, X.size(0), self.decoder_num_hidden).zero_())


class SelfAttention(nn.Module):
    def __init__(self, last_hidden_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.last_hidden_size = last_hidden_size
        self.hidden_size = hidden_size

        self.wq = nn.Linear(in_features=last_hidden_size,
                            out_features=hidden_size, bias=False)
        self.wk = nn.Linear(in_features=last_hidden_size,
                            out_features=hidden_size, bias=False)
        self.wv = nn.Linear(in_features=last_hidden_size,
                            out_features=hidden_size, bias=False)

        # 输出
        self.ln = nn.Linear(in_features=hidden_size, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        # h: batch_size * last_hidden_size
        # q k v  batch_size * hidden_size
        q = self.wq(h)
        k = self.wk(h)
        v = self.wv(h)

        dk = q.size(-1)
        # (b, hidden_size) * (hidden_size, b) ==> (b, b)
        z = torch.mm(q, k.t()) / math.sqrt(dk)
        beta = F.softmax(z, dim=1)
        # (b, b) * (b, hidden_size) ==> (b, hidden_size)
        st = torch.mm(beta, v)

        # b * 1
        y_res = self.ln(st)
        # y_res: (batch_size, 1)
        y_res = self.sigmoid(y_res.squeeze(1))
        return y_res


class Darnn_selfattention(nn.Module):
    def __init__(self, input_size, T,
                 encoder_num_hidden,
                 decoder_num_hidden,
                 drop_ratio):
        super(Darnn_selfattention, self).__init__()
        self.input_size = input_size
        self.T = T
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden

        self.Encoder = Encoder(input_size=input_size, hidden_size=encoder_num_hidden,
                               time_step=T, drop_ratio=0)
        self.Decoder = Decoder(encoder_num_hidden=encoder_num_hidden,
                               hidden_size=decoder_num_hidden,
                               time_step=T)
        self.attention = SelfAttention(
            last_hidden_size=encoder_num_hidden, hidden_size=decoder_num_hidden)
        self.drop_out = nn.Dropout(drop_ratio)
        self.loss_func = nn.BCELoss()

    def forward(self, x, y):
        out1 = self.Encoder(x)
        out1 = self.drop_out(out1)
        out2 = self.Decoder(out1, y)
        out3 = self.attention(out2)
        return out3




if __name__ == '__main__':
    random.seed(0)
    parser = argparse.ArgumentParser(description='输入参数yml文件')
    parser.add_argument('-opt', type=str, default='./train.yml',
                        help='Path to option YAML file.')
    args = parser.parse_args()
    opt = get_opt(args.opt)[__file__[:-3]]
    Train = Trainer(Darnn_selfattention,opt['model_conf'], opt['data_conf'], opt['train_conf'],__file__[:-3])
    Train.run()