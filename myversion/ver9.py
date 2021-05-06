#!/usr/bin/env python
# encoding: utf-8
'''
DARNN中使用CNNself-attention结构（1）
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

    def __init__(self, input_size,
                 hidden_size,
                 time_step,
                 drop_ratio = 0):
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
        
    def forward(self, X ,*args):
        
        #X:B*T*F --> B*F*T
        X = X.permute(0,2,1)
        #X:B*1*F*T
        X = X.unsqueeze(1)

        #B*128*F*T (B*C*V*L)
        
        X = self.CNN_sequential(X)
        #X:B*V*L*C
        X = X.permute(0,2,3,1)
        X = (X - torch.mean(X,dim =1).unsqueeze(1))/torch.std(X,dim = 1).unsqueeze(1) 
        Y = self.TA(X)
        Y = self.VA(Y)
        #Y:B*C*L*V 
        Y = Y.permute(0,3,2,1)

        #Y:B*1*L*V
        Y = self.global_pool(Y)
        #Y:B*L*V
        Y = Y.squeeze(1)
        
        return Y #B * T * F

class Encoder(nn.Module):
    """encoder in DA_RNN."""

    def __init__(self, input_size,
                 hidden_size,
                 time_step,
                 drop_ratio):
        """Initialize an encoder in DA_RNN."""
        super(Encoder, self).__init__()
        self.encoder_num_hidden = hidden_size
        self.input_size = input_size
        self.T = time_step

        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.encoder_num_hidden,
            num_layers=1
        )
        self.CA = CNN_self_attention(input_size = self.input_size, 
                                    hidden_size = self.encoder_num_hidden,
                                    time_step = self.T
                                    )
        

    def forward(self, X):
        X = self.CA(X)
        X_encoded = Variable(X.data.new(
            X.size(0), self.T, self.encoder_num_hidden).zero_())
        h_n = self._init_states(X)
        s_n = self._init_states(X)
        for t in range(self.T):
            self.encoder_lstm.flatten_parameters()
            _, final_state = self.encoder_lstm(
                X[:, t, :].unsqueeze(0), (h_n, s_n))
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

        self.CA = CNN_self_attention(input_size = self.decoder_num_hidden,
                                    hidden_size = self.decoder_num_hidden,
                                    time_step = self.T
                                    )
        self.lstm_layer = nn.LSTM(
            input_size=1,
            hidden_size=self.decoder_num_hidden
        )
        self.fc = nn.Linear(self.encoder_num_hidden + 1, 1)
        self.fc.weight.data.normal_()

    def forward(self, X_encoded, y_prev):
        d_n = self._init_states(X_encoded)
        c_n = self._init_states(X_encoded)
        
        X_encoded = self.CA(X_encoded)
        params = list(self.CA.named_parameters())#get the index by debuging
        #print(params[2][0])#name
        #print(params[2][1].data)

        for t in range(self.T):
            
            y_tilde = self.fc(
                torch.cat((X_encoded[:,t,:], y_prev[:, t].unsqueeze(1)), dim=1))
            _, final_states = self.lstm_layer(
                y_tilde.unsqueeze(0), (d_n, c_n))

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

        self.loss_func = nn.BCELoss()

    def forward(self, x, y):
        out1 = self.Encoder(x)
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
