#!/usr/bin/env python
# encoding: utf-8
'''
DARNN+self-attention 中attention中替换为selfattention  并且使用独立selfattention 带dropout
最后out3 使用resdual结构
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
from Base_model import SelfAttention

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

        self.encoder_attn = nn.Sequential(
            nn.Linear(2 * self.encoder_num_hidden + self.T, self.T),
            nn.Tanh(),
            nn.Linear(self.T, 1)
        )
        self.SelfAttention = SelfAttention(input_size = self.T,
                                            output_size = self.T,
                                            drop_ratio = 0)
        
    def forward(self, X):
        X_encoded = Variable(X.data.new(
            X.size(0), self.T, self.encoder_num_hidden).zero_())
        h_n = self._init_states(X)
        s_n = self._init_states(X)
        X = self.SelfAttention(X.permute(0,2,1)).permute(0,2,1)

        for t in range(self.T):
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim=2)
            x = self.encoder_attn(
                x.view(-1, self.encoder_num_hidden * 2 + self.T))
            alpha = F.softmax(x.view(-1, self.input_size), 1)
            x_tilde = torch.mul(alpha, X[:, t, :])
            self.encoder_lstm.flatten_parameters()

            _, final_state = self.encoder_lstm(
                x_tilde.unsqueeze(0), (h_n, s_n))
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

        self.attn_layer = nn.Sequential(
            nn.Linear(2 * self.decoder_num_hidden +
                      self.encoder_num_hidden, self.encoder_num_hidden),
            nn.Tanh(),
            nn.Linear(self.encoder_num_hidden, 1)
        )
        self.lstm_layer = nn.LSTM(
            input_size=1,
            hidden_size=self.decoder_num_hidden
        )
        self.fc = nn.Linear(self.encoder_num_hidden + 1, 1)
        
        self.fc.weight.data.normal_()
        self.SelfAttention = SelfAttention(input_size = encoder_num_hidden,
                                            output_size = encoder_num_hidden,
                                            drop_ratio = 0)
        
    def forward(self, X_encoded, y_prev):
        d_n = self._init_states(X_encoded)
        c_n = self._init_states(X_encoded)

        X_encoded = self.SelfAttention(X_encoded)

        for t in range(self.T):
            
            x = torch.cat((d_n.repeat(self.T, 1, 1).permute(1, 0, 2),
                           c_n.repeat(self.T, 1, 1).permute(1, 0, 2),
                           X_encoded), dim=2)
            beta = F.softmax(self.attn_layer(
                x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T), 1)

            context = torch.bmm(beta.unsqueeze(1), X_encoded)[:, 0, :]
            
            if t < self.T:
                y_tilde = self.fc(
                    torch.cat((context, y_prev[:, t].unsqueeze(1)), dim=1))
                _, final_states = self.lstm_layer(
                    y_tilde.unsqueeze(0), (d_n, c_n))

                d_n = final_states[0]  # 1 * batch_size * decoder_num_hidden
                c_n = final_states[1]  # 1 * batch_size * decoder_num_hidden
        return d_n.squeeze(0)

    def _init_states(self, X):
        return Variable(X.data.new(1, X.size(0), self.decoder_num_hidden).zero_())



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
            input_size=encoder_num_hidden, output_size=decoder_num_hidden, drop_ratio=drop_ratio)
        # 输出
        self.last_fc = nn.Sequential(
            
            nn.Linear(in_features=decoder_num_hidden, out_features=1),
            nn.Sigmoid()
        )
        #self.scale = Parameter(torch.randn(1), requires_grad=True)
        self.loss_func = nn.BCELoss()

    def forward(self, x, y):
        #x:B*T*F
        out1 = self.Encoder(x)#out1:B*T*EN
        
        out2 = self.Decoder(out1, y)#out2:B*DE
        
        out3 = self.attention(out2.unsqueeze(0)).squeeze(0)#out3:B*DEhidden
        out3 = out3 + out2
        out4 = self.last_fc(out3)
        return out4.squeeze(1)



if __name__ == '__main__':
    random.seed(0)
    parser = argparse.ArgumentParser(description='输入参数yml文件')
    parser.add_argument('-opt', type=str, default='./train.yml',
                        help='Path to option YAML file.')
    args = parser.parse_args()
    opt = get_opt(args.opt)[__file__[:-3]]
    Train = Trainer(Darnn_selfattention,opt['model_conf'], opt['data_conf'], opt['train_conf'],__file__[:-3])
    Train.run()
