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

    def forward(self, X):
        X_encoded = Variable(X.data.new(
            X.size(0), self.T, self.encoder_num_hidden).zero_())
        h_n = self._init_states(X)
        s_n = self._init_states(X)
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

    def forward(self, X_encoded, y_prev):
        d_n = self._init_states(X_encoded)
        c_n = self._init_states(X_encoded)

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


random.seed(0)

if __name__ == '__main__':
    random.seed(0)
    parser = argparse.ArgumentParser(description='输入参数yml文件')
    parser.add_argument('-opt', type=str, default='./train.yml',
                        help='Path to option YAML file.')
    args = parser.parse_args()
    opt = get_opt(args.opt)[__file__[:-3]]
    Train = Trainer(Darnn_selfattention,opt['model_conf'], opt['data_conf'], opt['train_conf'],__file__[:-3])
    Train.run()