#!/usr/bin/env python
# encoding: utf-8
'''
将Encoder 和 Decoder 模型部分换成自己的模型 测试是否效果不变
'''
import os
import math
import pickle
import random
import argparse
import pandas as pd
import numpy as np
from numpy.core.fromnumeric import trace

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

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
            num_layers = 1
        )

        self.encoder_attn = nn.Sequential(
            nn.Linear(2 * self.encoder_num_hidden + self.T, self.T),
            nn.Tanh(),
            nn.Linear(self.T, 1)
        )

    def forward(self, X):
        """forward.
        Args:
            X: input data
        """
        X_encoded = Variable(X.data.new(
            X.size(0), self.T, self.encoder_num_hidden).zero_())
        h_n = self._init_states(X)
        s_n = self._init_states(X)
        for t in range(self.T):
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim=2)
            x = self.encoder_attn(
                x.view(-1, self.encoder_num_hidden * 2 + self.T ))
            alpha = F.softmax(x.view(-1, self.input_size),1)
            x_tilde = torch.mul(alpha, X[:, t, :])
            self.encoder_lstm.flatten_parameters()

            _, final_state = self.encoder_lstm(x_tilde.unsqueeze(0), (h_n, s_n))
            h_n = final_state[0]
            s_n = final_state[1]

            X_encoded[:, t, :] = h_n
        return X_encoded

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X

        Returns:
            initial_hidden_states
        """
        # https://pytorch.org/docs/master/nn.html?#lstm
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
            nn.Linear(2 * self.decoder_num_hidden + self.encoder_num_hidden, self.encoder_num_hidden),
            nn.Tanh(),
            nn.Linear(self.encoder_num_hidden, 1)
        )
        self.lstm_layer = nn.LSTM(
            input_size=1,
            hidden_size=self.decoder_num_hidden
        )
        self.fc = nn.Linear(self.encoder_num_hidden + 1, 1)
        self.fc_final = nn.Linear(self.decoder_num_hidden + self.encoder_num_hidden, 3)
        self.softmax=nn.Softmax()
        self.fc.weight.data.normal_()

    def forward(self, X_encoded, y_prev):
        """forward."""
        d_n = self._init_states(X_encoded)
        c_n = self._init_states(X_encoded)

        for t in range(self.T):
            x = torch.cat((d_n.repeat(self.T , 1, 1).permute(1, 0, 2),
                           c_n.repeat(self.T , 1, 1).permute(1, 0, 2),
                           X_encoded), dim=2)
            beta = F.softmax(self.attn_layer(
                x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T),1)

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
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X
        Returns:
            initial_hidden_states

        """
        return Variable(X.data.new(1, X.size(0), self.decoder_num_hidden).zero_())

class SelfAttention(nn.Module):
    def __init__(self, last_hidden_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.last_hidden_size = last_hidden_size
        self.hidden_size = hidden_size

        self.wq = nn.Linear(in_features=last_hidden_size, out_features=hidden_size, bias=False)
        self.wk = nn.Linear(in_features=last_hidden_size, out_features=hidden_size, bias=False)
        self.wv = nn.Linear(in_features=last_hidden_size, out_features=hidden_size, bias=False)

        # 输出
        self.ln = nn.Linear(in_features=hidden_size, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        # h: batch_size * last_hidden_size


        #q k v  batch_size * hidden_size
        q = self.wq(h)
        k = self.wk(h)
        v = self.wv(h)

        dk = q.size(-1)
        z = torch.mm(q, k.t()) / math.sqrt(dk)  # (b, hidden_size) * (hidden_size, b) ==> (b, b)
        beta = F.softmax(z, dim=1)
        st = torch.mm(beta, v)  # (b, b) * (b, hidden_size) ==> (b, hidden_size)

        # b * 1
        y_res = self.ln(st)
        # y_res: (batch_size, 1)
        y_res = self.sigmoid(y_res.squeeze(1))
        return y_res

class Darnn_selfattention(nn.Module):
    def __init__(self,input_size, T,
                 encoder_num_hidden,
                 decoder_num_hidden,
                 learning_rate):
        super(Darnn_selfattention,self).__init__()
        self.input_size = input_size
        self.T = T
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden
        self.learning_rate = learning_rate
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.Encoder = Encoder(input_size=input_size,hidden_size=encoder_num_hidden,
                                    time_step=T,drop_ratio=0)
        self.Decoder = Decoder(encoder_num_hidden=encoder_num_hidden,
                                    hidden_size=decoder_num_hidden,
                                    time_step=T)
        self.attention = SelfAttention(last_hidden_size=encoder_num_hidden,hidden_size=decoder_num_hidden)
        
        if torch.cuda.is_available():
            self.Encoder = self.Encoder.cuda()
            self.Decoder = self.Decoder.cuda()
            self.attention = self.attention.cuda()
            
        self.Encoder_optim = optim.Adam(params=filter(lambda p: p.requires_grad,self.Encoder.parameters()),
                                        lr=self.learning_rate)
        self.Decoder_optim = optim.Adam(params=filter(lambda p: p.requires_grad,self.Decoder.parameters()),
                                        lr= self.learning_rate)
        self.attention_optim = optim.Adam(params=filter(lambda p: p.requires_grad,self.attention.parameters()),
                                          lr=self.learning_rate)
        
        self.loss_func = nn.BCELoss()
        #self.loss_func = nn.CrossEntropyLoss()
    def forward(self,x,y):
        out1 = self.Encoder(x)
        out2 = self.Decoder(out1, y)
        out3 = self.attention(out2)
        return out3





random.seed(0)

def load_train(years):
    train = None
    for y in years:
        with open(os.path.join('/home/xinkun/darnn/v1', 'v1_T20_yb1_%s.pickle' % (y)), 'rb') as fp:
            dataset = pickle.load(fp)

        if train is None:
            train = {}
            train['xs'] = dataset['x']
            train['ys'] = dataset['y']
            train['ts'] = dataset['t']
            #train = dataset
            continue

        train['xs'] = np.append(train['xs'], dataset['x'], axis=0)
        train['ys'] = np.append(train['ys'], dataset['y'], axis=0)
        train['ts'] = np.append(train['ts'], dataset['t'], axis=0)
        #train['ci'] = np.append(train['ci'], dataset['ci'], axis=0)

    return train
def load_dataset(train_years,test_years):
    dataset = {}
    dataset['train'] = load_train(train_years)
    dataset['test'] = load_train(test_years)
    return dataset


class dataset(Dataset):
    def __init__(self,data):
        self.data=data        
        
    def __getitem__(self,index):
        return self.data['x'][index], self.data['y'][index], self.data['t'][index]#返回的目标是0 ,1
    
    def __len__(self):
        return len(self.data['t'])




class Trainer:
    def __init__(self, time_step, hidden_size, lr, batch_size=256, drop_ratio=0, split=20):
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.learning_rate = lr
        #self.train_zscore = train_zscore
        self.batch_size = batch_size
        self.drop_ratio = 0
        self.validation_ratio = split

        self.dataset = load_dataset([2010,2011,2012,2013,2014,2015,2016,2017,2018],
                        [123,456,789,1012])
        self.feature_size = self.dataset['train']['xs'][0].shape[1]
        print(self.feature_size)
        
        self.model = Darnn_selfattention(
                            input_size = self.feature_size, 
                            T = time_step,
                            encoder_num_hidden = hidden_size,
                            decoder_num_hidden = hidden_size,
                            learning_rate = lr)
                    
        self.acc_train_max_diff = 0
        self.acc_val_max_diff = 0
        self.acc_test_max_diff = 0

    def train_minibatch(self, num_epochs):
        xs = self.dataset['train']['xs']
        ys = self.dataset['train']['ys']
        ts = self.dataset['train']['ts']

        test_data ={}
        test_data['x'] = self.dataset['test']['xs']
        test_data['y'] = self.dataset['test']['ys']
        test_data['t'] = self.dataset['test']['ts']
        TestDataloader = DataLoader(dataset(test_data), batch_size=self.batch_size, shuffle=False)

        train_size = len(ts)
        result = {}

        for epoch in range(num_epochs):
            print('====epoch:'+str(epoch)+' ====>正在训练')
            print('--------------------------------------------------------------')
            
            # 随机选n%做validation数据
            validation_index = random.sample(range(train_size), int(train_size*self.validation_ratio/100.))
            validation_mask = np.array([False] * train_size)
            validation_mask[validation_index] = True
            
            #验证集
            val_data = {}
            val_data['x'] = xs[validation_mask]
            val_data['y'] = ys[validation_mask]
            val_data['t'] = ts[validation_mask]
            ValDataloader = DataLoader(dataset(val_data), batch_size=self.batch_size, shuffle=False)
            #训练集
            train_data = {}
            train_data['x'] = xs[~validation_mask]
            train_data['y'] = ys[~validation_mask]
            train_data['t'] = ts[~validation_mask]
            TrainDataloader = DataLoader(dataset(train_data), batch_size=self.batch_size, shuffle=False)
            
            #-------------------------------------------------------------------------------
            print('\033[1;34m Train: \033[0m')
            loss_sum = 0
            t_pred = []
            t_ori = []
            self.model.train()
            for _,sample in enumerate(TrainDataloader):

                self.model.Encoder_optim.zero_grad()
                self.model.Decoder_optim.zero_grad()
                self.model.attention_optim.zero_grad()

                var_x = self.to_variable(sample[0].numpy())
                var_y = self.to_variable(sample[1].numpy()) 
                var_t = self.to_variable(sample[2].numpy())

                out1 = self.model.Encoder(var_x)
                out2 = self.model.Decoder(out1, var_y)
                out3 = self.model.attention(out2)
                out4 = (out3 >= 0.5) + 0

                t_pred.extend(out4.data.cpu().numpy())
                t_ori.extend(sample[2].numpy())

                loss = self.model.loss_func(out3, var_t)
                loss.backward()

                self.model.Encoder_optim.step()
                self.model.Decoder_optim.step()
                self.model.attention_optim.step()

                loss_sum += loss.data.item()

            train_accuracy, precision, recall, f1 = self.metrics(t_pred,t_ori)
            train_random = self.rand_acc(t_ori)
            self.acc_train_max_diff = max(self.acc_train_max_diff, train_accuracy-train_random)
            print('第 \033[1;34m %d \033[0m 轮的训练集正确率为:\033[1;32m %.4f \033[0m epoch_Loss 为: \033[1;32m %.4f \033[0m' %
                (epoch,train_accuracy,loss_sum))
            print('\033[1;31m Accuracy:%.4f Precision:%.4f Recall:%.4f F1:%.4f \033[0m' % (train_accuracy, precision, recall, f1))
            print('\033[1;31m Random:%.4f\tMaxAccDiff:%.6f \033[0m' % (train_random, self.acc_train_max_diff))

            #-------------------------------------------------------------------------------
            print('\033[1;34m Valid: \033[0m')
            with torch.no_grad():
                t_pred = []
                t_ori = []
                self.model.eval()
                for _,sample in enumerate(ValDataloader):

                    var_x = self.to_variable(sample[0].numpy())
                    var_y = self.to_variable(sample[1].numpy()) 
                    var_t = self.to_variable(sample[2].numpy())

                    out1 = self.model.Encoder(var_x)
                    out2 = self.model.Decoder(out1, var_y)
                    out3 = self.model.attention(out2)
                    out4 = (out3 >= 0.5) + 0

                    t_pred.extend(out4.data.cpu().numpy())
                    t_ori.extend(sample[2].numpy())

            validation_accuracy, precision, recall, f1 = self.metrics(t_pred,t_ori)
            validation_random = self.rand_acc(t_ori)
            self.acc_val_max_diff = max(self.acc_val_max_diff, validation_accuracy-validation_random)
            print('\033[1;31m Accuracy:%.4f Precision:%.4f Recall:%.4f F1:%.4f \033[0m' % (validation_accuracy, precision, recall, f1))
            print('\033[1;31m Random:%.4f\tMaxAccDiff:%.6f \033[0m' % (validation_random, self.acc_val_max_diff))

            
            #-------------------------------------------------------------------------------
            print('\033[1;34m Test: \033[0m')
            with torch.no_grad():
                t_pred = []
                t_ori = []
                self.model.eval()
                
                for _,sample in enumerate(TestDataloader):

                    var_x = self.to_variable(sample[0].numpy())
                    var_y = self.to_variable(sample[1].numpy()) 
                    var_t = self.to_variable(sample[2].numpy())

                    out1 = self.model.Encoder(var_x)
                    out2 = self.model.Decoder(out1, var_y)
                    out3 = self.model.attention(out2)
                    out4 = (out3 >= 0.5) + 0

                    t_pred.extend(out4.data.cpu().numpy())
                    t_ori.extend(sample[2].numpy())

            test_accuracy, precision, recall, f1 = self.metrics(t_pred,t_ori)
            test_random = self.rand_acc(t_ori)
            self.acc_test_max_diff = max(self.acc_test_max_diff, test_accuracy-test_random)
            print('\033[1;31m Accuracy:%.4f Precision:%.4f Recall:%.4f F1:%.4f \033[0m' % (test_accuracy, precision, recall, f1))
            print('\033[1;31m Random:%.4f\ttestMaxAccDiff:%.6f \033[0m' % (test_random, self.acc_test_max_diff))
            
            #-------------------------------------------------------------------------------
            # 存结果
            result[epoch] = {'loss': loss_sum,
                             'train_accuracy': train_accuracy,
                             'validation_accuracy': validation_accuracy,
                             'validation_maxdiff': self.acc_val_max_diff,
                             'test_accuarcy': test_accuracy,
                             'epoch': epoch
                             }

            continue
            

        r_name = os.path.join('result/',__file__[:-3], 'xyt%s_tz_rtv2_b%s_hs%s_ts%s_dr%s_tv%s.csv' \
                              % (self.feature_size,
                                 self.batch_size,
                                 self.hidden_size,
                                 self.time_step,
                                 self.drop_ratio,
                                 self.validation_ratio)
                              )
        pd.DataFrame(result).T.to_excel(r_name)
        torch.save(self.model, '../pth/'+__file__[:-3]+'.pth')

    def to_variable(self, x):
        if torch.cuda.is_available():
            return Variable(torch.from_numpy(x).float()).cuda()
        else:
            return Variable(torch.from_numpy(x).float())
    
    def metrics(self, results, ori_y):
        '''
            predict: 预测y
            ori_y: 原始的01标签
        '''
        accuracy = accuracy_score(ori_y, results)
        precision = precision_score(ori_y, results, labels=[1], average=None)[0]
        recall = recall_score(ori_y, results, labels=[1], average=None)[0]
        f1 = f1_score(ori_y, results, labels=[1], average=None)[0]
        return accuracy, precision, recall, f1

    def rand_acc(self,t_ori):
        return max([np.sum(np.array(t_ori) == r) for r in [0, 1]]) * 1. / len(t_ori)

    def update_lr(self):
        for param_group in self.model.Encoder_optim.param_groups:
            param_group['lr'] = param_group['lr'] * 0.9
        for param_group in self.model.Decoder_optim.param_groups:
            param_group['lr'] = param_group['lr'] * 0.9
        for param_group in self.model.attention_optim.param_groups:
            param_group['lr'] = param_group['lr'] * 0.9



def getArgParser():
    parser = argparse.ArgumentParser(description='Train the dual-stage attention-based model on stock')
    parser.add_argument(
        '-e', '--epoch', type=int, default=1500,
        help='the number of epochs')
    parser.add_argument(
        '-b', '--batch', type=int, default=256,
        help='the mini-batch size')
    parser.add_argument(
        '-ts', '--timestep', type=int, default=20,
        help='the length of time_step')
    parser.add_argument(
        '-hs', '--hiddensize', type=int, default=32,
        help='the length of hidden size')
    parser.add_argument(
        '-dr', '--dropratio', type=int, default=0,
        help='the ratio of drop')
    
    parser.add_argument(
        '-s', '--split', type=int, default=30,
        help='the split ratio of validation set')
    
    parser.add_argument(
        '-l', '--lrate', type=float, default=0.001,
        help='learning rate')
    return parser


if __name__ == '__main__':
    args = getArgParser().parse_args()
    num_epochs = args.epoch
    batch_size = args.batch
    time_step = args.timestep
    hidden_size = args.hiddensize
    drop_ratio = args.dropratio
    split = args.split
    lr = args.lrate

    print(time_step, hidden_size, lr, batch_size, drop_ratio, split)
    trainer = Trainer(time_step, hidden_size, lr, batch_size, drop_ratio, split)
    trainer.train_minibatch(num_epochs)
