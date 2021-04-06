#!/usr/bin/env python
# encoding: utf-8
import os
from numpy.core.fromnumeric import trace
import torch
import pickle
import random
import argparse
import matplotlib
import numpy as np
import pandas as pd
from torch import nn
matplotlib.use('Agg')
from torch import optim
'''from model import AttnEncoder
from model import AttnDecoder
from model import SelfAttention
from model import AttnAlphaStock'''
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

class AttnEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, time_step, drop_ratio):
        super(AttnEncoder, self).__init__()
        self.input_size = input_size   # 特征的数量
        self.hidden_size = hidden_size
        self.T = time_step

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        # nn.Linear: y=xA+b
        self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=self.T)
        self.attn2 = nn.Linear(in_features=self.T, out_features=self.T)
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(in_features=self.T, out_features=1)
        #self.attn = nn.Sequential(attn1, attn2, nn.Tanh(), attn3)

        self.drop = nn.Dropout(p=drop_ratio/100.)


    def forward(self, driving_x):
        # driving_x: batch_size, T, input_size
        driving_x = self.drop(driving_x)
        batch_size = driving_x.size(0)
        # batch_size * time_step * hidden_size
        code = self.init_variable(batch_size, self.T, self.hidden_size)  # 保存lstm的输出
        # initialize hidden state (output)
        h = self.init_variable(1, batch_size, self.hidden_size)
        # initialize cell state
        s = self.init_variable(1, batch_size, self.hidden_size)
        for t in range(self.T):
            # batch_size * input_size * (2 * hidden_size)
            x = torch.cat((self.embedding_hidden(h), self.embedding_hidden(s)), 2)
            # batch_size * input_size * T
            z1 = self.attn1(x)
            # batch_size * input_size * T
            z2 = self.attn2(driving_x.permute(0, 2, 1))
            # batch_size * input_size * T
            x = z1 + z2
            # batch_size * input_size * 1
            z3 = self.attn3(self.tanh(x))
            if batch_size > 1:
                # batch_size * input_size
                attn_w = F.softmax(z3.view(batch_size, self.input_size), dim=1)
            else:
                attn_w = self.init_variable(batch_size, self.input_size) + 1
            # batch_size * input_size (元素点乘)
            weighted_x = torch.mul(attn_w, driving_x[:, t, :])
            # torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度
            # torch.unsqueeze()这个函数主要是对数据维度进行扩充。给指定位置加上维数为一的维度
            # 1 * B * input_size （其中1是序列长度，此处都是对长度为1的进行多次迭代）
            _, states = self.lstm(weighted_x.unsqueeze(0), (h, s))
            '''
            Inputs: input, (h_0, c_0)
            input of shape (time_step, batch, input_size)  # 1 * batch_size, input_size
            '''
            h = states[0]  # 1, batch_size, hidden_size
            s = states[1]

            # encoding result
            # batch_size * time_step * encoder_hidden_size
            code[:, t, :] = h

        # code = self.drop(code)

        return code

    def init_variable(self, *args):
        zero_tensor = torch.zeros(args)
        if torch.cuda.is_available():
            zero_tensor = zero_tensor.cuda()
        return Variable(zero_tensor)

    def embedding_hidden(self, x):
        return x.repeat(self.input_size, 1, 1).permute(1, 0, 2)

class AttnDecoder(nn.Module):
    def __init__(self, code_hidden_size, hidden_size, time_step):
        super(AttnDecoder, self).__init__()
        self.code_hidden_size = code_hidden_size
        self.hidden_size = hidden_size
        self.T = time_step

        self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=code_hidden_size)
        self.attn2 = nn.Linear(in_features=code_hidden_size, out_features=code_hidden_size)
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(in_features=code_hidden_size, out_features=1)
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size)
        self.tilde = nn.Linear(in_features=self.code_hidden_size + 1, out_features=1)

    def forward(self, h, y_seq):
        # h: batch_size * time_step * layer1_hidden_size
        # y_seq: batch_size * time_step
        batch_size = h.size(0)
        d = self.init_variable(1, batch_size, self.hidden_size)
        s = self.init_variable(1, batch_size, self.hidden_size)
        ct = self.init_variable(batch_size, self.hidden_size)

        for t in range(self.T):
            # batch_size * time_step * (2 * decoder_hidden_size)
            x = torch.cat((self.embedding_hidden(d), self.embedding_hidden(s)), 2)
            # batch_size * time_step * layer1_hidden_size
            z1 = self.attn1(x)
            # batch_size * time_step * layer1_hidden_size
            z2 = self.attn2(h)
            x = z1 + z2
            # batch_size * time_step * 1
            z3 = self.attn3(self.tanh(x))
            if batch_size > 1:
                # batch_size * time_step
                beta_t = F.softmax(z3.view(batch_size, -1), dim=1)
            else:
                beta_t = self.init_variable(batch_size, self.code_hidden_size) + 1
            # batch_size * layer1_hidden_size
            # batch matrix mul: 第一个维度是batch_size，然后剩下的当普通矩阵乘
            ct = torch.bmm(beta_t.unsqueeze(1), h).squeeze(1)  # (b, 1, T) * (b, T, m)
            # batch_size * (1 + layer1_hidden_size)
            yc = torch.cat((y_seq[:, t].unsqueeze(1), ct), dim=1)
            # batch_size * (1 + layer1_hidden_size)
            y_tilde = self.tilde(yc)
            _, states = self.lstm(y_tilde.unsqueeze(0), (d, s))
            d = states[0]  # 1, batch_size, hidden_size
            s = states[1]

        # return torch.cat((d.squeese(0), ct), dim=1)
        return d.squeeze(0)

    def init_variable(self, *args):
        zero_tensor = torch.zeros(args)
        if torch.cuda.is_available():
            zero_tensor = zero_tensor.cuda()
        return Variable(zero_tensor)

    def embedding_hidden(self, x):
        # 重复T变，是因为这一层是对时间序列做attention，因此是T个值做softmax
        return x.repeat(self.T, 1, 1).permute(1, 0, 2)

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

random.seed(0)
#PWD = os.path.dirname(os.path.realpath(__file__))

def load_train(years):
    train = None
    for y in years:
        with open(os.path.join('./v1', 'v1_T20_yb1_%s.pickle' % (y)), 'rb') as fp:
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


class Trainer:
    def __init__(self, time_step, hidden_size, lr, train_zscore=False, batch_size=256, drop_ratio=0, split=20):
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.learning_rate = lr
        self.train_zscore = train_zscore
        self.batch_size = batch_size
        self.drop_ratio = 0
        self.validation_ratio = split

        #dataset = 'xyt%s_T%s_yb1' % ('_tvz' if train_zscore else '', time_step)
        self.dataset = load_dataset([2010,2011,2012,2013,2014,2015,2016,2017,2018],
                        [123,456,789,1012])
        feature_size = self.dataset['train']['xs'][0].shape[1]
        print(feature_size)
        self.layer1 = AttnEncoder(input_size=feature_size, hidden_size=hidden_size, time_step=time_step, drop_ratio=drop_ratio)
        self.layer2 = AttnDecoder(code_hidden_size=hidden_size, hidden_size=hidden_size, time_step=time_step)
        self.layer3 = SelfAttention(last_hidden_size=hidden_size, hidden_size=hidden_size)

        if torch.cuda.is_available():
            self.layer1 = self.layer1.cuda()
            self.layer2 = self.layer2.cuda()
            self.layer3 = self.layer3.cuda()
        self.layer1_optim = optim.Adam(self.layer1.parameters(), lr)
        self.layer2_optim = optim.Adam(self.layer2.parameters(), lr)
        self.layer3_optim = optim.Adam(self.layer3.parameters(), lr)

        self.loss_func = nn.BCELoss()
        self.T = time_step
        self.test_acc_max = 0

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


    def train_minibatch(self, num_epochs):
        xs = self.dataset['train']['xs']
        ys = self.dataset['train']['ys']
        ts = self.dataset['train']['ts']
        test_xs = self.dataset['test']['xs']
        test_ys = self.dataset['test']['ys']
        test_ts = self.dataset['test']['ts']
        train_size = len(xs)
        acc_max_diff, acc_max = 0, 0
        result = {}
        for epoch in range(num_epochs):
            i = 0
            loss_sum = 0
            y_predict = []
            y = []
            # 随机选n%做validation数据
            validation_index = random.sample(range(train_size), int(train_size*self.validation_ratio/100.))
            validation_mask = np.array([False] * train_size)
            validation_mask[validation_index] = True
            validation_xs = xs[validation_mask]
            validation_ys = ys[validation_mask]
            validation_ts = ts[validation_mask]
            train_xs = xs[~validation_mask]
            train_ys = ys[~validation_mask]
            train_ts = ts[~validation_mask]
            for i in range(0, len(train_xs), self.batch_size):
                batch = {'x': train_xs[i:i+self.batch_size],
                         'y': train_ys[i:i+self.batch_size],
                         't': train_ts[i:i+self.batch_size]}
                self.layer1.zero_grad()
                self.layer2.zero_grad()
                self.layer3.zero_grad()
                var_x = self.to_variable(batch['x'])
                var_y = self.to_variable(batch['y'])
                var_t = self.to_variable(batch['t'])
                out1 = self.layer1(var_x)
                out2 = self.layer2(out1, var_y)
                out3 = self.layer3(out2)
                #+0是为了True False 变成数字 0 和 1
                out4 = (out3 >= 0.5) + 0

                y_predict.extend(out4.data.cpu().numpy())
                y.extend(batch['t'])
                loss = self.loss_func(out3, var_t)
                loss.backward()
                self.layer1_optim.step()
                self.layer2_optim.step()
                self.layer3_optim.step()
                # print('[%d], loss is %f' % (epoch, 10000 * loss.item()))
                loss_sum += loss.data.item()
            print('\n--------------------------------------------------------------')
            print('epoch [%d] finished, the average loss is %f' % (epoch, loss_sum))

            print('train:')
            accuracy, precision, recall, f1 = self.metrics(y_predict, y)
            print('Accuract:%.4f\tPrecision:%.4f\tRecall:%.4f\tF1:%.4f' % (accuracy, precision, recall, f1))
            train_accuracy = accuracy

            print('validation:')
            validation_random = max([sum(validation_ts == r) for r in [0, 1]]) * 1. / len(validation_ts)
            accuracy, precision, recall, f1 = self.validation(validation_xs, validation_ys, validation_ts)
            acc_max_diff = max(acc_max_diff, accuracy-validation_random)
            print('Accuract:%.4f\tPrecision:%.4f\tRecall:%.4f\tF1:%.4f' % (accuracy, precision, recall, f1))
            print('Random:%.4f\tMaxAccDiff:%.6f' % (validation_random, acc_max_diff))
            validation_accuracy = accuracy

            print('test:')
            accuracy, precision, recall, f1 = self.validation(test_xs, test_ys, test_ts)
            acc_max = max(acc_max, accuracy)
            print('Accuracy:%.4f\tPrecision:%.4f\tRecall:%.4f\tF1:%.4f' % (accuracy, precision, recall, f1))
            print('MaxAcc:%.4f' % acc_max)
            # 存结果
            result[epoch] = {'loss': loss_sum,
                             'train_accuracy': train_accuracy,
                             'validation_accuracy': validation_accuracy,
                             'validation_maxdiff': acc_max_diff,
                             'test_accuarcy': accuracy,
                             'epoch': epoch
                             }

            continue
            

        '''r_name = os.path.join(PWD, 'results', 'xyt%s_tz_rtv2_b%s_hs%s_ts%s_dr%s_tv%s.csv' \
                              % (self.feature_size,
                                 self.batch_size,
                                 self.hidden_size,
                                 self.time_step,
                                 self.drop_ratio,
                                 self.validation_ratio)
                              )
        pd.DataFrame(result).T.to_excel(r_name)'''


    def validation(self, xs, ys, ts):
        y_predict = []
        y = []
        for i in range(0, len(xs), self.batch_size):
            batch = {'x': xs[i:i+self.batch_size],
                     'y': ys[i:i+self.batch_size],
                     't': ts[i:i+self.batch_size]}
            var_x = self.to_variable(batch['x'])
            var_y = self.to_variable(batch['y'])
            var_t = self.to_variable(batch['t'])
            out1 = self.layer1(var_x)
            out2 = self.layer2(out1, var_y)
            out3 = self.layer3(out2)
            out4 = (out3 >= 0.5) + 0
            y_predict.extend(out4.data.cpu().numpy())
            y.extend(batch['t'])
        return self.metrics(y_predict, y)


    def to_variable(self, x):
        if torch.cuda.is_available():
            return Variable(torch.from_numpy(x).float()).cuda()
        else:
            return Variable(torch.from_numpy(x).float())


def getArgParser():
    parser = argparse.ArgumentParser(description='Train the dual-stage attention-based model on stock')
    parser.add_argument(
        '-e', '--epoch', type=int, default=1,
        help='the number of epochs')
    parser.add_argument(
        '-b', '--batch', type=int, default=256,
        help='the mini-batch size')
    parser.add_argument(
        '-ts', '--timestep', type=int, default=10,
        help='the length of time_step')
    parser.add_argument(
        '-hs', '--hiddensize', type=int, default=8,
        help='the length of hidden size')
    parser.add_argument(
        '-dr', '--dropratio', type=int, default=30,
        help='the ratio of drop')
    parser.add_argument(
        '-tz', '--trainzscore', action='store_true',
        help='the way for z-score')
    parser.add_argument(
        '-s', '--split', type=int, default=20,
        help='the split ratio of validation set')
    parser.add_argument(
        '-i', '--interval', type=int, default=1,
        help='save models every interval epoch')
    parser.add_argument(
        '-l', '--lrate', type=float, default=0.01,
        help='learning rate')
    parser.add_argument(
        '-t', '--test', action='store_true',
        help='train or test')
    parser.add_argument(
        '-m', '--model', type=str, default='',
        help='the model name(after encoder/decoder)'
    )
    return parser


if __name__ == '__main__':
    args = getArgParser().parse_args()
    num_epochs = args.epoch
    batch_size = args.batch
    time_step = args.timestep
    hidden_size = args.hiddensize
    drop_ratio = args.dropratio
    train_zscore = args.trainzscore
    split = args.split
    interval = args.interval
    lr = args.lrate
    test = args.test
    mname = args.model

    print(time_step, hidden_size, lr, train_zscore, batch_size, drop_ratio, split)
    trainer = Trainer(time_step, hidden_size, lr, train_zscore, batch_size, drop_ratio, split)
    trainer.train_minibatch(num_epochs)
