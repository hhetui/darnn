#!/usr/bin/env python
# encoding: utf-8
'''
换成自己程序生成的程序，看准确率是否改变，如果不变，重点开始改模型跑实验
'''
import os
import math
import pickle
import random
import argparse
import pandas as pd
from collections import defaultdict
import numpy as np
from numpy.core.fromnumeric import trace

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

from utils import parse


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
        self.fc_final = nn.Linear(
            self.decoder_num_hidden + self.encoder_num_hidden, 3)
        self.softmax = nn.Softmax()
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


def load_dataset(train_years, test_years):
    def load_pickle(years):
        data_dic = None
        for y in years:
            with open(os.path.join(Datapath, 'v1_T20_yb1_%s.pickle' % (y)), 'rb') as fp:
                dataset = pickle.load(fp)

            if data_dic is None:
                data_dic = {}
                data_dic['x'] = dataset['x']
                data_dic['y'] = dataset['y']
                data_dic['t'] = dataset['t']

            data_dic['x'] = np.append(data_dic['x'], dataset['x'], axis=0)
            data_dic['y'] = np.append(data_dic['y'], dataset['y'], axis=0)
            data_dic['t'] = np.append(data_dic['t'], dataset['t'], axis=0)

        return data_dic
    dataset = {}
    dataset['train'] = load_pickle(train_years)
    dataset['test'] = load_pickle(test_years)

    return dataset


class dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # 返回的目标是0 ,1
        return self.data['x'][index], self.data['y'][index], self.data['t'][index]

    def __len__(self):
        return len(self.data['t'])


class Trainer:
    def __init__(self, model_conf, data_conf, train_conf):
        self.model_conf = model_conf
        self.data_conf = data_conf
        self.train_conf = train_conf

        self.result_path = os.path.join('../result/', __file__[:-3])
        self.Data = load_dataset(
            self.data_conf['train_list'], self.data_conf['test_list'])
        self.csv_name = os.path.join(self.result_path, __file__[:-3]+'.csv')
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.load_checkpoint()

    def load_checkpoint(self):
        self.model = Darnn_selfattention(**self.model_conf)
        self.optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.train_conf['learning_rate'])
        if self.train_conf['resume'] and os.path.exists(self.result_path):
            checkpoint = torch.load(os.path.join(self.result_path,
                                                 "last.pt"), map_location="cpu")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optim_state_dict'])

            self.cur_epoch = checkpoint['epoch']
            self.acc_train_max_diff = checkpoint['acc_train_max_diff']
            self.acc_val_max_diff = checkpoint['acc_val_max_diff']
            self.acc_test_max_diff = checkpoint['acc_test_max_diff']
            self.result = pd.read_csv(self.csv_name).to_dict(orient='list')
        else:
            self.cur_epoch = 0
            self.acc_train_max_diff = 0
            self.acc_val_max_diff = 0
            self.acc_test_max_diff = 0
            self.result = defaultdict(list)
        self.model = self.model.to(self.device)

    def train(self):
        xs = self.Data['train']['x']
        ys = self.Data['train']['y']
        ts = self.Data['train']['t']

        test_data = {}
        test_data['x'] = self.Data['test']['x']
        test_data['y'] = self.Data['test']['y']
        test_data['t'] = self.Data['test']['t']
        TestDataloader = DataLoader(
            dataset(test_data), batch_size=self.train_conf['batch'], shuffle=False)

        train_size = len(ts)
        while(self.cur_epoch < self.train_conf['epoch']):
            self.cur_epoch += 1
            print('====epoch:'+str(self.cur_epoch)+' ====>正在训练')
            print('--------------------------------------------------------------')

            # 随机选n%做validation数据
            validation_index = random.sample(range(train_size), int(
                train_size*self.train_conf['split']/100.))
            validation_mask = np.array([False] * train_size)
            validation_mask[validation_index] = True

            # 验证集
            val_data = {}
            val_data['x'] = xs[validation_mask]
            val_data['y'] = ys[validation_mask]
            val_data['t'] = ts[validation_mask]
            ValDataloader = DataLoader(
                dataset(val_data), batch_size=self.train_conf['batch'], shuffle=False)
            # 训练集
            train_data = {}
            train_data['x'] = xs[~validation_mask]
            train_data['y'] = ys[~validation_mask]
            train_data['t'] = ts[~validation_mask]
            TrainDataloader = DataLoader(
                dataset(train_data), batch_size=self.train_conf['batch'], shuffle=False)

            # -------------------------------------------------------------------------------
            print('\033[1;34m Train: \033[0m')
            loss_sum = 0
            t_pred = []
            t_ori = []
            self.model.train()
            for _, sample in enumerate(TrainDataloader):

                self.optimizer.zero_grad()

                var_x = self.to_variable(sample[0])
                var_y = self.to_variable(sample[1])
                var_t = self.to_variable(sample[2])

                out = self.model(var_x, var_y)
                pre_t = (out >= 0.5) + 0

                t_pred.extend(pre_t.data.cpu().numpy())
                t_ori.extend(sample[2].numpy())

                loss = self.model.loss_func(out, var_t)
                loss.backward()

                self.optimizer.step()
                loss_sum += loss.data.item()
            self.epoch_Loss = loss_sum/train_size
            train_accuracy, precision, recall, f1 = self.metrics(t_pred, t_ori)
            train_random = self.rand_acc(t_ori)
            self.acc_train_max_diff = max(
                self.acc_train_max_diff, train_accuracy-train_random)
            print('第 \033[1;34m %d \033[0m 轮的训练集正确率为:\033[1;32m %.4f \033[0m epoch_mean_Loss 为: \033[1;32m %.4f \033[0m' %
                  (self.cur_epoch, train_accuracy, self.epoch_Loss))
            print('\033[1;31m Accuracy:%.4f Precision:%.4f Recall:%.4f F1:%.4f \033[0m' % (
                train_accuracy, precision, recall, f1))
            print('\033[1;31m Random:%.4f\tMaxAccDiff:%.6f \033[0m' %
                  (train_random, self.acc_train_max_diff))

            # -------------------------------------------------------------------------------
            print('\033[1;34m Valid: \033[0m')
            with torch.no_grad():
                t_pred = []
                t_ori = []
                self.model.eval()
                for _, sample in enumerate(ValDataloader):
                    var_x = self.to_variable(sample[0])
                    var_y = self.to_variable(sample[1])
                    var_t = self.to_variable(sample[2])

                    out = self.model(var_x, var_y)
                    pre_t = (out >= 0.5) + 0

                    t_pred.extend(pre_t.data.cpu().numpy())
                    t_ori.extend(sample[2].numpy())

            validation_accuracy, precision, recall, f1 = self.metrics(
                t_pred, t_ori)
            validation_random = self.rand_acc(t_ori)

            model_best = validation_accuracy-validation_random > self.acc_val_max_diff
            self.acc_val_max_diff = max(
                self.acc_val_max_diff, validation_accuracy-validation_random)
            print('\033[1;31m Accuracy:%.4f Precision:%.4f Recall:%.4f F1:%.4f \033[0m' % (
                validation_accuracy, precision, recall, f1))
            print('\033[1;31m Random:%.4f\tMaxAccDiff:%.6f \033[0m' %
                  (validation_random, self.acc_val_max_diff))

            # -------------------------------------------------------------------------------
            print('\033[1;34m Test: \033[0m')
            with torch.no_grad():
                t_pred = []
                t_ori = []
                self.model.eval()

                for _, sample in enumerate(TestDataloader):

                    var_x = self.to_variable(sample[0])
                    var_y = self.to_variable(sample[1])
                    var_t = self.to_variable(sample[2])

                    out = self.model(var_x, var_y)
                    pre_t = (out >= 0.5) + 0

                    t_pred.extend(pre_t.data.cpu().numpy())
                    t_ori.extend(sample[2].numpy())

            test_accuracy, precision, recall, f1 = self.metrics(t_pred, t_ori)
            test_random = self.rand_acc(t_ori)
            self.acc_test_max_diff = max(
                self.acc_test_max_diff, test_accuracy-test_random)
            print('\033[1;31m Accuracy:%.4f Precision:%.4f Recall:%.4f F1:%.4f \033[0m' % (
                test_accuracy, precision, recall, f1))
            print('\033[1;31m Random:%.4f\ttestMaxAccDiff:%.6f \033[0m' %
                  (test_random, self.acc_test_max_diff))

            def save_checkpoint(self, best=True):
                self.result['epoch'].append(self.cur_epoch)
                self.result['loss'].append(epoch_Loss)
                self.result['train_accuracy'].append(train_accuracy)
                self.result['acc_train_max_diff'].append(
                    self.acc_train_max_diff)
                self.result['validation_accuracy'].append(validation_accuracy)
                self.result['acc_val_max_diff'].append(self.acc_val_max_diff)
                self.result['test_random'].append(test_random)
                self.result['test_accuarcy'].append(test_accuracy)
                self.result['acc_test_max_dif'].append(self.acc_test_max_dif)
                if not os.path.exists(self.result_path):
                    print('第一次保存，新建目录:', self.result_path)
                    os.mkdir(self.result_path)
                pd.DataFrame(self.result).to_csv(self.csv_name, index=False)
                torch.save(
                    {
                        "epoch": self.cur_epoch,
                        "epoch_Loss:": self.epoch_Loss,
                        "acc_train_max_diff": self.acc_train_max_diff,
                        "acc_val_max_diff": self.acc_val_max_diff,
                        "acc_test_max_diff": self.acc_test_max_diff,
                        "model_state_dict": self.model.state_dict(),
                        "optim_state_dict": self.optimizer.state_dict(),
                    },
                    os.path.join(self.result_path,
                                 "{0}.pt".format("best" if best else "last")))

            self.save_checkpoint(model_best)

    def to_variable(self, x):
        return Variable(x.type(torch.FloatTensor)).to(self.device)

    def metrics(self, results, ori_y):
        accuracy = accuracy_score(ori_y, results)
        precision = precision_score(
            ori_y, results, labels=[1], average=None)[0]
        recall = recall_score(ori_y, results, labels=[1], average=None)[0]
        f1 = f1_score(ori_y, results, labels=[1], average=None)[0]
        return accuracy, precision, recall, f1

    def rand_acc(self, t_ori):
        return max([np.sum(np.array(t_ori) == r) for r in [0, 1]]) * 1. / len(t_ori)

    def update_lr(self):
        for param_group in self.optim.param_groups:
            param_group['lr'] = param_group['lr'] * 0.9


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='输入参数yml文件')
    parser.add_argument('-opt', type=str, default='./train.yml',
                        help='Path to option YAML file.')
    args = parser.parse_args()

    opt = parse(args.opt)
    Datapath = opt['data_conf']['datapath']
    trainer = Trainer(opt['model_conf'], opt['data_conf'], opt['train_conf'])
    trainer.train()
