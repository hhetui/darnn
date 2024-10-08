# 模型训练统一框架
import os
import sys
import shutil
import random
import pandas as pd
from collections import defaultdict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

from utils import get_opt, get_logger, DataLoader_Generate


class Trainer:
    def __init__(self, myModel, model_conf, data_conf, train_conf, model_file_name):
        self.model_conf = model_conf
        self.data_conf = data_conf
        self.train_conf = train_conf
        self.model_file_name = model_file_name
        self.result_path = os.path.join(
            self.train_conf['checkpoint_path'], self.model_file_name)

        if os.path.exists(self.result_path) and not self.train_conf['resume']:
            print('检测到有checkpoint,确定要删除并重新训练吗？y/n')
            ans = input()
            if ans == 'y' or ans == '':
                shutil.rmtree(self.result_path)
            elif ans == 'n':
                raise ValueError("如果想读取checkpoint继续训练，请修改yml文件中的resume值为True!")
            else:
                raise ValueError("请输入y or n")
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
        self.logger = get_logger(
            os.path.join(self.result_path, self.train_conf['log_file']))
        self.device = torch.device(
            'cuda:'+str(self.train_conf['cudaid']) if torch.cuda.is_available() else 'cpu')
        self.logger.info('此次实验设备为:'+str(self.device))
        self.logger.info('实验参数如下:')
        self.logger.info(self.model_conf)
        self.logger.info(self.train_conf)
        self.logger.info(self.data_conf)
        self.model = myModel(**self.model_conf)
        self.load_checkpoint()
        self.logger.info('导入数据集......')
        self.DataLoader_Generate = DataLoader_Generate(
            self.train_conf, self.data_conf)
        self.logger.info('数据集导入成功！')
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.6, patience=self.train_conf['patience'], verbose=True, min_lr=0)
        self.scheduler.best = self.test_best_loss

    def load_checkpoint(self):
        self.logger.info('Load Checkpoint......')

        self.csv_name = os.path.join(
            self.result_path, self.model_file_name+'.csv')
        if self.train_conf['resume'] and os.path.exists(self.csv_name):
            self.logger.info('已有存档点，读取中......')
            checkpoint = torch.load(os.path.join(self.result_path,
                                                 "last.pt"), map_location='cpu')
            self.logger.info('正在导入模型、优化器参数......')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=self.train_conf['learning_rate'], weight_decay=self.train_conf['weight_decay'])
            self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
            self.cur_epoch = checkpoint['epoch']
            self.acc_train_max_diff = checkpoint['acc_train_max_diff']
            self.acc_val_max_diff = checkpoint['acc_val_max_diff']
            self.test_best_loss = checkpoint['test_best_loss']
            self.best_model_test_acc_diff = checkpoint['best_model_test_acc_diff']
            self.acc_test_max_diff = checkpoint['acc_test_max_diff']
            self.result = pd.read_csv(self.csv_name).to_dict(orient='list')
            self.logger.info('导入成功!')
            self.logger.info('存档点读取完毕，目前已训练{}轮。'.format(self.cur_epoch))
        else:
            self.logger.info('没有存档点，各种参数初始化')
            self.model = self.model.to(self.device)
            self.optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=self.train_conf['learning_rate'], weight_decay=self.train_conf['weight_decay'])
            self.cur_epoch = 0
            self.acc_train_max_diff = 0
            self.acc_val_max_diff = 0
            self.test_best_loss = float('inf')
            self.best_model_test_acc_diff = 0
            self.acc_test_max_diff = 0
            self.result = defaultdict(list)

    def save_checkpoint(self, best):

        self.result['epoch'].append(self.cur_epoch)
        self.result['loss'].append(self.epoch_Loss)
        self.result['lr'].append(self.optimizer.state_dict()[
            'param_groups'][0]['lr'])
        self.result['train_accuracy'].append(self.train_accuracy)
        self.result['acc_train_max_diff'].append(
            self.acc_train_max_diff)
        self.result['validation_accuracy'].append(self.validation_accuracy)
        self.result['validation_loss'].append(self.val_loss)
        self.result['acc_val_max_diff'].append(self.acc_val_max_diff)
        self.result['test_random'].append(self.test_random)
        self.result['test_accuarcy'].append(self.test_accuracy)
        self.result['test_loss'].append(self.test_loss)
        self.result['acc_test_max_dif'].append(self.acc_test_max_diff)

        # 保存csv结果
        pd.DataFrame(self.result).to_csv(self.csv_name, index=False)
        # 保存模型参数以及一些max数据
        self.save_pt('last')
        if best:
            self.save_pt('best')
        # 绘制折线图
        self.save_figure()

    def run(self):
        self.TestDataloader = self.DataLoader_Generate.Get_TestDataLoader()

        self.no_impr = 0
        while(self.cur_epoch < self.train_conf['epoch']):

            self.TrainDataloader, self.ValDataloader = self.DataLoader_Generate.Get_Train_ValLoader()

            self.cur_epoch += 1
            self.logger.info('======epoch:'+str(self.cur_epoch) +
                             ' 正在训练 ========================>')
            self.train()
            self.valid()
            self.test()

            if self.test_loss < self.test_best_loss:
                self.test_best_loss = self.test_loss
                self.best_model_test_acc_diff = self.test_accuracy - self.test_random
                self.no_impr = 0
                self.logger.info('Epoch: {:d}, now best test loss change: {:.8f}'.format(
                    self.cur_epoch, self.test_best_loss))
                self.save_checkpoint(best=True)
            else:
                self.no_impr += 1
                self.logger.info('{:d} no improvement, best loss: {:.4f}'.format(
                    self.no_impr, self.scheduler.best))
                self.save_checkpoint(best=False)

            if self.no_impr == self.train_conf['stop']:
                self.show_result()
                self.logger.info(
                    "Stop training cause no impr for {:d} epochs".format(self.no_impr))
                break
        self.show_result()

    def train(self):
        self.logger.info('\033[1;34m Train: \033[0m')
        train_loss = 0
        t_pred = []
        t_ori = []
        self.model.train()
        for _, sample in enumerate(self.TrainDataloader):

            self.optimizer.zero_grad()

            var_x, var_y, var_t = self.get_xy_from_sample(sample)

            out = self.model(var_x, var_y)

            if isinstance(self.model.loss_func, torch.nn.BCELoss): 
                pre_t = (out >= 0.5) + 0
            else:
                pre_t = out
            t_pred.extend(pre_t.data.cpu().numpy())
            t_ori.extend(var_t.cpu().numpy())

            loss = self.model.loss_func(out, var_t)
            loss.backward()

            self.optimizer.step()
            train_loss += loss.data.item()

        self.epoch_Loss = train_loss/len(t_ori)
        self.train_accuracy, precision, recall, f1 = self.metrics(
            t_pred, t_ori)
        if not hasattr(self,'train_random'):
            self.train_random = self.rand_acc(t_ori)
        self.acc_train_max_diff = max(
            self.acc_train_max_diff, self.train_accuracy-self.train_random)
        self.logger.info('第 \033[1;34m %d \033[0m 轮的训练集正确率为:\033[1;32m %.4f \033[0m epoch_mean_Loss 为: \033[1;32m %.8f \033[0m' %
                         (self.cur_epoch, self.train_accuracy, self.epoch_Loss))
        self.logger.info('\033[1;31m Accuracy:%.4f Precision:%.4f Recall:%.4f F1:%.4f \033[0m' % (
            self.train_accuracy, precision, recall, f1))
        self.logger.info('\033[1;31m Random:%.4f\tMaxAccDiff:%.6f \033[0m' %
                         (self.train_random, self.acc_train_max_diff))

    def valid(self):
        self.logger.info('\033[1;34m Valid: \033[0m')
        with torch.no_grad():
            t_pred = []
            t_ori = []
            self.val_loss = 0
            self.model.eval()
            for _, sample in enumerate(self.ValDataloader):
                var_x, var_y, var_t = self.get_xy_from_sample(sample)
                out = self.model(var_x, var_y)
                pre_t = (out >= 0.5) + 0
                self.val_loss += self.model.loss_func(out, var_t).data.item()
                t_pred.extend(pre_t.data.cpu().numpy())
                t_ori.extend(var_t.cpu().numpy())

            self.val_loss = self.val_loss/len(t_ori)
        self.validation_accuracy, precision, recall, f1 = self.metrics(
            t_pred, t_ori)
        if not hasattr(self,'validation_random'):
            self.validation_random = self.rand_acc(t_ori)

        self.scheduler.step(self.val_loss)
        sys.stdout.flush()
        self.acc_val_max_diff = max(
            self.acc_val_max_diff, self.validation_accuracy-self.validation_random)
        self.logger.info('\033[1;31m Accuracy:%.4f Precision:%.4f Recall:%.4f F1:%.4f val_loss:%.8f \033[0m' % (
            self.validation_accuracy, precision, recall, f1, self.val_loss))
        self.logger.info('\033[1;31m Random:%.4f\tMaxAccDiff:%.6f \tbest_model_test_acc_diff:%.6f \033[0m' %
                         (self.validation_random, self.acc_val_max_diff, self.best_model_test_acc_diff))

    def test(self):
        self.logger.info('\033[1;34m Test: \033[0m')
        with torch.no_grad():
            t_pred = []
            t_ori = []
            self.model.eval()
            self.test_loss = 0
            for _, sample in enumerate(self.TestDataloader):
                var_x, var_y, var_t = self.get_xy_from_sample(sample)

                out = self.model(var_x, var_y)
                pre_t = (out >= 0.5) + 0
                self.test_loss += self.model.loss_func(out, var_t).data.item()
                t_pred.extend(pre_t.data.cpu().numpy())
                t_ori.extend(var_t.cpu().numpy())
            self.test_loss = self.test_loss/len(t_ori)
        self.test_accuracy, precision, recall, f1 = self.metrics(t_pred, t_ori)
        if not hasattr(self,'test_random'):
            self.test_random = self.rand_acc(t_ori)
        self.acc_test_max_diff = max(
            self.acc_test_max_diff, self.test_accuracy-self.test_random)
        self.logger.info('\033[1;31m Accuracy:%.4f Precision:%.4f Recall:%.4f F1:%.4f test_loss:%.8f \033[0m' % (
            self.test_accuracy, precision, recall, f1, self.test_loss))
        self.logger.info('\033[1;31m Random:%.4f\ttestMaxAccDiff:%.6f \033[0m' %
                         (self.test_random, self.acc_test_max_diff))

    def get_xy_from_sample(self, sample):
        var_x, var_y, var_t = self.to_Tensor(sample)
        if self.data_conf['dataset_type'] == 2:
            var_x = var_x.squeeze(0)
            var_y = var_y.squeeze(0)
            var_t = var_t.squeeze(0)
        return var_x, var_y, var_t

    def show_result(self):
        self.logger.info('总共训练了{:d}轮~'.format(self.cur_epoch))
        self.logger.info('best_model_test_acc_diff:{:.8f}'.format(
            self.best_model_test_acc_diff))
        self.logger.info('acc_test_max_diff:{:.8f}'.format(
            self.acc_test_max_diff))

    def save_pt(self, pt_name):
        torch.save(
            {
                "epoch": self.cur_epoch,
                "epoch_Loss:": self.epoch_Loss,
                "acc_train_max_diff": self.acc_train_max_diff,
                "acc_val_max_diff": self.acc_val_max_diff,
                "test_best_loss": self.test_best_loss,
                "best_model_test_acc_diff": self.best_model_test_acc_diff,
                "acc_test_max_diff": self.acc_test_max_diff,
                "model_state_dict": self.model.state_dict(),
                "optim_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(self.result_path, str(pt_name)+".pt"))

    def save_figure(self):
        sns.set()
        plt.rcParams['lines.linewidth'] = 0.3
        plt.switch_backend('agg')
        train_Loss = plt.plot(
            self.result['epoch'], self.result['loss'], 'lightcoral', label='loss')
        test_Loss = plt.plot(
            self.result['epoch'], self.result['test_loss'], 'lightseagreen', label='test_loss')
        plt.title('Loss_figure')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.result_path,
                                 self.model_file_name+'_Loss_figure.jpg'), dpi=1500)
        plt.clf()
        train_Acc = plt.plot(
            self.result['epoch'], self.result['train_accuracy'], 'lightcoral', label='train_acc')
        test_Acc = plt.plot(
            self.result['epoch'], self.result['test_accuarcy'], 'lightseagreen', label='test_acc')
        plt.title('Acc_figure')
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend()
        plt.savefig(os.path.join(self.result_path,
                                 self.model_file_name+'_Acc_figure.jpg'), dpi=1500)

    def to_Tensor(self, sample):
        x = sample[0].type(torch.FloatTensor).to(self.device)
        y = sample[1].type(torch.FloatTensor).to(self.device)
        if isinstance(self.model.loss_func, torch.nn.BCELoss):
            t = sample[2].type(torch.FloatTensor).to(self.device)
        elif isinstance(self.model.loss_func, torch.nn.CrossEntropyLoss):
            t = sample[2].type(torch.LongTensor).to(self.device)
        return x, y, t

    def metrics(self, results, ori_y):
        if isinstance(self.model.loss_func, torch.nn.CrossEntropyLoss):
            prediction = torch.argmax(torch.Tensor(results), 1)
            correct = torch.sum(prediction == torch.Tensor(ori_y))
            return correct/len(ori_y),0,0,0
        accuracy = accuracy_score(ori_y, results)
        precision = precision_score(
            ori_y, results, labels=[1], average=None)[0]
        recall = recall_score(ori_y, results, labels=[1], average=None)[0]
        f1 = f1_score(ori_y, results, labels=[1], average=None)[0]
        return accuracy, precision, recall, f1

    def rand_acc(self, t_ori):
        return max([np.sum(np.array(t_ori) == r) for r in set(t_ori)]) * 1. / len(t_ori)
