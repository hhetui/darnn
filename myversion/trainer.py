# 模型训练统一框架
import os
import sys
import shutil
import random
import pandas as pd
from collections import defaultdict
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import  DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

from utils import get_opt, get_logger,load_dataset,dataset

class Trainer:
    def __init__(self, myModel ,model_conf, data_conf, train_conf,model_file_name):
        self.model_conf = model_conf
        self.data_conf = data_conf
        self.train_conf = train_conf
        self.model_file_name = model_file_name
        self.result_path = os.path.join(
            self.train_conf['checkpoint_path'], self.model_file_name)
        
        if os.path.exists(self.result_path) and not self.train_conf['resume']:
                print('检测到有checkpoint,确定要删除并重新训练吗？y/n')
                ans = input()
                if ans == 'y' or '\n':
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
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.logger.info('此次实验设备为:'+str(self.device))
        self.logger.info('实验参数如下:')
        self.logger.info(self.model_conf)
        self.logger.info(self.train_conf)
        self.logger.info(self.data_conf)
        self.model = myModel(**self.model_conf)
        self.load_checkpoint()
        
        self.logger.info('导入数据集......')
        self.Data = load_dataset(self.data_conf)
        self.logger.info('数据集导入成功！')
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.6, patience=self.train_conf['patience'], verbose=True, min_lr=0)
        self.scheduler.best = self.test_best_loss
    def load_checkpoint(self):
        self.logger.info('Load Checkpoint......')
        
        self.csv_name = os.path.join(self.result_path, self.model_file_name+'.csv')
        if self.train_conf['resume'] and os.path.exists(self.csv_name):
            self.logger.info('已有存档点，读取中......')
            checkpoint = torch.load(os.path.join(self.result_path,
                                                 "last.pt"), map_location='cpu')
            self.logger.info('正在导入模型、优化器参数......')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.train_conf['learning_rate'])
            self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
            self.logger.info('导入成功!')
            self.cur_epoch = checkpoint['epoch']
            self.acc_train_max_diff = checkpoint['acc_train_max_diff']
            self.acc_val_max_diff = checkpoint['acc_val_max_diff']
            self.test_best_loss = checkpoint['test_best_loss']
            self.best_model_test_acc_diff = checkpoint['best_model_test_acc_diff']
            self.acc_test_max_diff = checkpoint['acc_test_max_diff']
            self.result = pd.read_csv(self.csv_name).to_dict(orient='list')
            self.logger.info('存档点读取完毕，目前已训练{}轮。'.format(self.cur_epoch))
        else:
            self.logger.info('没有存档点，各种参数初始化')
            self.model = self.model.to(self.device)
            self.optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.train_conf['learning_rate'])
            self.cur_epoch = 0
            self.acc_train_max_diff = 0
            self.acc_val_max_diff = 0
            self.test_best_loss = float('inf')
            self.best_model_test_acc_diff = 0
            self.acc_test_max_diff = 0
            self.result = defaultdict(list)
        

    def run(self):
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
        self.no_impr = 0
        while(self.cur_epoch < self.train_conf['epoch']):
            self.cur_epoch += 1
            self.logger.info('======epoch:'+str(self.cur_epoch) +
                             ' 正在训练 ========================>')
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
            train_accuracy = self.train(TrainDataloader)
            validation_accuracy,val_loss = self.valid(ValDataloader)
            test_accuracy,test_random,test_loss= self.test(TestDataloader)
            
            def save_checkpoint(best=True):
                self.result['epoch'].append(self.cur_epoch)
                self.result['loss'].append(self.epoch_Loss)
                self.result['lr'].append(self.optimizer.state_dict()['param_groups'][0]['lr'])
                self.result['train_accuracy'].append(train_accuracy)
                self.result['acc_train_max_diff'].append(
                    self.acc_train_max_diff)
                self.result['validation_accuracy'].append(validation_accuracy)
                self.result['validation_loss'].append(val_loss)
                self.result['acc_val_max_diff'].append(self.acc_val_max_diff)
                self.result['test_random'].append(test_random)
                self.result['test_accuarcy'].append(test_accuracy)
                self.result['test_loss'].append(test_loss)
                self.result['acc_test_max_dif'].append(self.acc_test_max_diff)
                if not os.path.exists(self.result_path):
                    self.logger.info('第一次保存，新建目录:', self.result_path)
                    os.mkdir(self.result_path)
                pd.DataFrame(self.result).to_csv(self.csv_name, index=False)
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
                    os.path.join(self.result_path,
                                 "{0}.pt".format("best" if best else "last")))

            if test_loss < self.test_best_loss:
                save_checkpoint(best = True)
                self.test_best_loss = test_loss
                self.best_model_test_acc_diff = test_accuracy - test_random
                self.no_impr = 0
                self.logger.info('Epoch: {:d}, now best test loss change: {:.8f}'.format(self.cur_epoch,self.test_best_loss))
            else:
                self.no_impr += 1
                self.logger.info('{:d} no improvement, best loss: {:.4f}'.format(self.no_impr,self.scheduler.best))
            save_checkpoint(best = False)

            if self.no_impr == self.train_conf['stop']:
                self.show_result()
                self.logger.info(
                    "Stop training cause no impr for {:d} epochs".format(self.no_impr))
                break
        self.show_result()
    
    def train(self,TrainDataloader):
        self.logger.info('\033[1;34m Train: \033[0m')
        train_loss = 0
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
            train_loss += loss.data.item()
            
        self.epoch_Loss = train_loss/len(t_ori)
        train_accuracy, precision, recall, f1 = self.metrics(t_pred, t_ori)
        train_random = self.rand_acc(t_ori)
        self.acc_train_max_diff = max(
            self.acc_train_max_diff, train_accuracy-train_random)
        self.logger.info('第 \033[1;34m %d \033[0m 轮的训练集正确率为:\033[1;32m %.4f \033[0m epoch_mean_Loss 为: \033[1;32m %.8f \033[0m' %
                        (self.cur_epoch, train_accuracy, self.epoch_Loss))
        self.logger.info('\033[1;31m Accuracy:%.4f Precision:%.4f Recall:%.4f F1:%.4f \033[0m' % (
            train_accuracy, precision, recall, f1))
        self.logger.info('\033[1;31m Random:%.4f\tMaxAccDiff:%.6f \033[0m' %
                    (train_random, self.acc_train_max_diff))
        return train_accuracy
    
    def valid(self,ValDataloader):
        self.logger.info('\033[1;34m Valid: \033[0m')
        with torch.no_grad():
            t_pred = []
            t_ori = []
            val_loss = 0
            self.model.eval()
            for _, sample in enumerate(ValDataloader):
                var_x = self.to_variable(sample[0])
                var_y = self.to_variable(sample[1])
                var_t = self.to_variable(sample[2])

                out = self.model(var_x, var_y)
                pre_t = (out >= 0.5) + 0
                val_loss += self.model.loss_func(out, var_t).data.item()
                t_pred.extend(pre_t.data.cpu().numpy())
                t_ori.extend(sample[2].numpy())
                
            val_loss = val_loss/len(t_ori)
        validation_accuracy, precision, recall, f1 = self.metrics(
            t_pred, t_ori)
        validation_random = self.rand_acc(t_ori)

        self.scheduler.step(val_loss) 
        sys.stdout.flush()  
        self.acc_val_max_diff = max(
            self.acc_val_max_diff, validation_accuracy-validation_random)
        self.logger.info('\033[1;31m Accuracy:%.4f Precision:%.4f Recall:%.4f F1:%.4f val_loss:%.8f \033[0m' % (
            validation_accuracy, precision, recall, f1 , val_loss))
        self.logger.info('\033[1;31m Random:%.4f\tMaxAccDiff:%.6f \tbest_model_test_acc_diff:%.6f \033[0m' %
                        (validation_random, self.acc_val_max_diff,self.best_model_test_acc_diff))
        return validation_accuracy,val_loss

    def test(self,TestDataloader):
        self.logger.info('\033[1;34m Test: \033[0m')
        with torch.no_grad():
            t_pred = []
            t_ori = []
            self.model.eval()
            test_loss = 0
            for _, sample in enumerate(TestDataloader):

                var_x = self.to_variable(sample[0])
                var_y = self.to_variable(sample[1])
                var_t = self.to_variable(sample[2])

                out = self.model(var_x, var_y)
                pre_t = (out >= 0.5) + 0
                test_loss += self.model.loss_func(out, var_t).data.item()
                t_pred.extend(pre_t.data.cpu().numpy())
                t_ori.extend(sample[2].numpy())
            test_loss = test_loss/len(t_ori)  
        test_accuracy, precision, recall, f1 = self.metrics(t_pred, t_ori)
        test_random = self.rand_acc(t_ori)
        self.acc_test_max_diff = max(
            self.acc_test_max_diff, test_accuracy-test_random)
        self.logger.info('\033[1;31m Accuracy:%.4f Precision:%.4f Recall:%.4f F1:%.4f val_loss:%.8f \033[0m' % (
            test_accuracy, precision, recall, f1, test_loss))
        self.logger.info('\033[1;31m Random:%.4f\ttestMaxAccDiff:%.6f \033[0m' %
                        (test_random, self.acc_test_max_diff))
        return test_accuracy,test_random,test_loss

    def show_result(self):
        self.logger.info('总共训练了{:d}轮~'.format(self.cur_epoch))
        self.logger.info('best_model_test_acc_diff:{:.8f}'.format(self.best_model_test_acc_diff))
        self.logger.info('acc_test_max_diff:{:.8f}'.format(self.acc_test_max_diff))

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

