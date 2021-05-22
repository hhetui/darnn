# 公用模型工具函数
import os
import yaml
import logging
import pickle
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader


def get_opt(opt_path):
    '''
       opt_path: the path of yml file
       is_train: True
    '''
    #logger.info('Reading .yml file .......')
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    # Export CUDA_VISIBLE_DEVICES
    #gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    #os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    #logger.info('Export CUDA_VISIBLE_DEVICES = {}'.format(gpu_list))

    # is_train into option
    #opt['is_train'] = is_tain
    return opt


def get_logger(logfile, format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
               date_format='%Y-%m-%d %H:%M:%S'):
    logger = logging.getLogger(logfile)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    # file or console
    handler = logging.FileHandler(logfile)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    handler_str = logging.StreamHandler()
    handler_str.setLevel(logging.INFO)
    handler_str.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(handler_str)
    return logger



class DataLoader_Generate:
    def __init__(self,train_conf,data_conf):
        self.train_conf = train_conf
        self.data_conf = data_conf
        self.Data = self.Load_dataset()

    def Load_dataset(self):
        '''
        返回字典dataset 键值:train和test 各存储 一个Dic
        '''
        Data = {}
        Data['train'] = self.pickle2dic(self.data_conf['train_list'])
        Data['test'] = self.pickle2dic(self.data_conf['test_list'])
        #可能的进一步处理
        if self.data_conf['dataset_type'] == 1:
            #原始数据处理方式
            pass
        elif self.data_conf['dataset_type'] == 2:
            #按照时间合并同一天的数据，设置batch为1，然后在trainer内将弹出数据去掉第一个维度
            def day_data(ori_df):
                data_x = []
                data_y = []
                data_t = []
                dic = {}
                time_list = sorted(
                    set(ori_df['day']), key=list(ori_df['day']).index)
                for day in time_list:
                    res = ori_df[ori_df['day'] == day]
                    data_x.append(np.array(list(res['x'])))
                    data_y.append(np.array(list(res['y'])))
                    data_t.append(np.array(list(res['t'])))
                dic['x']=data_x
                dic['y']=data_y
                dic['t']=data_t
                return dic
            Data['train'] = day_data(Data['train'])
            Data['test'] = day_data(Data['test'])
        return Data

    def pickle2dic(self,year_list):
        '''
        将year_list内所有pickle字典数据拼接后，返回字典类型
        '''
        data_dic = None
        for y in year_list:
            with open(os.path.join(self.data_conf['datapath'],
                                'v1_T'+str(self.data_conf['T'])+'_yb1_%s.pickle' % (y)), 'rb') as fp:
                dataset = pickle.load(fp)
                if data_dic is None:
                    data_dic = dataset
                else:
                    for key in list(data_dic.keys()):
                        data_dic[key] = np.concatenate((data_dic[key],dataset[key]),0)

        return data_dic
    
    def Dic2DataLoader(self,Dic):
        class dataset(Dataset):
            def __init__(self, dic):
                self.dic = dic
                self.L = len(self.dic[list(self.dic.keys())[0]])
            def __getitem__(self, index):
                # 返回的目标是0 ,1
                
                return self.dic['x'][index],self.dic['y'][index],self.dic['t'][index],

            def __len__(self):
                return self.L
        return DataLoader(dataset(Dic),batch_size=self.train_conf['batch'], shuffle=False)

    def Get_TestDataLoader(self):
        return self.Dic2DataLoader(self.Data['test'])

    def Get_Train_ValLoader(self):
        Size = len(self.Data['train'][list(self.Data['train'].keys())[0]])
        validation_index = random.sample(range(Size), int(
                Size*self.train_conf['split']))
        validation_mask = np.array([False] * Size)
        validation_mask[validation_index] = True
        
        Train_data = self.getsubdata(self.Data['train'],~validation_mask)
        Val_data = self.getsubdata(self.Data['train'],validation_mask)



        TrainDataloader = self.Dic2DataLoader(Train_data)
        ValDataloader = self.Dic2DataLoader(Val_data)
        return TrainDataloader, ValDataloader
    def getsubdata(self,dic,mask):
        res = {}
        for key in list(dic.keys()):
            res[key] = dic[key][mask]
        return res