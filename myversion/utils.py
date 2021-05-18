# 公用模型工具函数
import os
import yaml
import logging
import pickle
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
        返回字典dataset 键值:train和test 各存储 一个Dataframe  
        '''
        Data = {}
        Data['train'] = self.pickle2DF(self.data_conf['train_list'])
        Data['test'] = self.pickle2DF(self.data_conf['test_list'])
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
                return pd.DataFrame(dic)
            Data['train'] = day_data(Data['train'])
            Data['test'] = day_data(Data['test'])
        return Data

    def pickle2DF(self,year_list):
        '''
        将year_list内所有pickle字典数据拼接后，封装成Dataframe类型返回
        '''
        data_dic = None
        for y in year_list:
            with open(os.path.join(self.data_conf['datapath'],
                                'v1_T'+str(self.data_conf['T'])+'_yb1_%s.pickle' % (y)), 'rb') as fp:
                dataset = pickle.load(fp)

                if data_dic is None:
                    data_dic = {}
                    data_dic['x'] = list(dataset['x'])
                    data_dic['y'] = list(dataset['y'])
                    data_dic['t'] = list(dataset['t'])
                    data_dic['day'] = list(dataset['day'])
                else:
                    data_dic['x'] = data_dic['x'] + list(dataset['x'])
                    data_dic['y'] = data_dic['y'] + list(dataset['y'])
                    data_dic['t'] = data_dic['t'] + list(dataset['t'])
                    data_dic['day'] = data_dic['day'] + list(dataset['day'])
        return pd.DataFrame(data_dic)
    
    def DF2DataLoader(self,DF):
        class dataset(Dataset):
            def __init__(self, df):
                self.df = df[['x', 'y', 't']]

            def __getitem__(self, index):
                # 返回的目标是0 ,1
                return self.df.iloc[index]['x'], self.df.iloc[index]['y'], self.df.iloc[index]['t']

            def __len__(self):
                return len(self.df['t'])
        return DataLoader(dataset(DF),batch_size=self.train_conf['batch'], shuffle=False)

    def Get_TestDataLoader(self):
        return self.DF2DataLoader(self.Data['test'])

    def Get_Train_ValLoader(self):
        Train_data = self.Data['train'].sample(frac=1-self.train_conf['split'])
        Train_data.sort_index(inplace=True)
        Val_data = self.Data['train'][~self.Data['train'].index.isin(
            Train_data.index)]
        Val_data.sort_index(inplace=True)

        TrainDataloader = self.DF2DataLoader(Train_data)
        ValDataloader = self.DF2DataLoader(Val_data)
        return TrainDataloader, ValDataloader
