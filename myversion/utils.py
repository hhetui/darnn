#公用模型工具函数
import os
import yaml
import logging
import pickle
import numpy as np
from torch.utils.data import Dataset

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

def load_dataset(data_conf):
    train_years = data_conf['train_list']
    test_years = data_conf['test_list']
    def load_pickle(years):
        data_dic = None
        for y in years:
            with open(os.path.join(data_conf['datapath'],
                'v1_T'+str(data_conf['T'])+'_yb1_%s.pickle' % (y)), 'rb') as fp:
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