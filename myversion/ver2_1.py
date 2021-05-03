#!/usr/bin/env python
# encoding: utf-8
'''
Darnn  改变lstm的隐变量长度
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

from ver5 import myLSTM


if __name__ == '__main__':
    random.seed(0)
    parser = argparse.ArgumentParser(description='输入参数yml文件')
    parser.add_argument('-opt', type=str, default='./train.yml',
                        help='Path to option YAML file.')
    args = parser.parse_args()
    opt = get_opt(args.opt)[__file__[:-3]]
    Train = Trainer(myLSTM,opt['model_conf'], opt['data_conf'], opt['train_conf'],__file__[:-3])
    Train.run()