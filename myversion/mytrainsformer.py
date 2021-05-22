#!/usr/bin/env python
# encoding: utf-8
'''
自己实现一个transformer
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
from Base_model import SelfAttention


class 