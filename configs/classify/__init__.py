'''
Author: your name
Date: 2020-11-20 22:28:19
LastEditTime: 2021-05-16 15:31:39
LastEditors: ze bai
Description: In User Settings Edit
FilePath: /classify_exp/configs/classify/__init__.py
'''
from utils.config import Config, configs
from datasets.classify_dataset import *
import torch.nn as nn
import torch.optim as optim

#root, split, with_flag=True, normalize=True, with_random_rot=True, jitter=True
configs.dataset = Config(ClassifyDataset, split=['train', 'valid', 'test'])
configs.dataset.root = "data/3dmatch"

configs.dataset.num_points = 10000
configs.dataset.normalize = False
configs.dataset.with_random_rot = False
configs.dataset.jitter = False

configs.train = Config()
configs.train.num_epochs = 200

configs.train.optimizer = Config(optim.Adam)
configs.train.optimizer.lr = 1e-4
configs.train.valid_interval = 1
configs.train.scheduler = Config(optim.lr_scheduler.CosineAnnealingLR)
configs.train.scheduler.T_max = configs.train.num_epochs

configs.train.criterion = Config(nn.CrossEntropyLoss)
configs.train.meters = Config()
configs.train.meters['train-acc_{}'] = Config(MeterClassify)

configs.evaluate = Config()
configs.evaluate.meters = Config()
configs.evaluate.meters['eval-acc_{}'] = Config(MeterClassify)
configs.evaluate.fn = None