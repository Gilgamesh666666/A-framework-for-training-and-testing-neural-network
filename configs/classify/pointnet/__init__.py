'''
Author: your name
Date: 2021-03-23 19:39:02
LastEditTime: 2021-05-17 15:54:48
LastEditors: ze bai
Description: In User Settings Edit
FilePath: /classify_exp/configs/classify/pointnet/__init__.py
'''
import os
from utils.config import Config, configs
from model import *
from tqdm import tqdm, trange
# k, emb_dims, dropout, output_channels
configs.model = Config(PointNet)
configs.model.emb_dims = 1024
configs.model.output_channels = 2
configs.dataset.with_flag = True
configs.model.with_flag = configs.dataset.with_flag

configs.exp_name = 'classify_pt_1024_0'

configs.dataloader.batch_size = {'train': 16, 'valid': 16, 'test': 16}
configs.dataloader.num_workers = 8

configs.train.ckpt_dir = os.path.join('checkpoint', configs.exp_name)
configs.train.common_ckpt_path = os.path.join(configs.train.ckpt_dir, 'common.ckpt.pth')
configs.train.best_ckpt_paths = os.path.join(configs.train.ckpt_dir, '{}.best.ckpt.pth')
configs.train.logfile = os.path.join(configs.train.ckpt_dir, 'train.log')

configs.train.optimizer.weight_decay = 1e-6