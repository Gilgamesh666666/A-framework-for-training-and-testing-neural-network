'''
Author: your name
Date: 2021-03-21 16:27:18
LastEditTime: 2022-01-18 17:10:39
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /classify_exp/configs/classify/dgcnn/__init__.py
'''
import os
from utils.config import Config, configs
from model import *
from tqdm import tqdm, trange
# k, emb_dims, dropout, output_channels
configs.model = Config(DGCNN)
configs.model.k = 20
configs.model.emb_dims = 1024
configs.model.dropout = 0.5
configs.model.output_channels = 2

configs.exp_name = 'classify_dgcnn_1'

configs.dataloader.batch_size = {'train': 2, 'valid': 2, 'test': 2}
configs.dataloader.num_workers = 2

configs.train.ckpt_dir = os.path.join('checkpoint', configs.exp_name)
configs.train.common_ckpt_path = os.path.join(configs.train.ckpt_dir, 'common.ckpt.pth')
configs.train.best_ckpt_paths = os.path.join(configs.train.ckpt_dir, '{}.best.ckpt.pth')
configs.train.logfile = os.path.join(configs.train.ckpt_dir, 'train.log')

configs.train.optimizer.weight_decay = 1e-6

# def step_train(model, dataloader, criterion, optimizer, scheduler, writer, current_step, device):
#     model.train()
#     for inputs, targets in tqdm(dataloader, desc='train', ncols=0):
#         if not isinstance(inputs, (list, tuple)):
#             inputs = inputs.to(device)
#         if not isinstance(targets, (list, tuple)):
#             targets = targets.to(device)
#         #inputs = inputs.cuda(1)
#         #targets = targets.cuda(1)
#         #print(f'inputs = {inputs[:, :3, :].permute(0, 2, 1)}')
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         writer.add_scalar('loss/train',loss.item(), current_step)
#         current_step += 1
#     if scheduler is not None:
#         scheduler.step()
#         #print(loss.item())
        
# configs.train.fn = step_train
