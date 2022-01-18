'''
Author: your name
Date: 2021-03-21 16:27:18
LastEditTime: 2022-01-18 17:13:21
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /classify_exp/datasets/classify_dataset.py
'''
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import open3d as o3d
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

#__all__ = ['ShapeNet']

class _ClassifyDataset(Dataset):
    def __init__(self, root, split, ):
        assert split in ['train', 'valid', 'test']
        
        self.root = root
        self.split = split
        

    def __getitem__(self, index):
        

        return data, targets

    def __len__(self):
        return len(self.data_pkl['label'])
        #return min(1000, len(self.data_pkl['label']))
    
class ClassifyDataset(dict):
    def __init__(self, root, split, ):
        super().__init__()
        if split is None:
            split = ['train', 'valid', 'test']
        elif not isinstance(split, (list, tuple)):
            split = [split]
        for s in split:
            self[s] = _ClassifyDataset(root=root, split=s, )

class MeterClassify:
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.true_sum = 0
        self.sample_count = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        
        self.true_sum += (torch.argmax(outputs, dim=1) == targets).sum()
        self.sample_count += outputs.shape[0]

    def compute(self):
        return self.true_sum / self.sample_count

if __name__ == '__main__':
    
    dataset = _ClassifyDataset("data/xxx", 'train', )
    dataloader = DataLoader(dataset, batch_size=1,shuffle=True,
                            num_workers=2, 
                            pin_memory=False,
                            worker_init_fn=lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id + 1))
    for data, targets in dataloader:
        pass