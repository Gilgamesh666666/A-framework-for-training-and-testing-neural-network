'''
Author: your name
Date: 2021-03-21 11:26:09
LastEditTime: 2021-05-12 20:36:08
LastEditors: ze bai
Description: Generator new training data
FilePath: /classify_exp/data/original_pkl/generator_train_data.py
'''
import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
# data_folder = os.path.join('data/indoor/train')
# scene_name = sorted(os.listdir(data_folder))

# generator from pkl
# 'rot', 'trans', 'src'filename, 'tgt'filename, 'overlap'

# train_valid_data_pkl = {'src':[], 'tgt':[], 'label':[], 'rot':[], 'trans':[], 'overlap':[]}
def generate_random_rotation(max_degree=360, max_amp=3):
    # inputs:[N, 3], None/[N, 3], [N, 3]
    x, degree, amp = np.random.rand(6), np.random.rand(1)*max_degree*np.pi/180, np.random.rand(1)*max_amp
    w, v= x[:3], x[3:]
    w, v = w/np.linalg.norm(w), v/np.linalg.norm(v)
    w *= degree
    v *= amp
    r = Rotation.from_rotvec(w)
    return r.as_matrix(), v[None].T

positive_probability = 0.5
valid_ratio = 0.3
max_degree, max_amp = 360, 1

with open('train_info.pkl', 'rb') as f:
    datapkl = pickle.load(f)
#print(datapkl['src'])
with open('train_zero.pkl', 'rb') as f:
    zero_datapkl = pickle.load(f)
#print(datapkl['src'])

train_valid_data_pkl = datapkl
train_valid_data_pkl['label'] = []
for i in range(len(datapkl['src'])):
    if np.random.rand(1) > positive_probability:
        jitter_rot, jitter_trans = generate_random_rotation(max_degree=max_degree, max_amp=max_amp)
        train_valid_data_pkl['rot'][i] = np.dot(jitter_rot, datapkl['rot'][i])
        #print(jitter_trans)
        #print(datapkl['trans'][i])
        train_valid_data_pkl['trans'][i] = jitter_trans + datapkl['trans'][i]
        train_valid_data_pkl['label'].append(0)
    else:
        train_valid_data_pkl['label'].append(1)

for key in zero_datapkl.keys():
    if isinstance(train_valid_data_pkl[key], list):
        train_valid_data_pkl[key].extend(zero_datapkl[key])
    else:
        train_valid_data_pkl[key] = np.concatenate((train_valid_data_pkl[key], zero_datapkl[key]), axis=0)

train_valid_data_pkl['label'].extend([0]*len(zero_datapkl['src']))

#print(train_valid_data_pkl['label'])
num = len(train_valid_data_pkl['label'])
valid_num = int(num*valid_ratio)
idx = np.random.choice(num, valid_num, replace=False)

validmask = np.zeros(num, dtype=bool)
validmask[idx] = True
trainmask = np.logical_not(validmask)

train_pkl = {'src':[], 'tgt':[], 'label':[], 'rot':[], 'trans':[], 'overlap':[]}
valid_pkl = train_pkl

train_pkl['src'] = list(np.asarray(train_valid_data_pkl['src'])[trainmask])
train_pkl['tgt'] = list(np.asarray(train_valid_data_pkl['tgt'])[trainmask])
train_pkl['rot'] = train_valid_data_pkl['rot'][trainmask]
train_pkl['trans'] = train_valid_data_pkl['trans'][trainmask]
train_pkl['overlap'] = train_valid_data_pkl['overlap'][trainmask]

valid_pkl['src'] = list(np.asarray(train_valid_data_pkl['src'])[validmask])
valid_pkl['tgt'] = list(np.asarray(train_valid_data_pkl['tgt'])[validmask])
valid_pkl['rot'] = train_valid_data_pkl['rot'][validmask]
valid_pkl['trans'] = train_valid_data_pkl['trans'][validmask]
valid_pkl['overlap'] = train_valid_data_pkl['overlap'][validmask]

#print(valid_pkl['label'])
#print(train_pkl['label'])
with open('classify_train.pkl', 'wb') as f:
    pickle.dump(train_pkl, f)
with open('classify_valid.pkl', 'wb') as f:
    pickle.dump(valid_pkl, f)