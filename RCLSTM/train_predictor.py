# encoding: utf-8

"""

@author: huayuxiu

"""

"""Train the model using traffic matrices."""
import argparse
import math
import os
import time
from datetime import datetime
from functools import partial

# import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
# from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from collections import OrderedDict
import pandas as pd 

import rclstm
from rclstm import RNN, LSTMCell, RCLSTMCell

np.random.seed(1000)

def get_args(parser):
    parser.add_argument('--data', default='./data/traffic-1-2.npy', help='path to dataset')
    parser.add_argument('--model', default='lstm', choices=['lstm', 'rclstm'], help='the model to use')
    parser.add_argument('--connectivity', type=float, default=.5, help='the neural connectivity')
    parser.add_argument('--save', default='./model', help='The path to save model files')
    parser.add_argument('--hidden-size', type=int, default=200, help='The number of hidden units')
    parser.add_argument('--batch-size', type=int, default=32, help='The size of each batch')
    parser.add_argument('--input-size', type=int, default=1, help='The size of input data')
    parser.add_argument('--max-iter', type=int, default=2, help='The maximum iteration count')
    parser.add_argument('--gpu', default=True, action='store_true', help='The value specifying whether to use GPU')
    parser.add_argument('--time-window', type=int, default=100, help='The length of time window')
    parser.add_argument('--dropout', type=float, default=1., help='Dropout')
    parser.add_argument('--num-layers', type=int, default=1, help='The number of RNN layers')
    return parser

# 获取模型参数
parser = argparse.ArgumentParser()
parser = get_args(parser)
args = parser.parse_args()
print(args)

data_path = args.data
model_name = args.model
# save_dir = args.save
hidden_size = args.hidden_size
batch_size = args.batch_size
max_iter = args.max_iter
use_gpu = args.gpu
# connectivity = args.connectivity
time_window = args.time_window
input_size = args.input_size
dropout = args.dropout
num_layers = args.num_layers

def shufflelists(X, Y):
    ri=np.random.permutation(len(X))
    X_shuffle = [X[i].tolist() for i in ri]
    Y_shuffle = [Y[i].tolist() for i in ri]
    return np.array(X_shuffle), np.array(Y_shuffle)

# load data
data = np.load(data_path)
# take the logarithm of the original data
new_data = []
for x in data:
    if x > 0:
        new_data.append(np.log10(x))
    else:
        new_data.append(0.001)
new_data = np.array(new_data)
# handle abnormal data
new_data = new_data[new_data>2.5]
data = new_data[new_data<6]
# min-max normalization
max_data = np.max(data)
min_data = np.min(data)
data = (data-min_data)/(max_data-min_data)

df = pd.DataFrame({'temp':data})
# define function for create N lags
def create_lags(df, N):
    for i in range(N):
        df['Lag' + str(i+1)] = df.temp.shift(i+1)
    return df

# create time-windows lags
df = create_lags(df,time_window)
# the first 1000 days will have missing values. can't use them.
df = df.dropna()
# create X and y
y = df.temp.values
X = df.iloc[:, 1:].values
# train on 70% of the data
train_idx = int(len(df) * .7)
# create train and test data
train_X, train_Y, test_X, test_Y = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]

print('the number of train data: ', len(train_X))
print('the number of test data: ', len(test_X))
print('the shape of input: ', train_X.shape)
print('the shape of target: ', train_Y.shape)

def compute_loss_accuracy(loss_fn, data, label):
    hx = None
    _, (h_n, _) = model[0](input_=data, hx=hx)
    logits = model[1](h_n[-1])
    loss = torch.sqrt(loss_fn(input=logits, target=label))
    return loss, logits

#learning rate decay
def exp_lr_scheduler(optimizer, epoch, init_lr=1e-2, lr_decay_epoch=3):
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print("LR is set to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

save_dir = args.save
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
loss_fn = nn.MSELoss()
num_batch = int(math.ceil(len(train_X) // batch_size))
print('the number of batches: ', num_batch)

# train RCLSTM with different neural connection ratio
for connectivity in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    print('neural connection ratio:', connectivity)
    if model_name in ['lstm', 'rclstm']:
        rnn_model = RNN(device=device, cell_class=model_name, input_size=input_size,
                        hidden_size=hidden_size, connectivity=connectivity, 
                        num_layers=num_layers, batch_first=True, dropout=dropout)
    else:
        raise ValueError
    fc2 = nn.Linear(in_features=hidden_size, out_features=input_size)
    model = nn.Sequential(OrderedDict([
            ('rnn', rnn_model),
            ('fc2', fc2),
            ]))

    # if use_gpu:
    #     model.cuda()
    model.to(device)
    
    optim_method = optim.Adam(params=model.parameters())

    iter_cnt = 0
    while iter_cnt < max_iter:
        train_inputs, train_targets = shufflelists(train_X, train_Y)
        optimizer = exp_lr_scheduler(optim_method, iter_cnt, init_lr=0.01, lr_decay_epoch=3)
        for i in range(num_batch):
            low_index = batch_size*i
            high_index = batch_size*(i+1)
            if low_index <= len(train_inputs)-batch_size:
                batch_inputs = train_inputs[low_index:high_index].reshape(batch_size, time_window, 1).astype(np.float32)
                batch_targets = train_targets[low_index:high_index].reshape((batch_size, 1)).astype(np.float32)
            else:
                batch_inputs = train_inputs[low_index:].astype(float)
                batch_targets = train_targets[low_index:].astype(float)

            batch_inputs = torch.from_numpy(batch_inputs).to(device)
            batch_targets = torch.from_numpy(batch_targets).to(device)
            
            # if use_gpu:
            #     batch_inputs = batch_inputs.cuda()
            #     batch_targets = batch_targets.cuda()

            model.train(True)
            model.zero_grad()
            train_loss, logits = compute_loss_accuracy(loss_fn=loss_fn, data=batch_inputs, label=batch_targets)
            train_loss.backward()
            optimizer.step()
            
            if i % 20 == 0:
                print('the %dth iter, the %dth batch, train loss is %.4f' % (iter_cnt, i, train_loss.item()))

        # save model
        save_path = '{}/{}'.format(save_dir, int(round(connectivity/.01)))
        if os.path.exists(save_path):
            torch.save(model, os.path.join(save_path, str(iter_cnt)+'.pt'))
        else:
            os.makedirs(save_path)
            torch.save(model, os.path.join(save_path, str(iter_cnt)+'.pt'))
        iter_cnt += 1
    
    
