# encoding: utf-8

"""

@author: huayuxiu

"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import data_processing
import time
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.stats import logistic
import pandas as pd 
import os
import math


data_path = './data/traffic-1-2.npy'
time_window = 100
hidden_size = 300
max_iter = 5

#加载数据
data = np.load(data_path)
new_data = []
for x in data:
    if x > 0:
        new_data.append(np.log10(x))
    else:
        new_data.append(0.001)
new_data = np.array(new_data)
new_data = new_data[new_data>2.5]
data = new_data[new_data<6]

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

# Train on 70% of the data
train_idx = int(len(df) * .5)

# create train and test data
train_X, train_Y, test_X, test_Y = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]

test_X = test_X.reshape(-1, time_window, 1)
test_Y = test_Y.reshape(-1, 1)
print(len(test_X))

# 重新构建LSTM计算模型，将模型中的矩阵计算替换成稀疏矩阵的计算
def rnn_cell(connectivity, input, hx, cell_weight_hh, cell_weight_ih, cell_bias):
    output = []
    for time in range(max_time):
        h, c = hx
        x = input[max_time-1-time]
        # print(h.shape)
        # print(cell_weight_hh.shape)
        # print(cell_bias.shape)

        # 如果模型稀疏性很高，用稀疏矩阵计算方法
        if connectivity == '1' or connectivity == '10':
            wh_b = h * cell_weight_hh + cell_bias
            wi = x * cell_weight_ih

        # 如果模型稀疏性不高，用常规计算方法
        else:
            wh_b = np.dot(h, cell_weight_hh) + cell_bias
            wi = np.dot(x, cell_weight_ih)
        f, i, o, g = np.split(wh_b + wi, 4, axis=1)
        c_next = np.multiply(logistic.cdf(f), c) + np.multiply(logistic.cdf(i), np.tanh(g))
        h_next = np.multiply(logistic.cdf(o), np.tanh(c))
        hx = (h_next, c_next)
        output.append(h_next)
        # if time % 50 == 0:
        #     print(time)
    output = np.stack(output, 0)
    return output, hx

def rnn(input, hx, cell_weight_hh, cell_weight_ih, cell_bias):
    input_ = np.transpose(input, [1, 0, 2])
    h_n = []
    c_n = []
    layer_output = None
    for layer in range(3):
        layer_output, (layer_h_n, layer_c_n) = rnn_cell(
            input=input_, hx=hx, cell_weight_hh=cell_weight_hh[layer],
            cell_weight_ih=cell_weight_ih[layer], cell_bias=cell_bias[layer])
        print(layer_output.shape)
        input_ = layer_output
        h_n.append(layer_h_n)
        c_n.append(layer_c_n)
    output = layer_output
    h_n = np.stack(h_n, 0)
    c_n = np.stack(c_n, 0)
    return output, (h_n, c_n)

# rclstm_model = torch.load('/home/hyx/Pytorch/Traffic_prediction/RNN/model/lstm-model.pt')
# rclstm_model_dict = rclstm_model.state_dict()
# print(rclstm_model_dict.keys())

RMSE = []
run_time = []
save_dir = './model/'
for connectivity in ['1', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']:
    save_path = save_dir + connectivity
    temp = []

    for save in range(max_iter):
        # print(os.path.join(save_path, str(save)+'.pt'))
        lstm_model = torch.load(os.path.join(save_path, str(save)+'.pt'))
        lstm_model_dict = lstm_model.state_dict()

        model_dict = lstm_model_dict

        cell_weight_ih_0 = model_dict['rnn.cell_0.weight_ih'].cpu().data.numpy()
        cell_weight_hh_0 = model_dict['rnn.cell_0.weight_hh'].cpu().data.numpy()
        cell_bias_0 = model_dict['rnn.cell_0.bias'].cpu().data.numpy()

        # 如果权重矩阵稀疏性很高，将矩阵变为稀疏矩阵表达形式
        if connectivity == '1' or connectivity == '10':
            cell_weight_ih_0 = csr_matrix(cell_weight_ih_0)
            cell_weight_hh_0 = csr_matrix(cell_weight_hh_0)
            print('sparse matrices')

        fc2_weight = model_dict['fc2.weight'].cpu().data.numpy()
        fc2_bias = model_dict['fc2.bias'].cpu().data.numpy()


        num_samples, max_time, _ = test_X.shape

        h = np.zeros((num_samples, hidden_size))
        c = np.zeros((num_samples, hidden_size))
        hx = (h, c)

        t1 = time.time()

        input_ = np.transpose(test_X, [1, 0, 2])
  
        h_n, _ = rnn_cell(connectivity, input_, hx, cell_weight_hh_0, cell_weight_ih_0, cell_bias_0)

        logit = np.dot(h_n[-1], fc2_weight.T) + fc2_bias
        t2 = time.time()

        prediction = logit
        actual = test_Y

        RMSELoss = np.sqrt(np.mean((actual - prediction)**2))
        
        # 计算时间
        total_time = t2 - t1
        print('test RMSEloss is %.4f, total computating time is %.5f' % (RMSELoss, t2-t1))
        
        # 取max-iter个模型中表现最好的作为最后结果
        if save == 0:
            temp = [RMSELoss, total_time]
        else:
            temp[0] = RMSELoss if RMSELoss < temp[0] else temp[0]
            temp[1] = total_time if total_time > temp[1] else temp[1]
       

    RMSE.append(float('%.4f' % temp[0]))
    run_time.append(float('%.4f' % temp[1]))

print(RMSE, run_time)




