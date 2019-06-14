#!/usr/bin/env python3
# encoding: utf-8
'''
author: Yuxiu Hua

Generate traffic matrix
'''

import numpy as np 
import os
import time
import pandas as pd

save_datetime = False
save_traffic = False
save_normalized = False

num_day = 10772
num_nodes = 23
# The path of 'csv' files
csv_path = './data/csv/'
# Initialize the traffic matrix for all data
traffic_matrix = np.empty([num_day, num_nodes, num_nodes])
# Initialize the traffic matrix for all normalized data
normalized_traffic_matrix = np.empty([num_day, num_nodes, num_nodes])

date_time = []
i = 0
# Get the name of 'csv' files
filenames = os.listdir(csv_path)
sorted_filenames = sorted(filenames)

for filename in sorted_filenames[1:]:
    if save_datetime:
        date_time_dict = {}
        portion = filename.split('-')
        date_time_dict['year'] = portion[1]
        date_time_dict['month'] = portion[2]
        date_time_dict['day'] = portion[3]
        date_time_dict['hour'] = portion[4]
        date_time_dict['min'] = portion[5].split('.')[0]
        date_time.append(date_time_dict)
    
    traffic_matrix_per_time = pd.read_csv(csv_path + filename, delimiter = ',',  engine='python', header=None)
    traffic_matrix[i,:,:] = traffic_matrix_per_time
    i += 1

mu = np.mean(traffic_matrix)
std = np.std(traffic_matrix)
normalized_traffic_matrix = (traffic_matrix - mu) / std

# get the traffic from point 1 to point 2
# traffic_matrix = np.load('./data/traffic-matrix.npy')
time_step, src, dst = traffic_matrix.shape
traffic_1_2 = []
for i in range(time_step):
    traffic_1_2.append(traffic_matrix[i][0][1])
traffic_1_2 = np.array(traffic_1_2)
print(traffic_1_2.shape)
np.save('./data/traffic-1-2.npy', traffic_1_2)
print('traffic-1-2 has been saved')


if save_datetime:
    np.save('./data/date-time.npy', date_time)
    print('date time have been saved as npy file')

if save_traffic:
    np.save('./data/traffic-matrices.npy', traffic_matrix)
    print('traffic matrix has been saved as npy file')

if save_normalized:
    np.save('./data/normalized-traffic-matrices.npy', normalized_traffic_matrix)
    print('normalized traffic matrix has been saved as npy file')
