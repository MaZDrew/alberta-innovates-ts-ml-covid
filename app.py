# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:47:12 2020

@author: Brayden
"""

from modules.machinelearning import data_request
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)


uni_data = data_request.getData()
uni_data.plot()

TRAIN_SPLIT = round(len(uni_data)*0.7)

tf.set_random_seed(69)

uni_data = uni_data.values

uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()

uni_data = (uni_data - uni_train_mean)/uni_train_std

univariate_past_history = 14
univariate_future_target = 0

x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT, 
                                           univariate_past_history,
                                           univariate_future_target)

x_val_uni, y_valu_uni = univariate_data(uni_data, TRAIN_SPLIT, None, 
                                        univariate_past_history,
                                        univariate_future_target)

print('Single window of past history')
print(x_train_uni[0])
print('\n Target deaths to predict')
print(y_train_uni[0])

def create_time_steps(length):
    return list(range(-length, 0))

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0
    
    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label = labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i],
                     label = labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt

show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Bruh moment')
