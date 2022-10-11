from logs import get_traces

import LSTM_model

import util

import torch.nn as nn
import torch as t

from statistics import mean

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

FILE_NAME = "logs/M1.xes"

traces_dict = {int(key[9:]): val for key,
               val in get_traces(filename=FILE_NAME).items()}

# build maps to map from and to activity labels and unique ids
labels = {val for values in traces_dict.values() for val in values}

labels_to_idx = {}
idx_to_labels = {}

for i, label in enumerate(labels):
    labels_to_idx[label] = i
    idx_to_labels[i] = label

inputs, targets = util.make_data_set(traces_dict, labels_to_idx)

batch_s = 20

train_dataloader, eval_dataloader, test_dataloader = util.load_data(len(targets), 0.2, batch_s, inputs, targets)

# Setting hyper parameters for the model:
input_size = 6
hidden_size = 128
num_layers = 1
num_classes = 1
learning_rate = 0.0001
WEIGHT_DECAY = 0.033

lstm = LSTM_model.LSTM(num_classes, input_size, hidden_size, num_layers)

if LSTM_model.use_cuda:
  lstm.cuda()

criterion = nn.MSELoss() 
optimizer = t.optim.Adam(lstm.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY, amsgrad=True)

t.nn.init.xavier_uniform_(lstm.fc1.weight)
t.nn.init.xavier_uniform_(lstm.fc2.weight)

# Setup settings for training