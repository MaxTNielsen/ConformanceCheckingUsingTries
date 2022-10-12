#%%
from logs import get_traces
import LSTM_model
import util

import time

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

inputs, targets = util.create_dataset(traces_dict, labels_to_idx)

BATCH_SIZE = 20

train_dataloader, eval_dataloader, test_dataloader = util.load_data(
    len(targets), 0.2, BATCH_SIZE, inputs, targets)

# Setting hyper parameters for the model:
INPUT_SIZE = 36
HIDDEN_SIZE = 254
NUM_LAYERS = 1
NUM_CLASSES = len(labels)
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.033

lstm = LSTM_model.LSTM(NUM_CLASSES, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)

if LSTM_model.use_cuda:
    lstm.cuda()

criterion = nn.NLLLoss()
optimizer = t.optim.Adam(lstm.parameters(), lr=LEARNING_RATE,
                         weight_decay=WEIGHT_DECAY, amsgrad=True)

t.nn.init.xavier_uniform_(lstm.fc1.weight)
t.nn.init.xavier_uniform_(lstm.fc2.weight)

# Setup settings for training
NUM_EPOCHS = 1000
EVAL_EVERY = 20

train_iter = []
train_loss = []

eval_loss = {}

start = time.time()
for epoch in range(NUM_EPOCHS):
  
    # Eval network
    if epoch % EVAL_EVERY == 0:
        eval_loss[epoch] = []
        
        lstm.eval()
        for eval_batch_index, (eval_batch, eval_target) in enumerate(eval_dataloader):

            eval_outputs = lstm(LSTM_model.get_variable(eval_batch))

            loss = t.sqrt(
            criterion(eval_outputs['out'], LSTM_model.get_variable(eval_target)))

            eval_loss[epoch].append(LSTM_model.get_numpy(loss).item())

    train_loss_epoch = []

    # Train network
    lstm.train()
    for batch_train_index, (train_batch, train_target) in enumerate(train_dataloader):

        train_outputs = lstm(LSTM_model.get_variable(train_batch))

        optimizer.zero_grad()

        loss = t.sqrt(
            criterion(train_outputs['out'], LSTM_model.get_variable(train_target)))

        train_iter.append(batch_train_index)
        train_loss_epoch.append(LSTM_model.get_numpy(loss).item())
        train_loss.append(LSTM_model.get_numpy(loss).item())

        loss.backward()
        optimizer.step()

    if epoch % EVAL_EVERY == 0:
        print("time {}".format(util.timeSince(start)))
        print("Eval loss {} at epoch {}".format(round(mean(eval_loss[epoch]),5), epoch))
        print("Train loss {} at epoch {}".format(round(mean(train_loss_epoch),5), epoch))
        print("#"*80)
        print("\n")

fig = plt.figure(figsize=(10,5))
plt.plot(train_loss, label="Train loss in each epoch")

eval_x = list(eval_loss.keys())
eval_y = list(map(lambda x: mean(x), list(eval_loss.values())))
plt.plot(eval_x, eval_y, label="Eval loss in each epoch")
plt.ylabel("NLLLoss")
fig.tight_layout()
plt.legend(loc='upper right')
plt.show()

eval_best_idx = np.argmin(np.array(eval_y))
eval_best = eval_y[eval_best_idx]
eval_best_epoch = eval_x[eval_best_idx]

print("Min eval loss {} at epoch {}".format(round(eval_best,5), eval_best_epoch))
# %%
