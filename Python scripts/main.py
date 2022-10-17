# %%
from log_parser import get_traces
import model
import util

import time

import torch.nn as nn
import torch as t

from statistics import mean

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


store = {
    'labels': dict,
    'labels_to_idx': dict,
    'idx_to_labels': dict,
    'train_dataloader': t.utils.data.dataloader.DataLoader,
    'eval_dataloader': t.utils.data.dataloader.DataLoader,
    'test_dataloader': t.utils.data.dataloader.DataLoader,
    'model': model.LSTM.__class__
}


def get_model_param(filename) -> int:
    global store
    FILE_NAME = filename

    traces_dict = {int(key[9:]): val for key,
                   val in get_traces(filename=FILE_NAME).items()}

    # build maps to map from and to activity labels and unique ids
    store['labels'] = {val for values in traces_dict.values()
                       for val in values}

    labels_to_idx = {}
    idx_to_labels = {}

    for i, label in enumerate(store['labels']):
        labels_to_idx[label] = i
        idx_to_labels[i] = label

    store['labels_to_idx'] = labels_to_idx
    store['idx_to_labels'] = idx_to_labels

    inputs, targets = util.create_dataset(traces_dict, labels_to_idx)

    BATCH_SIZE = 20

    train, eval, test = util.load_data(
        len(targets), 0.2, BATCH_SIZE, inputs, targets)

    store['train_dataloader'] = train
    store['eval_dataloader'] = eval
    store['test_dataloader'] = test

    # Setting hyper parameters for the model:
    INPUT_SIZE = 36
    HIDDEN_SIZE = 128
    NUM_LAYERS = 1
    NUM_CLASSES = len(store['labels'])
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.033

    lstm = model.LSTM(NUM_CLASSES, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)

    if model.use_cuda:
        lstm.cuda()

    criterion = nn.NLLLoss()
    optimizer = t.optim.Adam(lstm.parameters(), lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY, amsgrad=True)

    t.nn.init.xavier_uniform_(lstm.fc1.weight)
    t.nn.init.xavier_uniform_(lstm.fc2.weight)

    # Setup settings for training
    NUM_EPOCHS = 300
    EVAL_EVERY = 10

    train_iter = []
    train_loss = []

    eval_loss = {}

    start = time.time()
    for epoch in range(NUM_EPOCHS):

        # Eval network
        if epoch % EVAL_EVERY == 0:
            eval_loss[epoch] = []

            lstm.eval()
            for eval_batch_index, (eval_batch, eval_target) in enumerate(store['eval_dataloader']):

                eval_outputs = lstm(model.get_variable(eval_batch))

                loss = t.sqrt(
                    criterion(eval_outputs['out'], model.get_variable(eval_target)))

                eval_loss[epoch].append(model.get_numpy(loss).item())

        train_loss_epoch = []

        # Train network
        lstm.train()
        for batch_train_index, (train_batch, train_target) in enumerate(store['train_dataloader']):

            train_outputs = lstm(model.get_variable(train_batch))

            optimizer.zero_grad()

            loss = t.sqrt(
                criterion(train_outputs['out'], model.get_variable(train_target)))

            train_iter.append(batch_train_index)
            train_loss_epoch.append(model.get_numpy(loss).item())
            train_loss.append(model.get_numpy(loss).item())

            loss.backward()
            optimizer.step()

        if epoch % EVAL_EVERY == 0:
            pass
            print("time {}".format(util.timeSince(start)))
            print("Eval loss {} at epoch {}".format(
                round(mean(eval_loss[epoch]), 5), epoch))
            print("Train loss {} at epoch {}".format(
                round(mean(train_loss_epoch), 5), epoch))
            print("#"*80)
            print("\n")

    eval_x = list(eval_loss.keys())
    eval_idx = [i*int(len(train_loss)/len(eval_x)) for i in range(len(eval_x))]
    eval_y = list(map(lambda x: mean(x), list(eval_loss.values())))

    # fig = plt.figure(figsize=(10, 5))
    # plt.plot(train_loss, label="Train loss in each epoch")
    # plt.plot(eval_idx, eval_y, label="Eval loss in each epoch")
    # plt.ylabel("NLLLoss")
    # fig.tight_layout()
    # plt.legend(loc='upper right')
    # plt.show()

    eval_best_idx = np.argmin(np.array(eval_y))
    eval_best = eval_y[eval_best_idx]
    eval_best_epoch = eval_x[eval_best_idx]

    print("Min eval loss {} at epoch {}".format(
        round(eval_best, 5), eval_best_epoch))

    store['model'] = lstm # remove

    return eval_best_epoch


def run_test() -> model.LSTM:
    global store
    criterion = nn.NLLLoss()
    test_iter = []
    test_loss = []
    for batch_test_index, (test_batch, test_target) in enumerate(store['test_dataloader']):
        
        test_outputs = store['model'](model.get_variable(test_batch))

        loss = t.sqrt(
            criterion(test_outputs['out'], model.get_variable(test_target)))

        test_iter.append(batch_test_index)
        test_loss.append(model.get_numpy(loss).item())

    # fig = plt.figure(figsize=(10, 5))
    # plt.plot(test_loss, label="Test loss in each epoch")
    # plt.ylabel("NLLLoss")
    # fig.tight_layout()
    # plt.legend(loc='upper right')
    # plt.show()


def get_model(epochs):
    global store
    # Setting hyper parameters for the model:
    INPUT_SIZE = 36
    HIDDEN_SIZE = 128
    NUM_LAYERS = 1
    NUM_CLASSES = len(store['labels'])
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.033

    lstm = model.LSTM(NUM_CLASSES, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)

    if model.use_cuda:
        lstm.cuda()

    criterion = nn.NLLLoss()
    optimizer = t.optim.Adam(lstm.parameters(), lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY, amsgrad=True)

    t.nn.init.xavier_uniform_(lstm.fc1.weight)
    t.nn.init.xavier_uniform_(lstm.fc2.weight)

    # Setup settings for training
    NUM_EPOCHS = epochs

    for epoch in range(NUM_EPOCHS):

        # Train network
        lstm.train()
        for batch_train_index, (train_batch, train_target) in enumerate(store['train_dataloader']):

            train_outputs = lstm(model.get_variable(train_batch))

            optimizer.zero_grad()

            loss = t.sqrt(
                criterion(train_outputs['out'], model.get_variable(train_target)))

            loss.backward()
            optimizer.step()

    store['model'] = lstm


def init(file_name):
    epochs = get_model_param(file_name)
    get_model(epochs)


def make_prediction(input: dict) -> float:
    global store
    tensor = util.preprocess_input(trace=input["trace"])
    output = store['model'](tensor)
    output = t.exp(output['out'][0])
    output = output.detach().numpy()
    # max_output_idx = np.argmax(output)
    # max_output = output[max_output_idx]
    # p = max_output
    # s = store['idx_to_labels'][max_output_idx]
    # k = s
    output_idx = store['labels_to_idx'][input["target"]]
    output_ = output[output_idx].item()
    # return [output_, max_output]
    return output_


if __name__ == "__main__":
    FILE_NAME = "logs/M1.xes"
    epochs = get_model_param(FILE_NAME)
    #get_model(epochs)
    run_test()
# %%

# out
# tensor([[-3.7180, -3.9482, -3.9199, -3.8766, -3.6933, -2.7918, -3.7715, -3.9515,
#          -3.8268, -3.8325, -3.9177, -3.9920, -3.6475, -3.9693, -3.8594, -3.9929,
#          -3.8086, -3.8516, -3.6172, -4.0001, -3.1120, -3.8540, -3.8572, -3.8582,
#          -3.7546, -3.8987, -2.0661, -3.2422, -3.7446, -3.8792, -3.7933, -4.0914,
#          -3.9499, -3.9910, -2.6115, -3.8883]])

# exp(out)
# tensor([0.0243, 0.0193, 0.0198, 0.0207, 0.0249, 0.0613, 0.0230, 0.0192, 0.0218,
#         0.0217, 0.0199, 0.0185, 0.0261, 0.0189, 0.0211, 0.0184, 0.0222, 0.0212,
#         0.0269, 0.0183, 0.0445, 0.0212, 0.0211, 0.0211, 0.0234, 0.0203, 0.1267,
#         0.0391, 0.0236, 0.0207, 0.0225, 0.0167, 0.0193, 0.0185, 0.0734, 0.0205])

# max_output_idx = 26
# max_output_label = 'B'
# max_output_exp = 0.12668033

# output_idx = 23
# label = 'J'
# output_ = 0.02110593020915985