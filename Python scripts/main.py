# %%
from distutils.command.build import build
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


store = {
    "labels": dict,
    "train_dataloader": t.utils.data.dataloader.DataLoader,
    "eval_dataloader": t.utils.data.dataloader.DataLoader,
    "test_dataloader": t.utils.data.dataloader.DataLoader,
    "model": LSTM_model.LSTM.__class__,
    "labels_to_idx": dict
}


def model_build(filename) -> int:
    global store
    FILE_NAME = filename

    traces_dict = {int(key[9:]): val for key,
                   val in get_traces(filename=FILE_NAME).items()}

    # build maps to map from and to activity labels and unique ids
    store["labels"] = {val for values in traces_dict.values()
                       for val in values}

    labels_to_idx = {}
    idx_to_labels = {}

    for i, label in enumerate(store["labels"]):
        labels_to_idx[label] = i
        idx_to_labels[i] = label

    store["labels_to_idx"] = labels_to_idx
    inputs, targets = util.create_dataset(traces_dict, labels_to_idx)

    BATCH_SIZE = 20

    train, eval, test = util.load_data(
        len(targets), 0.2, BATCH_SIZE, inputs, targets)

    store["train_dataloader"] = train
    store["eval_dataloader"] = eval
    store["test_dataloader"] = test

    # Setting hyper parameters for the model:
    INPUT_SIZE = 36
    HIDDEN_SIZE = 254
    NUM_LAYERS = 1
    NUM_CLASSES = len(store["labels"])
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
    NUM_EPOCHS = 100
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
            for eval_batch_index, (eval_batch, eval_target) in enumerate(store["eval_dataloader"]):

                eval_outputs = lstm(LSTM_model.get_variable(eval_batch))

                loss = t.sqrt(
                    criterion(eval_outputs['out'], LSTM_model.get_variable(eval_target)))

                eval_loss[epoch].append(LSTM_model.get_numpy(loss).item())

        train_loss_epoch = []

        # Train network
        lstm.train()
        for batch_train_index, (train_batch, train_target) in enumerate(store["train_dataloader"]):

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
            pass
            print("time {}".format(util.timeSince(start)))
            print("Eval loss {} at epoch {}".format(
                round(mean(eval_loss[epoch]), 5), epoch))
            print("Train loss {} at epoch {}".format(
                round(mean(train_loss_epoch), 5), epoch))
            print("#"*80)
            print("\n")

    eval_x = list(eval_loss.keys())
    eval_y = list(map(lambda x: mean(x), list(eval_loss.values())))

    # fig = plt.figure(figsize=(10, 5))
    # plt.plot(train_loss, label="Train loss in each epoch")
    # plt.plot(eval_x, eval_y, label="Eval loss in each epoch")
    # plt.ylabel("NLLLoss")
    # fig.tight_layout()
    # plt.legend(loc='upper right')
    # plt.show()

    eval_best_idx = np.argmin(np.array(eval_y))
    eval_best = eval_y[eval_best_idx]
    eval_best_epoch = eval_x[eval_best_idx]

    print("Min eval loss {} at epoch {}".format(
        round(eval_best, 5), eval_best_epoch))

    return eval_best_epoch


def run_test(lstm: LSTM_model.LSTM) -> LSTM_model.LSTM:
    criterion = nn.NLLLoss()
    test_iter = []
    test_loss = []
    for batch_test_index, (test_batch, test_target) in enumerate(store["test_dataloader"]):

        train_outputs = lstm(LSTM_model.get_variable(test_batch))

        loss = t.sqrt(
            criterion(train_outputs['out'], LSTM_model.get_variable(test_target)))

        test_iter.append(batch_test_index)
        test_loss.append(LSTM_model.get_numpy(loss).item())

    # fig = plt.figure(figsize=(10, 5))
    # plt.plot(test_loss, label="Test loss in each epoch")
    # plt.show()


def run(epochs):
    global store
    # Setting hyper parameters for the model:
    INPUT_SIZE = 36
    HIDDEN_SIZE = 254
    NUM_LAYERS = 1
    NUM_CLASSES = len(store["labels"])
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
    NUM_EPOCHS = epochs

    for epoch in range(NUM_EPOCHS):

        # Train network
        lstm.train()
        for batch_train_index, (train_batch, train_target) in enumerate(store["train_dataloader"]):

            train_outputs = lstm(LSTM_model.get_variable(train_batch))

            optimizer.zero_grad()

            loss = t.sqrt(
                criterion(train_outputs['out'], LSTM_model.get_variable(train_target)))

            loss.backward()
            optimizer.step()

    store["model"] = lstm


def init(file_name):
    epochs = model_build(file_name)
    run(epochs)


def make_prediction(input: dict) -> float:
    global store
    tensor = util.preprocess_input(trace=input["trace"])
    output = store["model"](tensor)
    output = t.exp(output['out'][0])
    output = output.detach().numpy()
    output_idx = store["labels_to_idx"][input["target"]]
    # max_output_idx = np.argmax(output)
    # max_output = output[max_output_idx].item()
    output_ = output[output_idx].item()
    # return [output_, max_output]
    return output_


if __name__ == "__main__":
    FILE_NAME = "logs/M1.xes"
    epochs = model_build(FILE_NAME)
    store["model"] = run(epochs)
    # run_test(model)
# %%
