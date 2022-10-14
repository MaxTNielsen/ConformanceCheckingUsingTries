# %%
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
    'labels': dict,
    'labels_to_idx': dict,
    'idx_to_labels': dict,
    'train_dataloader': t.utils.data.dataloader.DataLoader,
    'eval_dataloader': t.utils.data.dataloader.DataLoader,
    'test_dataloader': t.utils.data.dataloader.DataLoader,
    'model': LSTM_model.LSTM.__class__
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
    HIDDEN_SIZE = 254
    NUM_LAYERS = 1
    NUM_CLASSES = len(store['labels'])
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

                eval_outputs = lstm(LSTM_model.get_variable(eval_batch))

                loss = t.sqrt(
                    criterion(eval_outputs['out'], LSTM_model.get_variable(eval_target)))

                eval_loss[epoch].append(LSTM_model.get_numpy(loss).item())

        train_loss_epoch = []

        # Train network
        lstm.train()
        for batch_train_index, (train_batch, train_target) in enumerate(store['train_dataloader']):

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


def run_test() -> LSTM_model.LSTM:
    global store
    criterion = nn.NLLLoss()
    test_iter = []
    test_loss = []
    for batch_test_index, (test_batch, test_target) in enumerate(store['test_dataloader']):
        
        test_outputs = store['model'](LSTM_model.get_variable(test_batch))

        loss = t.sqrt(
            criterion(test_outputs['out'], LSTM_model.get_variable(test_target)))

        test_iter.append(batch_test_index)
        test_loss.append(LSTM_model.get_numpy(loss).item())

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
    HIDDEN_SIZE = 254
    NUM_LAYERS = 1
    NUM_CLASSES = len(store['labels'])
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
        for batch_train_index, (train_batch, train_target) in enumerate(store['train_dataloader']):

            train_outputs = lstm(LSTM_model.get_variable(train_batch))

            optimizer.zero_grad()

            loss = t.sqrt(
                criterion(train_outputs['out'], LSTM_model.get_variable(train_target)))

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
    max_output_idx = np.argmax(output)
    max_output = output[max_output_idx]
    p = max_output
    s = store['idx_to_labels'][max_output_idx]
    k = s
    output_idx = store['labels_to_idx'][input["target"]]
    output_ = output[output_idx].item()
    # return [output_, max_output]
    return output_


if __name__ == "__main__":
    FILE_NAME = "logs/M1.xes"
    epochs = get_model_param(FILE_NAME)
    get_model(epochs)
    run_test()
# %%


# out
# tensor([[-3.3229, -4.0123, -4.1178, -3.5945, -4.0334, -3.9988, -4.0593, -4.1274,
#          -3.9292, -1.2458, -4.0401, -3.9725, -4.0404, -3.9891, -3.8863, -3.9007,
#          -4.1528, -3.8372, -4.2687, -3.8359, -4.0050, -4.3265, -3.9911, -3.9496,
#          -4.2549, -3.2833, -3.0695, -3.3123, -3.9639, -4.0928, -4.2169, -4.1232,
#          -4.0305, -3.9399, -4.3716, -3.9660]])

# exp(out)
# tensor([0.0360, 0.0181, 0.0163, 0.0275, 0.0177, 0.0183, 0.0173, 0.0161, 0.0197,
#         0.2877, 0.0176, 0.0188, 0.0176, 0.0185, 0.0205, 0.0202, 0.0157, 0.0216,
#         0.0140, 0.0216, 0.0182, 0.0132, 0.0185, 0.0193, 0.0142, 0.0375, 0.0464,
#         0.0364, 0.0190, 0.0167, 0.0147, 0.0162, 0.0178, 0.0194, 0.0126, 0.0189])

# max_output_idx = 9
# max_output_label = 'B'
# max_output_exp = 0.28769872

# output_idx = 19
# label = 'A'
# output_ = 0.0215808916836977