# %%
from contextlib import nullcontext
from log_parser import get_traces
import old.model as model
import utils

import time

from os.path import exists

import copy

import torch.nn as nn
import torch as torch

from statistics import mean

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


store = {
    'labels': dict,
    'labels_to_idx': dict,
    'idx_to_labels': dict,
    'train_dataloader': torch.utils.data.dataloader.DataLoader,
    'eval_dataloader': torch.utils.data.dataloader.DataLoader,
    'test_dataloader': torch.utils.data.dataloader.DataLoader,
    'model': model.LSTM.__class__,
    'batchsize': str.__class__,
    'filename': str.__class__,
    'traces_dict': dict,
    'model_params': dict
}


def load_model():
    lstm = model.LSTM(store['model_params']['num_classes'], store['model_params']['input_size'],
                      store['model_params']['hidden_size'], store['model_params']['num_layers'])
    lstm.load_state_dict(
        torch.load('saved_models/'+store['file_name'].split('/')[-1]+'.model.pt'))
    store['model'] = lstm


def get_test_dataloader():
    """used for validating the model on the test data afer instatiating a model with loaded state"""

    store['test_dataloader'] = torch.load(
        "saved_models/test_dataloaders/" + store['file_name'].split('/')[-1]+".test_samp.pt")


def build_traces_dicts(filename: str.__class__):
    global store

    traces_dict = get_traces(filename=filename)
    store['traces_dict'] = traces_dict

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


def set_params():
    """params initialisation for model configuration"""

    global store
    store['model_params'] = {}
    store['model_params']['input_size'] = len(store['labels'])
    store['model_params']['hidden_size'] = 80
    store['model_params']['num_layers'] = 1
    store['model_params']['num_classes'] = store['model_params']['input_size']
    store['model_params']['learning_rate'] = 0.00075316416466055
    store['model_params']['weight_decay'] = 0.033
    store['batchsize'] = 10


def build():
    """build datasets, dataloaders for train, eval and test. Also stores some globals"""

    global store
    inputs, targets = utils.create_dataset(
        store['traces_dict'], store['labels_to_idx'])

    train, eval, test = utils.load_data(
        len(targets), 0.2, store['batchsize'], inputs, targets)

    torch.save(test, "saved_models/test_dataloaders/" +
           store['file_name'].split('/')[-1]+".test_samp.pt")

    store['train_dataloader'] = train
    store['eval_dataloader'] = eval
    store['test_dataloader'] = test


def get_model(filename: str.__class__):
    """train and eval model and save model state (weights) for each epoch. Retrieve model state from best epoch and store model in global store"""

    global store
    build_traces_dicts(filename=filename)
    set_params()
    build()

    # Setting hyper parameters for the model:
    INPUT_SIZE = store['model_params']['input_size']
    HIDDEN_SIZE = store['model_params']['hidden_size']
    NUM_LAYERS = store['model_params']['num_layers']
    NUM_CLASSES = store['model_params']['num_classes']
    LEARNING_RATE = store['model_params']['learning_rate']
    WEIGHT_DECAY = store['model_params']['weight_decay']

    lstm = model.LSTM(NUM_CLASSES, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)

    if model.use_cuda:
        lstm.cuda()

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY, amsgrad=True)

    torch.nn.init.xavier_uniform_(lstm.fc1.weight)
    torch.nn.init.xavier_uniform_(lstm.fc2.weight)

    # Setup settings for training
    NUM_EPOCHS = 300
    EVAL_EVERY = 10

    train_iter = []
    train_loss = []

    eval_loss = {}
    best_model_state = {}

    start = time.time()
    for epoch in range(NUM_EPOCHS):

        # Eval network
        if epoch % EVAL_EVERY == 0:
            eval_loss[epoch] = []

            lstm.eval()
            for eval_batch_index, (eval_batch, eval_target) in enumerate(store['eval_dataloader']):

                eval_outputs = lstm(model.get_variable(eval_batch))

                ys = model.get_variable(eval_target)
                y_hat = eval_outputs['out']

                #loss = []
                loss = 0
                for i, y in enumerate(ys):
                    loss += criterion(y_hat[i], y)
                    #loss.append(criterion(y_hat[i], y))

               #loss = loss / BATCH_SIZE
               #loss = torch.mean(torch.tensor(loss, requires_grad=True))

                """for many-to-one"""
               #loss = criterion(train_outputs['out'], model.get_variable(train_target))

                eval_loss[epoch].append(model.get_numpy(loss).item())
                best_model_state[epoch] = copy.deepcopy(lstm.state_dict())

        train_loss_epoch = []

        # Train network
        lstm.train()
        for batch_train_index, (train_batch, train_target) in enumerate(store['train_dataloader']):

            train_outputs = lstm(model.get_variable(train_batch))

            optimizer.zero_grad()

            ys = model.get_variable(train_target)
            y_hat = train_outputs['out']

            #loss = []
            loss = 0
            for i, y in enumerate(ys):
                loss += criterion(y_hat[i], y)
                #loss.append(criterion(y_hat[i], y))

           #loss = loss / BATCH_SIZE
           #loss = torch.mean(torch.tensor(loss, requires_grad=True))

            """for many-to-one"""
           #loss = criterion(train_outputs['out'], model.get_variable(train_target))

            train_iter.append(batch_train_index)
            train_loss_epoch.append(model.get_numpy(loss).item())
            train_loss.append(model.get_numpy(loss).item())

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(lstm.parameters())
            optimizer.step()

        if epoch % EVAL_EVERY == 0:
            pass
            print("time {}".format(utils.timeSince(start)))
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

    del lstm

    best_model = model.LSTM(store['model_params']['num_classes'], store['model_params']['input_size'],
                      store['model_params']['hidden_size'], store['model_params']['num_layers'])
    best_model.load_state_dict(best_model_state[eval_best_epoch])
    store['model'] = best_model
    torch.save(store['model'].state_dict(),
           'saved_models/'+store['file_name'].split('/')[-1]+'.model.pt')
    del best_model_state


def run_test() -> model.LSTM:
    global store
    criterion = nn.NLLLoss()
    test_iter = []
    test_loss = []

    store['model'].eval()
    for batch_test_index, (test_batch, test_target) in enumerate(store['test_dataloader']):

        test_outputs = store['model'](model.get_variable(test_batch))

        ys = model.get_variable(test_target)
        y_hat = test_outputs['out']

        #loss = []
        loss = 0
        for i, y in enumerate(ys):
            loss += criterion(y_hat[i], y)
            #loss.append(criterion(y_hat[i], y))

        #loss = loss / store['batchsize']
        #loss = torch.mean(torch.tensor(loss, requires_grad=True))

        # for many-to-one
        # loss = criterion(train_outputs['out'], model.get_variable(train_target))

        test_iter.append(batch_test_index)
        test_loss.append(model.get_numpy(loss).item())

    # fig = plt.figure(figsize=(10, 5))
    # plt.plot(test_loss, label="Test loss in each epoch")
    # plt.ylabel("NLLLoss")
    # fig.tight_layout()
    # plt.legend(loc='upper right')
    # plt.show()

    print("average test loss: {}".format(round(mean(test_loss), 5)))


def init(file_name):
    """builds new model if model state '"filename".model.pt' does not exist. Else load stored model state from memory"""

    global store
    store['file_name'] = file_name
    if not exists('saved_models/'+store['file_name'].split('/')[-1]+'.model.pt'):
        get_model(filename=file_name)
    else:
        build_traces_dicts(store['file_name'])
        set_params()
        get_test_dataloader()
        load_model()


def make_prediction(input: dict) -> float:
    global store
    tensor = utils.preprocess_input(
        trace=input['trace'], label_to_idx=store['labels_to_idx'])
    store['model'].eval()
    output = store['model'](tensor)
    output = torch.exp(output['out'][0])
    output = output.detach().numpy()
    if input['target'] in store['labels_to_idx']:
        output_idx = store['labels_to_idx'][input['target']]
        output_ = output[-1][output_idx].item()
    else:
        output_ = np.mean(output[-1])
    return output_


if __name__ == "__main__":
    FILE_NAME = "input/M-models/M1.xes"
    init(file_name=FILE_NAME)
    run_test()
# %%

# out
# tensor([[5.2227e-02, 1.5767e-05, 3.8406e-05, 1.2556e-02, 7.3353e-01, 1.3051e-04,
#          1.8213e-04, 2.1693e-05, 1.6332e-02, 7.1894e-05, 1.0421e-02, 1.4459e-05,
#          7.6112e-02, 5.2439e-03, 3.1881e-05, 2.7352e-03, 5.5395e-03, 1.4748e-03,
#          3.3418e-05, 1.0223e-02, 3.4732e-05, 5.5473e-03, 4.6181e-05, 1.5196e-03,
#          1.9349e-04, 3.1653e-05, 6.2700e-02, 1.7104e-04, 8.2270e-04, 3.3331e-05,
#          6.7925e-04, 1.0769e-03, 5.2478e-05, 6.2348e-06, 1.2742e-04, 2.5622e-05]])

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


# max_output_idx = np.argmax(output)
    # max_output = output[max_output_idx]
    # p = max_output
    # s = store['idx_to_labels'][max_output_idx]
