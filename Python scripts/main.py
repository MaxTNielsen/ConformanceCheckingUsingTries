# %%
from log_parser import get_traces
import model
import utils

import optuna

import json

from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch

from os.path import exists

import copy

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
    'filename': str.__class__,
    'traces_dict': dict,
    'model_params': dict,
    'test_dataset': utils.Dataset.__class__
}


def load_model():

    with open('saved_models/'+store['file_name'].split('/')[-1]+'.model.hyperparams.json', 'r') as openfile:
        params = json.load(openfile)

    lstm = build_model(params=params)

    lstm.load_state_dict(
        torch.load('saved_models/'+store['file_name'].split('/')[-1]+'.model.pt'))

    store['model'] = lstm


def get_test_dataloader():
    """used for validating the model on the test data afer instatiating a model with loaded state"""

    store['test_dataloader'] = torch.load(
        "saved_models/test_dataloaders/" + store['file_name'].split('/')[-1]+".test_samp.pt")


def build_traces_dicts(filename: str.__class__):
    global store
    store['file_name'] = filename

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
    store['model_params']['num_classes'] = store['model_params']['input_size']


def get_model(params, lstm):
    """hyperparameter tuning"""

    global store

    inputs, targets = utils.create_dataset(
        store['traces_dict'], store['labels_to_idx'])

    train, test = utils.load_data(0.2, inputs, targets)

    store['test_dataset'] = test

    val_size = int(0.2 * len(train))
    train_size = len(train) - val_size

    train_data, val_data = random_split(
        train, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(
        train_data, batch_size=params['batch_size'], shuffle=False, collate_fn=utils.collate_fn)

    val_dataloader = DataLoader(
        val_data, batch_size=params['batch_size'], shuffle=False, collate_fn=utils.collate_fn)

    store['train_dataloader'] = train_dataloader
    store['eval_dataloader'] = val_dataloader

    if model.use_cuda:
        lstm.cuda()

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=params['learning_rate'],
                                 weight_decay=params['weight_decay'], amsgrad=True)

    torch.nn.init.xavier_uniform_(lstm.fc1.weight)
    torch.nn.init.xavier_uniform_(lstm.fc2.weight)

    # Setup settings for training
    NUM_EPOCHS = 100

    eval_loss = {}

    for epoch in range(NUM_EPOCHS):

        # Train network
        lstm.train()
        for batch_train_index, (train_batch, train_target) in enumerate(store['train_dataloader']):

            train_outputs = lstm(model.get_variable(train_batch))

            ys = model.get_variable(train_target)
            y_hat = train_outputs['out']

            loss = 0
            for i, y in enumerate(ys):
                loss += criterion(y_hat[i], y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Eval network
        eval_loss[epoch] = []

        lstm.eval()
        for eval_batch_index, (eval_batch, eval_target) in enumerate(store['eval_dataloader']):

            eval_outputs = lstm(model.get_variable(eval_batch))

            ys = model.get_variable(eval_target)
            y_hat = eval_outputs['out']

            loss = 0
            for i, y in enumerate(ys):
                loss += criterion(y_hat[i], y)

            eval_loss[epoch].append(model.get_numpy(loss).item())

    return mean(list(eval_loss.values())[-1])


def get_optimal_model(params):
    """train and eval model and save model state (weights) for each epoch using optimal model from hyperparameter tuning experiment. Retrieve model state from best epoch trial."""

    global store

    lstm = build_model(params=params)

    if model.use_cuda:
        lstm.cuda()

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=params['learning_rate'],
                                 weight_decay=params['weight_decay'], amsgrad=True)

    torch.nn.init.xavier_uniform_(lstm.fc1.weight)
    torch.nn.init.xavier_uniform_(lstm.fc2.weight)

    # Setup settings for training
    NUM_EPOCHS = 100

    best_model_state = dict()
    eval_loss = dict()

    for epoch in range(NUM_EPOCHS):

        # Train network
        lstm.train()
        for batch_train_index, (train_batch, train_target) in enumerate(store['train_dataloader']):

            train_outputs = lstm(model.get_variable(train_batch))

            ys = model.get_variable(train_target)
            y_hat = train_outputs['out']

            loss = 0
            for i, y in enumerate(ys):
                loss += criterion(y_hat[i], y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Eval network
        eval_loss[epoch] = []

        lstm.eval()
        for eval_batch_index, (eval_batch, eval_target) in enumerate(store['eval_dataloader']):

            eval_outputs = lstm(model.get_variable(eval_batch))

            ys = model.get_variable(eval_target)
            y_hat = eval_outputs['out']

            loss = 0
            for i, y in enumerate(ys):
                loss += criterion(y_hat[i], y)

            eval_loss[epoch].append(model.get_numpy(loss).item())
            best_model_state[epoch] = copy.deepcopy(lstm.state_dict())

    eval_x = list(eval_loss.keys())
    eval_y = list(map(lambda x: mean(x), list(eval_loss.values())))
    eval_best_idx = np.argmin(np.array(eval_y))
    eval_best_epoch = eval_x[eval_best_idx]

    return best_model_state[eval_best_epoch]


def run_test():
    """validate optimal model on independent test data, set aside from the train, val and test split"""

    global store

    criterion = nn.NLLLoss()
    test_iter = []
    test_loss = []

    store['model'].eval()
    for batch_test_index, (test_batch, test_target) in enumerate(store['test_dataloader']):

        test_outputs = store['model'](model.get_variable(test_batch))

        ys = model.get_variable(test_target)
        y_hat = test_outputs['out']

        loss = 0
        for i, y in enumerate(ys):
            loss += criterion(y_hat[i], y)

        test_iter.append(batch_test_index)
        test_loss.append(model.get_numpy(loss).item())

    # fig = plt.figure(figsize=(10, 5))
    # plt.plot(test_loss, label="Test loss in each epoch")
    # plt.ylabel("NLLLoss")
    # fig.tight_layout()
    # plt.legend(loc='upper right')
    # plt.show()

    print("average test loss: {}".format(round(mean(test_loss), 5)))


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


def build_model(params):
    return model.LSTM(store['model_params']['num_classes'], store['model_params']['input_size'], params)


def objective(trial):
    global store
    build_traces_dicts(filename=store['file_name'])
    set_params()

    params = {
        'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        'learning_rate': trial.suggest_float("learning_rate", 1e-5, 1e-1),
        'weight_decay': trial.suggest_float("weight_decay", 0.0, 1e-3),
        'dropout_rate': trial.suggest_float("dropout_rate", 0.0, 1.0),
        'n_unit': trial.suggest_int("n_unit", 80, 240, step=40),
        'num_layers': trial.suggest_int("num_layers", 1, 2),
        'bi': trial.suggest_int("bi", 0,1),
        'batch_size': trial.suggest_int("batch_size", 10, 50, step=10)
    }

    model = build_model(params)

    loss = get_model(params, model)

    return loss


def init(file_name):
    """builds new model if model state 'filename.model.pt' is not stored. Else load stored model state from memory"""

    global store
    store['file_name'] = file_name
    if not exists('saved_models/'+store['file_name'].split('/')[-1]+'.model.pt'):

        # begin hyperparameter tuning experiments
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=15)
        best_trial = study.best_trial

        for key, value in best_trial.params.items():
            print("{}: {}".format(key, value))

        # build test_dataloader and then store it in global dict
        test_dataloader = DataLoader(
            store['test_dataset'], batch_size=best_trial.params['batch_size'], shuffle=False, collate_fn=utils.collate_fn)

        store['test_dataloader'] = test_dataloader

        # save the test dataloader for consistency
        torch.save(test_dataloader, "saved_models/test_dataloaders/" +
                   store['file_name'].split('/')[-1]+".test_samp.pt")

        # save the optimal params for model initialisation
        with open('saved_models/'+store['file_name'].split('/')[-1]+'.model.hyperparams.json', "w") as outfile:
            json.dump(best_trial.params, outfile)
    
        # refit a model configured with the optimal params and retrieve model state with lowest validation loss
        best_model = build_model(params=best_trial.params)
        model_state = get_optimal_model(best_trial.params)

        # save model state (weigths etc)
        torch.save(model_state,'saved_models/'+store['file_name'].split('/')[-1]+'.model.pt')

        # store optimal model for prediction task
        best_model.load_state_dict(model_state)
        store['model'] = best_model

    else:
        build_traces_dicts(store['file_name'])
        set_params()
        get_test_dataloader()
        load_model()


if __name__ == "__main__":
    FILE_NAME = "input/M-models/M1.xes"
    init(FILE_NAME)
    run_test()
# %%
# M1
# Best is trial 0 with value: 17.8267240524292.
# optimizer: Adam
# learning_rate: 0.09649521228409057
# weight_decay: 0.0009190811150309003
# dropout_rate: 0.46793037652094216
# n_unit: 100
# num_layers: 1
# batch_size: 10
# average test loss: 27.0022

# LSTM(
#   (dropout): Dropout(p=0.34681952549291417, inplace=False)
#   (batchnorm): BatchNorm1d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (lstm): LSTM(36, 160, num_layers=2, batch_first=True)
#   (fc1): Linear(in_features=160, out_features=160, bias=True)
#   (fc2): Linear(in_features=160, out_features=36, bias=True)
#   (relu): ReLU()
#   (softmax): LogSoftmax(dim=1)
# )