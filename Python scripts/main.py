# %%
from log_parser import get_traces
import model
import utils

import time

import optuna

import json

from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch
from torch import optim

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
    'test_dataset': utils.Dataset.__class__,
    'train_dataset': utils.Dataset.__class__,
    'eval_dataset': utils.Dataset.__class__
}


def load_model():
    with open('saved_models/'+store['file_name'].split('/')[-1]+'.model.hyperparams.json', 'r') as openfile:
        params = json.load(openfile)

    lstm = build_model(params=params)

    lstm.load_state_dict(
        torch.load('saved_models/'+store['file_name'].split('/')[-1]+'.model.pt'))

    # _, epochs, _  = get_optimal_model(params)
    # store['model_params']['epochs'] = epochs
    # _, _, lstm = get_optimal_model(params)

    # store optimal model for prediction task
    store['model'] = lstm


def get_dataloaders():
    store['train_dataloader'] = torch.load(
        "saved_models/dataloaders/" + store['file_name'].split('/')[-1]+".train_samp.pt")
    store['eval_dataloader'] = torch.load(
        "saved_models/dataloaders/" + store['file_name'].split('/')[-1]+".eval_samp.pt")
    store['test_dataloader'] = torch.load(
        "saved_models/dataloaders/" + store['file_name'].split('/')[-1]+".test_samp.pt")


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
    store['model_params']['epochs'] = 100


def get_model(params, lstm):
    """hyperparameter tuning"""

    global store

    # inputs, targets = utils.create_dataset(
    #     store['traces_dict'], store['labels_to_idx'])

    # train, test = utils.load_data(0.2, inputs, targets)

    # store['test_dataset'] = test

    # val_size = int(0.2 * len(train))
    # train_size = len(train) - val_size

    # train_data, val_data = random_split(
    #     train, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(
        store['train_dataset'], batch_size=params['batch_size'], shuffle=False, collate_fn=utils.collate_fn)

    val_dataloader = DataLoader(
        store['eval_dataset'], batch_size=params['batch_size'], shuffle=False, collate_fn=utils.collate_fn)

    # store['train_dataloader'] = train_dataloader
    # store['eval_dataloader'] = val_dataloader

    if model.use_cuda:
        lstm.cuda()

    criterion = nn.NLLLoss()

    optimizer = getattr(optim, params['optimizer'])(lstm.parameters(
    ), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    torch.nn.init.xavier_uniform_(lstm.fc1.weight)
    torch.nn.init.xavier_uniform_(lstm.fc2.weight)

    # Setup settings for training
    NUM_EPOCHS = 30

    eval_loss = {}

    for epoch in range(NUM_EPOCHS):

        # Train network
        lstm.train()
        for batch_train_index, (train_batch, train_target) in enumerate(train_dataloader):

            train_outputs = lstm(model.get_variable(train_batch))

            ys = model.get_variable(train_target)
            y_hat = train_outputs['out']

            loss = 0
            for i, y in enumerate(ys):
                loss += criterion(y_hat[i], y)

            loss = loss / len(train_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Eval network
        eval_loss[epoch] = []

        lstm.eval()
        for eval_batch_index, (eval_batch, eval_target) in enumerate(val_dataloader):

            eval_outputs = lstm(model.get_variable(eval_batch))

            ys = model.get_variable(eval_target)
            y_hat = eval_outputs['out']

            loss = 0
            for i, y in enumerate(ys):
                loss += criterion(y_hat[i], y)

            loss = loss / len(train_target)

            eval_loss[epoch].append(model.get_numpy(loss).item())

    return mean(list(eval_loss.values())[-1])


def get_optimal_model(params: dict):
    """train and eval model and save model state (weights) for each epoch using optimal model from hyperparameter tuning experiment. Retrieve model state from best epoch trial."""

    global store

    lstm = build_model(params=params)

    if model.use_cuda:
        lstm.cuda()

    criterion = nn.NLLLoss()
    # optimizer = torch.optim.Adam(lstm.parameters(), lr=params['learning_rate'],
    #                              weight_decay=params['weight_decay'], amsgrad=True)

    optimizer = getattr(optim, params['optimizer'])(lstm.parameters(
    ), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    torch.nn.init.xavier_uniform_(lstm.fc1.weight)
    torch.nn.init.xavier_uniform_(lstm.fc2.weight)

    # Setup settings for training
    NUM_EPOCHS = store['model_params']['epochs']
    EVAL_EVERY = 10

    best_model_state = dict()
    eval_loss = dict()

    train_iter = []
    train_loss = []

    start = time.time()
    for epoch in range(NUM_EPOCHS):

        # Train network
        train_loss_epoch = []
        lstm.train()
        for batch_train_index, (train_batch, train_target) in enumerate(store['train_dataloader']):

            train_outputs = lstm(model.get_variable(train_batch))

            ys = model.get_variable(train_target)
            y_hat = train_outputs['out']

            loss = 0
            for i, y in enumerate(ys):
                loss += criterion(y_hat[i], y)

            loss = loss / len(train_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_iter.append(batch_train_index)
            train_loss_epoch.append(model.get_numpy(loss).item())
            train_loss.append(model.get_numpy(loss).item())

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

            loss = loss / len(train_target)

            # eval_loss[epoch].append(model.get_numpy(loss).item())
            eval_loss[epoch].append(model.get_numpy(loss).item())

        best_model_state[epoch] = copy.deepcopy(lstm.state_dict())

        if mean(eval_loss[epoch]) < 0.1:
            break

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
    eval_y = list(map(lambda x: mean(x), list(eval_loss.values())))
    eval_best_idx = np.argmin(np.array(eval_y))
    eval_best = eval_y[eval_best_idx]
    eval_best_epoch = eval_x[eval_best_idx]

    print("Min eval loss {} at epoch {}".format(
        round(eval_best, 5), eval_best_epoch))

    return best_model_state[eval_best_epoch], eval_best_epoch, lstm


def run_test():
    """validate optimal model on independent test data, set aside from the train, eval and test split"""

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

        loss = loss / len(test_target)

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
    trace = [label for label in input['trace']
             if label in store['labels_to_idx']]
    tensor = utils.preprocess_input(
        trace=trace, label_to_idx=store['labels_to_idx'])
    store['model'].eval()
    output = store['model'](tensor)
    output = torch.exp(output['out'][0])
    output = output.detach().numpy()
    targets = input['trace'][1:]+[input['target']]
    targets = [label for label in targets if label in store['labels_to_idx']]
    output_ = 0
    for idx, target in enumerate(targets):
        if target in store['labels_to_idx']:
            output_idx = store['labels_to_idx'][target]
            output_ += output[idx][output_idx].item()
        else:
            output_ += np.mean(output[idx])
    output_ = output_ / len(targets)
    return output_


def build_model(params: dict):
    return model.LSTM(store['model_params']['num_classes'], store['model_params']['input_size'], params)


def objective(trial):
    global store

    params = {
        'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        'learning_rate': trial.suggest_float("learning_rate", 1e-5, 1e-1),
        'weight_decay': trial.suggest_float("weight_decay", 0.0, 1e-4),
        'dropout_rate': trial.suggest_float("dropout_rate", 0.0, 1.0),
        'n_unit': trial.suggest_int("n_unit", 50, 250, step=50),
        'num_layers': trial.suggest_int("num_layers", 1, 3),
        'bi': trial.suggest_int("bi", 0, 0),
        'batch_size': trial.suggest_int("batch_size", 10, 30, step=5)
    }

    model = build_model(params)

    loss = get_model(params, model)

    return loss


def init(file_name: str.__class__):
    """builds new model if model state 'filename.model.pt' is not stored. Else load stored model state from memory"""

    global store
    store['file_name'] = file_name
    if not exists('saved_models/'+store['file_name'].split('/')[-1]+'.model.pt'):
        # if not exists('saved_models/'+store['file_name'].split('/')[-1]+'.model.hyperparams.json'):
        build_traces_dicts(filename=store['file_name'])
        set_params()

        inputs, targets = utils.create_dataset(
            store['traces_dict'], store['labels_to_idx'])

        train, eval, test = utils.load_data(0.2, inputs, targets)

        store['train_dataset'] = train
        store['eval_dataset'] = eval
        store['test_dataset'] = test

        # begin hyperparameter tuning experiments
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=30)
        best_trial = study.best_trial

        for key, value in best_trial.params.items():
            print("{}: {}".format(key, value))

        # build test_dataloader and then store it in global dict
        train_dataloader = DataLoader(
            store['train_dataset'], batch_size=best_trial.params['batch_size'], shuffle=False, collate_fn=utils.collate_fn)

        val_dataloader = DataLoader(
            store['eval_dataset'], batch_size=best_trial.params['batch_size'], shuffle=False, collate_fn=utils.collate_fn)

        test_dataloader = DataLoader(
            store['test_dataset'], batch_size=best_trial.params['batch_size'], shuffle=False, collate_fn=utils.collate_fn)

        store['train_dataloader'] = train_dataloader
        store['eval_dataloader'] = val_dataloader
        store['test_dataloader'] = test_dataloader

        # save the dataloaders for consistency
        torch.save(store['train_dataloader'], "saved_models/dataloaders/" +
                   store['file_name'].split('/')[-1]+".train_samp.pt")
        torch.save(store['eval_dataloader'], "saved_models/dataloaders/" +
                   store['file_name'].split('/')[-1]+".eval_samp.pt")
        torch.save(test_dataloader, "saved_models/dataloaders/" +
                   store['file_name'].split('/')[-1]+".test_samp.pt")

        # save the optimal params for model initialisation
        with open('saved_models/'+store['file_name'].split('/')[-1]+'.model.hyperparams.json', "w") as outfile:
            json.dump(best_trial.params, outfile)

        # refit a model configured with the optimal params and retrieve model state with lowest validation loss
        best_model = build_model(params=best_trial.params)
        best_model_state, _, _ = get_optimal_model(best_trial.params)

        # _, epochs, _  = get_optimal_model(best_trial.params)
        # store['model_params']['epochs'] = epochs
        # _, _, best_model = get_optimal_model(best_trial.params)

        # save model state (weigths etc)
        torch.save(best_model_state, 'saved_models/' +
                   store['file_name'].split('/')[-1]+'.model.pt')

        # store optimal model for prediction task
        best_model.load_state_dict(best_model_state)
        store['model'] = best_model

    else:
        build_traces_dicts(store['file_name'])
        set_params()
        get_dataloaders()
        load_model()


if __name__ == "__main__":
    FILE_NAME = "input/M-models/M1.xes"
    init(FILE_NAME)
    run_test()
# %%
