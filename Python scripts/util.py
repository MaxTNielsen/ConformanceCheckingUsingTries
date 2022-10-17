import torch as t
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence

import time
import math


label_to_idx: dict


class Dataset(t.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        X = self.inputs[index]
        y = self.targets[index]
        return X, y


def load_data(dataset_size, test_split, batch_size, inputs, targets):

    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size

    dataset = Dataset(inputs, targets)

    train_eval_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=t.Generator().manual_seed(42))

    # train_eval_dataset = dataset[:train_size]
    # test_dataset = dataset[train_size:]
    # train_eval_dataset = Dataset(train_eval_dataset[0], train_eval_dataset[1])
    # test_dataset = Dataset(test_dataset[0], test_dataset[1])

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    eval_size = int(test_split * len(train_eval_dataset))
    train_size = len(train_eval_dataset) - eval_size

    train_dataset, eval_dataset = random_split(
        train_eval_dataset, [train_size, eval_size], generator=t.Generator().manual_seed(42))

    # train_dataset = train_eval_dataset[:train_size]
    # eval_dataset = train_eval_dataset[train_size:]
    # train_dataset = Dataset(train_dataset[0], train_dataset[1])
    # eval_dataset = Dataset(eval_dataset[0], eval_dataset[1])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)

    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, eval_dataloader, test_dataloader


def create_dataset(traces_dict, label_to_idx_):
    global label_to_idx
    label_to_idx = label_to_idx_

    l = list(label_to_idx.values())
    n_activities = len(l)

    inputs = []
    targets = []

    # for trace in traces_dict.values():
    #     one_hot_tensor = t.zeros(len(trace), n_activities)
    #     for li, activity in enumerate(trace):
    #         one_hot_tensor[li][label_to_idx[activity]] = 1
    #     for i in range(1,len(trace)):
    #         inputs.append(one_hot_tensor[0:i, :])
    #         targets.append(t.tensor(label_to_idx[trace[i]]))

    for trace in traces_dict.values():
        one_hot_tensor = t.zeros(len(trace), n_activities)
        for li, activity in enumerate(trace):
            one_hot_tensor[li][label_to_idx[activity]] = 1
        inputs.append(one_hot_tensor[0:len(trace)-1, :])
        targets.append(t.tensor(label_to_idx[trace[-1]]))

    inputs = pad_sequence(inputs, True)

    return inputs, targets


def preprocess_input(trace: list) -> float:
    global label_to_idx
    l = list(label_to_idx.values())
    n_activities = len(l)
    tensor = t.zeros(len(trace), n_activities)
    for li, activity in enumerate(trace):
        tensor[li][label_to_idx[activity]] = 1

    tensor = tensor.reshape(1, tensor.shape[0], tensor.shape[1])

    return tensor


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


if __name__ == "__main__":
    pass
