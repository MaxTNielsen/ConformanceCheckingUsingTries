import torch as torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader

import time
import math


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        X = self.inputs[index]
        y = self.targets[index]
        return X, y


def collate_fn(batch):
    return tuple(zip(*batch))


def load_data(test_split, inputs, targets):

    dataset_size = len(targets)

    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size

    dataset = Dataset(inputs, targets)

    train_eval_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    val_size = int(0.2 * len(train_eval_dataset))
    train_size = len(train_eval_dataset) - val_size

    train_data, val_data = random_split(
        train_eval_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    return train_data, val_data, test_dataset


def create_dataset(traces_dict, label_to_idx):
    l = list(label_to_idx.values())
    n_activities = len(l)

    inputs = []
    targets = []

    """many-to-many prediction task"""
    for trace in traces_dict.values():
        one_hot_tensor = torch.zeros(len(trace), n_activities)
        for li, activity in enumerate(trace):
            one_hot_tensor[li][label_to_idx[activity]] = 1
        # for i in range(int(len(trace)/2),len(trace)):
        input = one_hot_tensor[0:len(trace)-1, :]
        input = torch.cat((torch.zeros(1, n_activities), input))
        inputs.append(input)
        targets.append(torch.tensor([label_to_idx[trace[i]]
                       for i in range(len(trace))]))

    """many to one prediction task"""
    # for trace in traces_dict.values():
    #     one_hot_tensor = torch.zeros(len(trace), n_activities)
    #     for li, activity in enumerate(trace):
    #         one_hot_tensor[li][label_to_idx[activity]] = 1
    #     inputs.append(one_hot_tensor[0:len(trace)-1, :])
    #     targets.append(torch.tensor(label_to_idx[trace[-1]]))

    #inputs = pad_sequence(inputs, True)

    return inputs, targets


def preprocess_input(trace: list, label_to_idx: dict) -> float:
    l = list(label_to_idx.values())
    n_activities = len(l)
    tensor = torch.zeros(len(trace), n_activities)
    for li, activity in enumerate(trace):
        if activity in label_to_idx:
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
