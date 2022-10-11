import torch as t
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence


class Dataset(t.utils.data.Dataset):
    def __init__(self, inputs, targets, transform=None, target_transform=None):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        X = self.inputs[index]
        y = self.targets[index]
        return X, y


def load_data(size, test_split, batch_size, inputs, targets):
    dataset_size = size

    train_size = int(test_split * dataset_size)
    test_size = dataset_size - train_size

    dataset = Dataset(inputs, targets)

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    train_size = int(test_split * len(train_dataset))
    eval_size = len(train_dataset) - train_size

    train_dataset, eval_dataset = random_split(
        train_dataset, [train_size, eval_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)

    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, eval_dataloader, test_dataloader


def make_data_set(traces_dict, labels_to_idx):
    data_set = []
    l = list(labels_to_idx.values())
    n_activities = len(l)

    data_set = []
    for trace in traces_dict.values():
        tensor = t.zeros(len(trace), n_activities)
        for li, activity in enumerate(trace):
            tensor[li][labels_to_idx[activity]] = 1
        data_set.append(tensor)

    inputs = []

    for trace in data_set:
        tensor = t.zeros(1, n_activities)
        temp = trace[0:trace.shape[1]-1, :]
        input = t.cat((tensor, temp))
        inputs.append(input)

    inputs = pad_sequence(inputs)
    targets = pad_sequence(data_set)

    reshaped_inputs = []
    reshaped_targets = []

    for i in range(len(inputs)):
        reshaped_inputs.append(inputs[i].reshape(
            1, inputs[i].shape[0], inputs[i].shape[1]))
        reshaped_targets.append(targets[i].reshape(
            1, targets[i].shape[0], targets[i].shape[1]))

    return reshaped_inputs, reshaped_targets
