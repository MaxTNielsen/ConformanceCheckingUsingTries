import torch.nn as nn
import torch as torch

use_cuda = torch.cuda.is_available()

print("Running GPU.") if use_cuda else print("No GPU available.")


def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if use_cuda:
        return x.cuda()
    return x


def get_numpy(x):
    if use_cuda:
        x = x.cpu()
    return x.data.numpy()


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.isBi = False
        self.biMultiplier = 2 if self.isBi else 1
        self.dropout = nn.Dropout(p=0.40802631010146606)
        self.batchnorm = nn.BatchNorm1d(self.hidden_size*self.biMultiplier)

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, bidirectional=self.isBi)

        self.fc1 = nn.Linear(self.hidden_size*self.biMultiplier,
                             self.hidden_size*self.biMultiplier)

        self.fc2 = nn.Linear(self.hidden_size*self.biMultiplier, num_classes)

        self.relu = nn.ReLU()

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = {}

        outputs = []

        for input in x:

            # h_0 = torch.zeros(self.self.num_layers*self.biMultiplier, input.size(0), self.hidden_size)

            # c_0 = torch.zeros(self.self.num_layers*self.biMultiplier, input.size(0), self.hidden_size)

            h_0 = torch.zeros(self.num_layers*self.biMultiplier, self.hidden_size)

            c_0 = torch.zeros(self.num_layers*self.biMultiplier, self.hidden_size)

            self.dropout(input)

            # Propagate input through LSTM
            ula, (h_out, _) = self.lstm(
                input, (get_variable(h_0), get_variable(c_0)))

            #h_out = h_out.view(-1, self.hidden_size*self.self.num_layers*self.biMultiplier)

            input = self.fc1(ula)

            self.dropout(input)

            input = self.relu(input)

            input = self.fc2(input)

            input = self.softmax(input)

            outputs.append(input)

        out['out'] = outputs

        return out


if __name__ == "__main__":
    pass
