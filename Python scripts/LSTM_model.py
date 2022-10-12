import torch.nn as nn
import torch as t

use_cuda = t.cuda.is_available()

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
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm = nn.BatchNorm1d(hidden_size*num_layers)

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_size*num_layers, hidden_size*num_layers)

        self.fc2 = nn.Linear(hidden_size*num_layers, num_classes)

        self.relu = nn.ReLU()

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = {}
        h_0 = t.zeros(self.num_layers, x.size(0), self.hidden_size)

        c_0 = t.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (get_variable(h_0), get_variable(c_0)))

        h_out = h_out.view(-1, self.hidden_size*self.num_layers)

        x = self.fc1(h_out)

        # x = self.batchnorm(x)

        # self.dropout(x)

        x = self.relu(x)

        x = self.fc2(x)

        x = self.softmax(x)

        out['out'] = x

        return out


if __name__ == "main":
    pass
