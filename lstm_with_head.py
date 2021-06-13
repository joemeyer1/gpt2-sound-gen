
import torch
from torch import nn


class LSTMWithHead(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_size = kwargs.get("input_size")
        self.hidden_size = kwargs.get("hidden_size")
        self.num_layers = kwargs.get("num_layers")
        self.output_size = kwargs.get("output_size", 1)
        self.lstm_layer = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.activation = kwargs.get("activation", nn.ReLU())
        self.linear_layer = nn.Linear(self.hidden_size, self.output_size)
        self.std_for_decoding = torch.mean(kwargs.get("std_for_decoding", torch.tensor(1)))
        self.mean_for_decoding = torch.mean(kwargs.get("mean_for_decoding", torch.tensor(1)))

    def forward(self, x, hncn=None):
        xn, (hn, cn) = self.lstm_layer(x, hncn)
        out = self.linear_layer(self.activation(xn))
        return out, (hn, cn)

    def decode_output(self, lstm_output):
        return (lstm_output * self.std_for_decoding) + self.mean_for_decoding
