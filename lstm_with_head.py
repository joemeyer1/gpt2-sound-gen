
from torch import nn


class LSTMWithHead(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_size = kwargs.get("input_size")
        self.hidden_size = kwargs.get("hidden_size")
        self.num_layers = kwargs.get("num_layers")
        self.output_size = kwargs.get("output_size", 1)
        self.lstm_layer = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.linear_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        xn, (hn, cn) = self.lstm_layer(x)
        out = self.linear_layer(xn)
        return out

