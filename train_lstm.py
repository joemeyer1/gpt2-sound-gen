
import fire
import torch
from lstm_with_head import LSTMWithHead as LSTMWithHead

def train_lstm():

    # get data
    input_size = 1
    output_size = 1
    hidden_size = 20
    num_layers = 2
    batch_size = 1
    seq_length = 5
    features = torch.randn(seq_length, batch_size, input_size)
    labels = torch.cat((features[1:], torch.tensor([[[5]]])), 0)

    # get net
    net = LSTMWithHead(input_size=input_size, output_size=output_size, hidden_size=hidden_size, num_layers=num_layers, )
    epochs = 1000
    lr = .01
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for i in range(epochs):
        optimizer.zero_grad()
        y_pred = net(features)
        loss = loss_fn(y_pred, labels)
        print(f"loss: {loss}\n")
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    fire.Fire(train_lstm)
