
import fire
import numpy as np
import torch
from lstm_with_head import LSTMWithHead as LSTMWithHead
from data_parsing_helpers.data_fetcher import get_training_data

def train_lstm(
    epochs: int = 100,
    n_max_files: int = 1,
    in_wav_dir_name: str = "sound_data",
    wav_str_filename: str = "sound_short.txt",
    use_previous_training_data: bool = True,
    learning_rate=1000,
    load_model_from_chkpt=None,
    save_model_every_n_epochs=1000,
    overwrite_previous_model=False,
):

    # get data
    input_size = 1
    output_size = 1
    hidden_size = 20
    num_layers = 2
    # batch_size = 1
    # seq_length = 5
    # features = torch.randn(seq_length, batch_size, input_size)
    # labels = torch.cat((features[1:], torch.tensor([[[5]]])), 0)
    data = torch.tensor(np.expand_dims(get_training_data(read_wav_from_dir="sound_data", n_max_files=5), -1).astype(np.float32))
    # data = torch.tensor([[[1], [2], [3], [4]]]).float()
    features = data[:, :-1]
    labels = data[:, 1:]
    batch_size, seq_length, input_size = features.shape

    # get net
    net = LSTMWithHead(input_size=input_size, output_size=output_size, hidden_size=hidden_size, num_layers=num_layers, )
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
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
