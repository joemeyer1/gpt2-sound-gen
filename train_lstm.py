
import fire
import numpy as np
import torch
from math import floor
from lstm_with_head import LSTMWithHead as LSTMWithHead
from data_parsing_helpers.data_fetcher import get_training_data
# from data_parsing_helpers.vec_to_wav import int_to_hex
from generate_gpt2_text import write_wav

def train_lstm(
    epochs: int = 10,
    n_max_files: int = 1,
    in_wav_dir_name: str = "sound_data",
    wav_str_filename: str = "sound_short.txt",
    use_previous_training_data: bool = True,
    learning_rate=.01,
    output_wav_len=100000,
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

    # normalize data
    data_std = torch.std(input=data, axis=0)
    data_std[data_std == 0] = 1
    data_avg = torch.mean(input=data, axis=0)
    data = (data - data_avg) / data_std

    features = data[:, :-1]
    labels = data[:, 1:]
    batch_size, seq_length, input_size = features.shape

    # get net
    net = LSTMWithHead(input_size=input_size, output_size=output_size, hidden_size=hidden_size, num_layers=num_layers, )
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    try:
        for i in range(epochs):
            optimizer.zero_grad()
            y_pred, _ = net(features)
            loss = loss_fn(y_pred, labels)
            print(f"epoch {i} loss: {loss}\n")
            loss.backward()
            optimizer.step()
    except Exception as e:
        print(e)

    prompt = torch.tensor([[[0]]], dtype=torch.float)
    hncn = None
    for i in range(output_wav_len - 1):
        y_pred, hncn = net(prompt, hncn)
        prompt = torch.cat((prompt, y_pred[-1:]), axis=0)
    print(prompt)
    prompt = (prompt * torch.mean(data_std)) + torch.mean(data_avg)
    print(prompt)

    def int_to_hex_str(int_to_convert, n_bits):
        ret = hex(int_to_convert)[2:]
        ret = '0' * (n_bits - len(ret)) + ret
        return ret

    hex_prompt = [int_to_hex_str(int_to_convert=floor(i), n_bits=8) for i in prompt.flatten()]
    wav_body = ""
    for h in hex_prompt:
        wav_body += h

    write_wav_to_filename = "trash.wav"
    print(f"writing wav file '{write_wav_to_filename}'")
    write_wav(wav_txt=wav_body, write_wav_to_filename=write_wav_to_filename)


if __name__ == "__main__":
    fire.Fire(train_lstm)
