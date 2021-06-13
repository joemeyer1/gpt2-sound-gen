
import fire
import numpy as np
import torch
from math import floor
from lstm_with_head import LSTMWithHead as LSTMWithHead
from data_parsing_helpers.data_fetcher import get_training_data
# from data_parsing_helpers.vec_to_wav import int_to_hex
from generate_gpt2_text import make_name_unique
from generate_lstm_text import gen_wav_with_lstm

def train_lstm(
    epochs: int = 10,
    n_max_files: int = 1,
    in_wav_dir_name: str = "sound_data_percussion",
    write_wav_to_filename="generated_drums.wav",
    learning_rate=.01,
    output_wav_len=100,
    load_model_from_chkpt=None,
    save_model_to_chkpt='lstm.pt',
    save_model_every_n_epochs=5,
    overwrite_previous_model=False,
):

    # get data
    input_size = 1
    output_size = 1
    hidden_size = 64
    num_layers = 4
    # batch_size = 1
    data = torch.tensor(np.expand_dims(get_training_data(read_wav_from_dir=in_wav_dir_name, n_max_files=n_max_files), -1).astype(np.float32))

    # normalize data
    data_std = torch.std(input=data)
    if data_std == 0:
        data_std = torch.tensor(1, dtype=torch.float)
    data_avg = torch.mean(input=data)
    data = (data - data_avg) / data_std
    print(f"data_std: {data_std}")
    print(f"data_avg: {data_avg}")

    features = data[:, :-1]
    labels = data[:, 1:]
    batch_size, seq_length, input_size = features.shape

    # get net
    if load_model_from_chkpt:
        print(f"loading model from chkpt: {load_model_from_chkpt}")
        net = torch.load(load_model_from_chkpt)
        net.std_for_decoding = data_std
        net.mean_for_decoding = data_avg
    else:
        print("creating new net")
        net = LSTMWithHead(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            std_for_decoding=data_std,
            mean_for_decoding=data_avg,

        )
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    if not overwrite_previous_model:
        save_model_to_chkpt = make_name_unique(save_model_to_chkpt)

    try:
        for i in range(epochs):
            optimizer.zero_grad()
            y_pred, _ = net(features)
            loss = loss_fn(y_pred, labels)
            print(f"epoch {i} loss: {loss}\n")
            if i > 0 and i % save_model_every_n_epochs == 0:
                print(f"Saving model {save_model_to_chkpt}")
                torch.save(net, save_model_to_chkpt)
            loss.backward()
            optimizer.step()
    except KeyboardInterrupt:
        print(f"Training interrupted")
    print(f"Saving model {save_model_to_chkpt}")
    torch.save(net, save_model_to_chkpt)

    gen_wav_with_lstm(
        write_wav_to_filename=write_wav_to_filename,
        output_wav_len=output_wav_len,
        load_model_from_chkpt=save_model_to_chkpt,
    )


if __name__ == "__main__":
    fire.Fire(train_lstm)
