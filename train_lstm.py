
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
    batch_size: int = 1,
    in_wav_dir_name: str = "sound_data_percussion",
    write_wav_to_filename="generated_drums.wav",
    learning_rate=.01,
    output_wav_len=100,
    load_model_from_chkpt=None,
    save_model_to_chkpt='lstm.pt',
    save_model_every_n_epochs=5,
    generate_wav_every_n_epochs=20,
    overwrite_previous_model=False,
    overwrite_wav=False,
):

    # get data
    output_size = 1
    hidden_size = 32
    num_layers = 2
    data = torch.tensor(np.expand_dims(get_training_data(read_wav_from_dir=in_wav_dir_name, n_max_files=n_max_files), -1).astype(np.float32))
    n_files = data.shape[0]
    assert batch_size > 0
    while n_files % batch_size > 0:
        batch_size -= 1
    n_batches = n_files // batch_size
    batched_data_shape = [n_batches, batch_size] + list(data.shape[1:])
    data = data.reshape(batched_data_shape)

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
    input_size = features.shape[-1]

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
            activation=torch.nn.ReLU(),
        )
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    if not overwrite_previous_model:
        save_model_to_chkpt = make_name_unique(save_model_to_chkpt)

    try:
        for i in range(epochs):
            epoch_loss = 0
            for features_batch, labels_batch in zip(features, labels):
                optimizer.zero_grad()
                y_pred, _ = net(features_batch)
                loss = loss_fn(y_pred, labels_batch)
                epoch_loss += loss
                loss.backward()
                optimizer.step()
            print(f"epoch {i} loss: {epoch_loss / n_batches}\n")
            if i % save_model_every_n_epochs == 0:
                print(f"Saving model {save_model_to_chkpt}")
                torch.save(net, save_model_to_chkpt)
            if i % generate_wav_every_n_epochs == 0:
                gen_wav_with_lstm(
                    write_wav_to_filename=write_wav_to_filename,
                    overwrite_wav=overwrite_wav,
                    output_wav_len=output_wav_len,
                    load_model_from_chkpt=save_model_to_chkpt,
                )
    except KeyboardInterrupt:
        print(f"Training interrupted")
    print(f"Saving model {save_model_to_chkpt}")
    torch.save(net, save_model_to_chkpt)

    gen_wav_with_lstm(
        write_wav_to_filename=write_wav_to_filename,
        overwrite_wav=overwrite_wav,
        output_wav_len=output_wav_len,
        load_model_from_chkpt=save_model_to_chkpt,
    )


if __name__ == "__main__":
    fire.Fire(train_lstm)
