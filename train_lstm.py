
import fire
import numpy as np
import torch
from tqdm import tqdm

from data_parsing_helpers.data_fetcher import get_training_data
# from data_parsing_helpers.vec_to_wav import int_to_hex
from generate_gpt2_text import make_name_unique
from generate_lstm_text import gen_wav_with_lstm
from lstm_with_head import LSTMWithHead as LSTMWithHead


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

    # get net
    if load_model_from_chkpt:
        print(f"loading model from chkpt: {load_model_from_chkpt}\n")
        net = torch.load(load_model_from_chkpt)
        net.std_for_decoding = data_std
        net.mean_for_decoding = data_avg
    else:
        hidden_size = 128
        num_layers = 4
        activation = torch.nn.Tanh()
        print(f"creating new net\n\tnum_layers: '{num_layers}'\n\thidden_size: '{hidden_size}'\n\tactivation '{activation}'\n")
        net = LSTMWithHead(
            input_size=1,
            output_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            std_for_decoding=data_std,
            mean_for_decoding=data_avg,
            activation=activation,
        )
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    if not overwrite_previous_model:
        save_model_to_chkpt = make_name_unique(save_model_to_chkpt)

    try:
        for epoch_i in range(epochs):
            tot_batch_loss = 0
            epoch_loss = 0
            with tqdm(range(n_batches), leave=False) as batch_counter:
                for batch_i in batch_counter:
                    features_batch = features[batch_i]
                    labels_batch = labels[batch_i]
                    optimizer.zero_grad()
                    y_pred, _ = net(features_batch)
                    loss = loss_fn(y_pred, labels_batch)
                    epoch_loss += loss
                    loss.backward()
                    optimizer.step()
                    tot_batch_loss += loss.item()
                    running_loss = tot_batch_loss / float(batch_i + 1)
                    batch_counter.desc = "Epoch {} Loss: {}".format(epoch_i, running_loss)
                batch_counter.close()
            # report loss
            epoch_loss = tot_batch_loss / float(n_batches)
            print("Epoch {} Avg Loss: {}\n".format(epoch_i, epoch_loss))
            # save model
            if epoch_i % save_model_every_n_epochs == 0:
                print(f"\tSaving model '{save_model_to_chkpt}'\n")
                torch.save(net, save_model_to_chkpt)
            # generate wav sample
            if epoch_i % generate_wav_every_n_epochs == 0:
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
