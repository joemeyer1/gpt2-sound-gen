#!usr/bin/env python3
# Copyright (c) Joseph Meyer (2021). All rights reserved.

from dataclasses import dataclass
from typing import Optional, Callable

import fire
import numpy as np
import torch
from tqdm import tqdm

from data_parsing_helpers.data_fetcher import get_training_data
from generate_gpt2_text import make_name_unique
from generate_lstm_text import gen_wav_with_lstm
from lstm_with_head import LSTMWithHead as LSTMWithHead


@dataclass
class ModelConfig:
    # either pass a path to load model from, or pass model parameters
    load_model_from_chkpt: Optional[str] = None  # path to load model from
    #    model parameters:
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    activation: Optional[Callable] = None

@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    save_model_every_n_epochs: int
    generate_wav_every_n_epochs: int
    save_model_to_chkpt: str
    overwrite_previous_model: bool

@dataclass
class DataConfig:
    in_wav_dir_name: str
    n_max_files: int
    write_wav_to_filename: str
    output_wav_len: int
    overwrite_wav: bool


def interface_to_train_lstm(
    # data config
    #   in
    in_wav_dir_name: str = "sound_data_percussion",
    n_max_files: int = 1,
    #   out
    write_wav_to_filename: str = "generated_drums.wav",
    output_wav_len: int = 100,
    overwrite_wav: bool = False,
    # model config
    load_model_from_chkpt: Optional[str] = None,
    hidden_size: Optional[int] = 32,  # only needed if load_model_from_chkpt=None
    num_layers: Optional[int] = 2,  # only needed if load_model_from_chkpt=None
    # training config
    epochs: int = 10,
    batch_size: int = 1,
    learning_rate=.01,
    save_model_to_chkpt='lstm.pt',
    overwrite_previous_model=False,
    save_model_every_n_epochs=5,
    generate_wav_every_n_epochs=20,
) -> None:

    data_config = DataConfig(
        in_wav_dir_name=in_wav_dir_name,
        n_max_files=n_max_files,
        write_wav_to_filename=write_wav_to_filename,
        output_wav_len=output_wav_len,
        overwrite_wav=overwrite_wav,
    )

    model_config = ModelConfig(
        load_model_from_chkpt=load_model_from_chkpt,
        hidden_size=hidden_size,
        num_layers=num_layers,
        activation=torch.nn.Tanh(),
    )

    training_config = TrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        save_model_every_n_epochs=save_model_every_n_epochs,
        generate_wav_every_n_epochs=generate_wav_every_n_epochs,
        save_model_to_chkpt=save_model_to_chkpt,
        overwrite_previous_model=overwrite_previous_model,
    )

    return train_lstm(data_config=data_config, model_config=model_config, training_config=training_config)


def train_lstm(
    data_config: DataConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
):

    # get data
    data = torch.tensor(np.expand_dims(get_training_data(read_wav_from_dir=data_config.in_wav_dir_name, n_max_files=data_config.n_max_files), -1).astype(np.float32))
    n_files = data.shape[0]
    assert training_config.batch_size > 0
    while n_files % training_config.batch_size > 0:
        training_config.batch_size -= 1
    n_batches = n_files // training_config.batch_size
    batched_data_shape = [n_batches, training_config.batch_size] + list(data.shape[1:])
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
    if model_config.load_model_from_chkpt:
        print(f"loading model from chkpt: {model_config.load_model_from_chkpt}\n")
        net = torch.load(model_config.load_model_from_chkpt)
        net.std_for_decoding = data_std
        net.mean_for_decoding = data_avg
    else:
        hidden_size = 32
        num_layers = 3
        activation = torch.nn.Tanh()
        print(f"creating new net\n\tnum_layers: '{num_layers}'\n\thidden_size: '{model_config.hidden_size}'\n\tactivation '{activation}'\n")
        net = LSTMWithHead(
            input_size=1,
            output_size=1,
            hidden_size=model_config.hidden_size,
            num_layers=model_config.num_layers,
            std_for_decoding=data_std,
            mean_for_decoding=data_avg,
            activation=model_config.activation,
        )
    optimizer = torch.optim.Adam(net.parameters(), lr=training_config.learning_rate)
    loss_fn = torch.nn.MSELoss()

    if not training_config.overwrite_previous_model:
        training_config.save_model_to_chkpt = make_name_unique(training_config.save_model_to_chkpt)

    try:
        for epoch_i in range(training_config.epochs):
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
            if epoch_i % training_config.save_model_every_n_epochs == 0:
                print(f"\tSaving model '{training_config.save_model_to_chkpt}'\n")
                torch.save(net, training_config.save_model_to_chkpt)
            # generate wav sample
            if epoch_i % training_config.generate_wav_every_n_epochs == 0:
                gen_wav_with_lstm(
                    write_wav_to_filename=data_config.write_wav_to_filename,
                    overwrite_wav=data_config.overwrite_wav,
                    output_wav_len=data_config.output_wav_len,
                    load_model_from_chkpt=training_config.save_model_to_chkpt,
                )
    except KeyboardInterrupt:
        print(f"Training interrupted")
    print(f"Saving model {training_config.save_model_to_chkpt}")
    torch.save(net, training_config.save_model_to_chkpt)

    gen_wav_with_lstm(
        write_wav_to_filename=data_config.write_wav_to_filename,
        overwrite_wav=data_config.overwrite_wav,
        output_wav_len=data_config.output_wav_len,
        load_model_from_chkpt=training_config.save_model_to_chkpt,
    )


if __name__ == "__main__":
    fire.Fire(interface_to_train_lstm)
