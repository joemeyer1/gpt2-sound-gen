
import fire
import numpy as np
import torch
from math import floor
from data_parsing_helpers.data_fetcher import get_training_data
# from data_parsing_helpers.vec_to_wav import int_to_hex
from generate_gpt2_text import write_wav

def train_lstm(
    epochs: int = 10,
    n_max_files: int = 1,
    in_wav_dir_name: str = "sound_data_percussion",
    write_wav_to_filename="generated_drums.wav",
    wav_str_filename: str = "sound_short.txt",
    use_previous_training_data: bool = True,
    learning_rate=.01,
    output_wav_len=100,
    load_model_from_chkpt=None,
    save_model_every_n_epochs=5,
    overwrite_previous_model=False,
):



    # get data
    data = torch.tensor(np.expand_dims(get_training_data(read_wav_from_dir=in_wav_dir_name, n_max_files=n_max_files), -1).astype(np.float32))

    # normalize data
    data_std = torch.std(input=data, axis=0)
    data_std[data_std == 0] = 1
    data_avg = torch.mean(input=data, axis=0)
    data = (data - data_avg) / data_std

    features = data[:, :-1]
    labels = data[:, 1:]
    batch_size, seq_length, input_size = features.shape

    # get net
    net = torch.load('lstm.pt')
    net

    # generate wav
    generated_wav_body = torch.tensor([[[0]]], dtype=torch.float)
    hncn = None
    for i in range(output_wav_len - 1):
        next_input = generated_wav_body[-1:]
        y_pred, hncn = net(next_input, hncn)
        generated_wav_body = torch.cat((generated_wav_body, y_pred[-1:]), axis=0)
    print(generated_wav_body)
    generated_wav_body = net.decode_output(generated_wav_body)
    print(generated_wav_body)

    def int_to_hex_str(int_to_convert, n_bits):
        ret = hex(int_to_convert)[2:]
        ret = '0' * (n_bits - len(ret)) + ret
        return ret

    hex_prompt = [int_to_hex_str(int_to_convert=floor(i), n_bits=8) for i in generated_wav_body.flatten()]
    wav_body = ""
    for h in hex_prompt:
        wav_body += h

    print(f"writing wav file '{write_wav_to_filename}'")
    write_wav(wav_txt=wav_body, write_wav_to_filename=write_wav_to_filename)


if __name__ == "__main__":
    fire.Fire(train_lstm)
