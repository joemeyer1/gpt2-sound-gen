#!usr/bin/env python3
# Copyright (c) Joseph Meyer (2021). All rights reserved.

import fire
import numpy as np
import torch
from math import floor
from data_parsing_helpers.data_fetcher import get_training_data
# from data_parsing_helpers.vec_to_wav import int_to_hex
from generate_gpt2_text import write_wav, make_name_unique

def gen_wav_with_lstm(
    write_wav_to_filename="generated_drums.wav",
    overwrite_wav=False,
    output_wav_len=100,
    load_model_from_chkpt='lstm.pt',
    verbose=True,  # write ints, useful for debugging
):

    if not overwrite_wav:
        write_wav_to_filename = make_name_unique(write_wav_to_filename)
    print(f"\tGenerating wav '{write_wav_to_filename}'")

    net = torch.load(load_model_from_chkpt)

    # generate wav
    generated_wav_body = torch.tensor([[[0]]], dtype=torch.float)
    hncn = None
    for i in range(output_wav_len - 1):
        next_input = generated_wav_body[-1:]
        y_pred, hncn = net(next_input, hncn)
        generated_wav_body = torch.cat((generated_wav_body, y_pred[-1:]))
    generated_wav_body = net.decode_output(generated_wav_body)
    if verbose:
        print(f"generated_wav_body: {generated_wav_body}")
        write_ints_to_filename = write_wav_to_filename.split('.')[0] + '_ints.txt'
        with open(write_ints_to_filename, 'w') as f:
            for i in generated_wav_body.flatten().tolist():
                f.write(i + ' ')

    def int_to_hex_str(int_to_convert, n_bits):
        ret = hex(int_to_convert)[2:]
        ret = '0' * (n_bits - len(ret)) + ret
        return ret

    hex_prompt = [int_to_hex_str(int_to_convert=floor(i), n_bits=8) for i in generated_wav_body.flatten()]
    wav_body = ""
    for h in hex_prompt:
        wav_body += h

    print(f"\twriting wav file '{write_wav_to_filename}'\n")
    write_wav(wav_txt=wav_body, write_wav_to_filename=write_wav_to_filename)


if __name__ == "__main__":
    fire.Fire(gen_wav_with_lstm)
