#!usr/bin/env python3
# Copyright (c) Joseph Meyer. All rights reserved.


import random

import numpy as np

from data_parsing_helpers.wav_to_vector import extract_data, extract_binned_data
from data_parsing_helpers.data_fetcher import get_training_data
from data_parsing_helpers.file_helpers import bin_data, unbin_data, get_n_bytes_int

from data_parsing_helpers.vec_to_wav import vec_to_wav, int_to_hex




def test_bytes_fetch():
    hex_ls = ['ff', '7f', '00', 'ff']
    n_bytes, i = get_n_bytes_int(n=2, hex_ls=hex_ls, i=0)
    assert n_bytes == 32767
    n_bytes, i = get_n_bytes_int(n=2, hex_ls=hex_ls, i=i)
    assert n_bytes == -256
    print(i)


def test_int_to_hex():
    int_to_convert0 = 32767
    hex0 = int_to_hex(int_to_convert=int_to_convert0, bytes=2, signed=True)
    assert int.from_bytes(hex0, byteorder='little', signed=True) == int_to_convert0
    int_to_convert1 = -256
    hex1 = int_to_hex(int_to_convert=int_to_convert1, bytes=2, signed=True)
    assert int.from_bytes(hex1, byteorder='little', signed=True) == int_to_convert1


def test_bin_data():
    data = [-(16**i)/2 for i in range(0, 5)] + [(16**i)/2 for i in range(0, 5)]
    binned_data = bin_data(data)
    print(binned_data)
    assert min(binned_data) >= 0
    assert max(binned_data) < 256
    reconstructed_data = unbin_data(binned_data)
    print(reconstructed_data)
    assert np.sum(np.array(data) - np.array(reconstructed_data)) < 16**4


def test_extract_data(read_wav_from_filename: str = "/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_toy/violin_G4_phrase_forte_harmonic-glissando.wav"):
    raw_data_channels, head_info = extract_data(read_wav_from_filename)
    print(raw_data_channels)
    quantized_data_channels = {ix: bin_data(data_channel) for ix, data_channel in raw_data_channels.items()}
    print(quantized_data_channels)
    reconstructed_data = unbin_data(quantized_data_channels[0])
    print(reconstructed_data)


def test_write_wav(read_wav_from_filename: str = "/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_toy/violin_G4_phrase_forte_harmonic-glissando.wav"):
    write_wav_to_filename = f"{read_wav_from_filename.split('.')[0]}_data_channels"
    _, header_info = extract_binned_data(read_wav_from_filename=read_wav_from_filename, write_wav_to_filename=write_wav_to_filename)
    vec_to_wav(write_wav_to_filename, header_info)

# def test_bin_data():
#     data = extract_data("/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_toy/violin_G4_phrase_forte_harmonic-glissando.wav")
#     print(data)
#     bin_data(data[0][0])
#
# def test_main():
#     read_wav_from_filename = "/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_toy/violin_G4_phrase_forte_harmonic-glissando.wav"
#     write_wav_to_filename = f"{read_wav_from_filename.split('.')[0]}_data_channels"
#     _, header_info = extract_data(read_wav_from_filename=read_wav_from_filename, write_wav_to_filename=write_wav_to_filename)
#     vec_to_wav(write_wav_to_filename, header_info)

def test_wav_encoding():
    data = get_training_data("/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_toy", n_max_files=2)
    print(data)

# def test_decoding():
#     pass