#!usr/bin/env python3
# Copyright (c) Joseph Meyer. All rights reserved.


import random

from data_parsing_helpers.wav_to_vector import extract_data
from data_parsing_helpers.data_fetcher import get_training_data
from data_parsing_helpers.file_helpers import bin_data, unbin_data, get_n_bytes_int



def test_():
    hex_ls = ['ff', 'f0', '00', 'ff']
    n_bytes = get_n_bytes_int(4, hex_ls, 0)
    print(n_bytes)


def test_mus_law():
    data = [16**(i + 1) - 1 for i in range(16)]
    binned_data = bin_data(data)
    print(binned_data)
    reconstructed_data = unbin_data(binned_data)
    print(reconstructed_data)



def test_extract_data():
    data = extract_data("/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_toy/violin_G4_phrase_forte_harmonic-glissando.wav")
    print(data)

def test_bin_data():
    data = extract_data("/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_toy/violin_G4_phrase_forte_harmonic-glissando.wav")
    print(data)
    bin_data(data[0][0])

def test_main():
    read_wav_from_filename = "/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_toy/violin_G4_phrase_forte_harmonic-glissando.wav"
    write_wav_to_filename = f"{read_wav_from_filename.split('.')[0]}_data_channels"
    _, header_info = extract_data(read_wav_from_filename=read_wav_from_filename, write_wav_to_filename=write_wav_to_filename)
    vec_to_wav(write_wav_to_filename, header_info)

def test_wav_encoding():
    data = get_training_data("/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_toy", n_max_files=2)
    print(data)

def test_decoding():
    pass