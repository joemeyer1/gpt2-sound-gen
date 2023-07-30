#!usr/bin/env python3
# Copyright (c) Joe Meyer (2020). All rights reserved.

import numpy as np

from data_parsing_helpers.file_helpers import unbin_data


def vec_to_wav(
        vec_filename,  # e.g. "violin_Gs3_1_piano_arco-sul-tasto_data_channels"
        header_info,
):
    vec = np.loadtxt(f"{vec_filename}.txt", dtype=int)
    unbinned_vec = unbin_data(vec.tolist())
    header_info['chunk_size'] = (len(unbinned_vec) * 4) + 36
    header_info['subchunk2size'] = len(unbinned_vec) * 4
    hex_str = write_header(header_info)  # e.g.chunk_size=259312, num_channels=1, sample_rate=44100, bits_per_sample=16, subchunk2size=258048
    for n in unbinned_vec:
        hex_str += int_to_hex(int_to_convert=n, bytes=header_info['bits_per_sample'] // 8, signed=True)

    # hex_str += bytearray(vec)
    with open(f"{vec_filename}.wav", 'wb') as f:
        f.write(hex_str)
    return hex_str


def write_header(header_info: dict):
    num_channels = header_info['num_channels']
    sample_rate = header_info['sample_rate']
    bits_per_sample = header_info['bits_per_sample']
    chunk_size = header_info['chunk_size']
    subchunk2size = header_info['subchunk2size']
    subchunk1size = 16  # PCM (Pulse Code Modulation)
    audio_format = 1  # PCM
    byte_rate = sample_rate * num_channels * bits_per_sample / 8
    block_align = num_channels * bits_per_sample / 8
    hex_str = b''
    hex_str += str_to_hex('RIFF')
    hex_str += int_to_hex(chunk_size, 4)
    hex_str += str_to_hex('WAVEfmt ')
    hex_str += int_to_hex(subchunk1size, 4)
    hex_str += int_to_hex(audio_format, 2)
    hex_str += int_to_hex(num_channels, 2)
    hex_str += int_to_hex(sample_rate, 4)
    hex_str += int_to_hex(byte_rate, 4)
    hex_str += int_to_hex(block_align, 2)
    hex_str += int_to_hex(bits_per_sample, 2)
    hex_str += str_to_hex("data")
    hex_str += int_to_hex(subchunk2size, 4)
    return hex_str


def str_to_hex(str):
    hex_str = b''
    for s in str:
        hex_str += int_to_hex(ord(s), bytes=1)
    return hex_str


def int_to_hex(int_to_convert, bytes, signed: bool = False, byteorder='little'):
    ret = (int(int_to_convert)).to_bytes(bytes, byteorder=byteorder, signed=signed)
    return ret
