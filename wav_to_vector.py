#!usr/bin/env python3
# Copyright (c) Joe Meyer (2020). All rights reserved.


from typing import Tuple, Dict, Optional

from helpers import file_to_hex_ls, get_n_bytes, get_n_bytes_int, get_n_bytes_str


def extract_header(hex_ls):
    i = 0
    # check chunk format
    expected_chunk_format = 'RIFF'
    chunk_format, i = get_n_bytes_str(4, hex_ls, i)
    assert chunk_format == expected_chunk_format, chunk_format
    # get chunk size
    chunk_size, i = get_n_bytes_int(4, hex_ls, i)
    print(f"chunk_size: {chunk_size}")
    # check fmt stuff
    expected_next_chunk = 'WAVEfmt'
    next_chunk, i = get_n_bytes_str(7, hex_ls, i)
    assert next_chunk == expected_next_chunk, next_chunk
    # skip a byte
    _, i = get_n_bytes(1, hex_ls, i)
    # get subchunk1size
    subchunk1size, i = get_n_bytes_int(4, hex_ls, i)
    print(f"subchunk1size: {subchunk1size}")
    # make sure it's PCM
    audio_format, i = get_n_bytes_int(2, hex_ls, i)
    pulse_code_modulation_format = 1
    assert audio_format == pulse_code_modulation_format, audio_format
    # get num chans
    num_channels, i = get_n_bytes_int(2, hex_ls, i)
    print(f"num_channels: {num_channels}")
    # get sample rate
    sample_rate, i = get_n_bytes_int(4, hex_ls, i)
    print(f"sample_rate: {sample_rate}")
    # get byte_rate
    byte_rate, i = get_n_bytes_int(4, hex_ls, i)
    print(f"byte_rate: {byte_rate}")
    # get block_align
    block_align, i = get_n_bytes_int(2, hex_ls, i)
    print(f"block_align: {block_align}")
    # get bits_per_sample
    bits_per_sample, i = get_n_bytes_int(2, hex_ls, i)
    print(f"bits_per_sample: {bits_per_sample}")
    print("header extracted.")
    # skip any padding
    next_bytes = ''
    while next_bytes != 'data':
        next_bytes, i = get_n_bytes_str(4, hex_ls, i)
    # get subchunk2size
    subchunk2size, i = get_n_bytes_int(4, hex_ls, i)
    print(f"subchunk2size: {subchunk2size}")
    header_info = {
        "num_channels": num_channels,
        "sample_rate": sample_rate,
        "bits_per_sample": bits_per_sample,
        "subchunk2size": subchunk2size,
    }
    return header_info, hex_ls, i

def extract_body(hex_ls, i, header_info) -> Dict[int, list]:
    num_channels = header_info['num_channels']
    binary_bits_per_sample = header_info['bits_per_sample']
    assert binary_bits_per_sample % 4 == 0
    hex_bits_per_sample = binary_bits_per_sample // 4  # 16 = 2**4
    assert hex_bits_per_sample % num_channels == 0
    hex_bits_per_sample_per_channel = hex_bits_per_sample // num_channels
    from collections import defaultdict
    data_channels = defaultdict(list)  # maps {channel(int): data(list)}
    while i + hex_bits_per_sample <= header_info['subchunk2size']:
        for channel in range(num_channels):
            channel_sample, i = get_n_bytes_int(hex_bits_per_sample_per_channel, hex_ls, i)
            data_channels[channel].append(channel_sample)
    return data_channels

def extract_data(
        read_wav_from_filename: str = "violin_Gs3_1_piano_arco-sul-tasto.wav",
        write_wav_to_filename: Optional[str] = None,
) -> Tuple[Dict[int, list], Dict[str, int]]:
    data_hex_ls = file_to_hex_ls(read_wav_from_filename)
    header_info, hex_ls, i = extract_header(data_hex_ls)
    data_channels = extract_body(hex_ls, i, header_info)
    if write_wav_to_filename:
        import numpy as np
        np.savetxt(fname=write_wav_to_filename, X=np.array(list(data_channels.values())).transpose(), fmt='%d', delimiter=' ')
    return data_channels, header_info
