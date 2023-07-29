#!usr/bin/env python3
# Copyright (c) Joseph Meyer. All rights reserved.


import unittest

import numpy as np

from data_parsing_helpers.file_helpers import bin_data, unbin_data, get_n_bytes_int
from data_parsing_helpers.vec_to_wav import int_to_hex, write_header
from data_parsing_helpers.wav_to_vector import extract_data


class DataParsingTests(unittest.TestCase):
    def test_bytes_fetch(self):
        hex_ls = ['ff', '7f', '00', 'ff']
        n_bytes, i = get_n_bytes_int(n=2, hex_ls=hex_ls, i=0)
        assert n_bytes == 32767
        n_bytes, i = get_n_bytes_int(n=2, hex_ls=hex_ls, i=i)
        assert n_bytes == -256
        print(i)

    def test_int_to_hex(self):
        int_to_convert0 = 32767
        hex0 = int_to_hex(int_to_convert=int_to_convert0, bytes=2, signed=True)
        assert int.from_bytes(hex0, byteorder='little', signed=True) == int_to_convert0
        int_to_convert1 = -256
        hex1 = int_to_hex(int_to_convert=int_to_convert1, bytes=2, signed=True)
        assert int.from_bytes(hex1, byteorder='little', signed=True) == int_to_convert1

    def test_bin_data(self):
        data = [-(16**i)/2 for i in range(0, 5)] + [(16**i)/2 for i in range(0, 5)]
        binned_data = bin_data(data)
        print(binned_data)
        assert min(binned_data) >= 0
        assert max(binned_data) < 256
        reconstructed_data = unbin_data(binned_data)
        print(reconstructed_data)
        assert np.sum(np.array(data) - np.array(reconstructed_data)) < 16**4

    def test_extract_data(self, read_wav_from_filename: str = "/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_toy/violin_G4_phrase_forte_harmonic-glissando.wav"):
        raw_data_channels, head_info = extract_data(read_wav_from_filename)
        print(raw_data_channels)
        quantized_data_channels = {ix: bin_data(data_channel) for ix, data_channel in raw_data_channels.items()}
        print(quantized_data_channels)
        reconstructed_data = unbin_data(quantized_data_channels[0])
        print(reconstructed_data)


    def test_bytes_to_pretty_str(
        self,
        read_wav_from_filename: str = "/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_toy/violin_G4_phrase_forte_harmonic-glissando.wav",
        write_to_filename: str = "/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_output_toy/violin_G4_phrase_forte_harmonic-glissando_test_bytes_to_pretty_str.wav",
    ):
        raw_data_channels, header_info = extract_data(read_wav_from_filename)
        data_bytes = b''.join(map(
            lambda int_to_convert: int_to_hex(int_to_convert=int_to_convert, bytes=2, signed=True),
            raw_data_channels[0],
        ))
        header_info['chunk_size'] = len(data_bytes) + 36
        header_info['subchunk2size'] = len(data_bytes) * header_info['num_channels'] * (header_info['bits_per_sample'] // 8)
        header_bytes = write_header(header_info)
        whole_wav = header_bytes + data_bytes
        with open(write_to_filename, 'wb') as f:
            f.write(whole_wav)


if __name__ == "__main__":
    unittest.main()
