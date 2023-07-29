#!usr/bin/env python3
# Copyright (c) Joseph Meyer. All rights reserved.


import unittest

import numpy as np

from data_parsing_helpers.data_fetcher import get_training_data
from data_parsing_helpers.file_helpers import bin_data, unbin_data, get_n_bytes_int
from data_parsing_helpers.make_wav_str_file import convert_wav_to_text_file
from data_parsing_helpers.vec_to_wav import vec_to_wav, int_to_hex, write_header
from data_parsing_helpers.wav_to_vector import extract_data, extract_binned_data
from generate_gpt2_text import write_wav, decode_generated_text


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

    def test_vec_to_wav(
        self,
        read_wav_from_filename: str = "/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_toy/violin_G4_phrase_forte_harmonic-glissando.wav",
        write_to_filename: str = "/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_output_toy/violin_G4_phrase_forte_harmonic-glissando_test_vec_to_wav",
    ) -> None:
        """Converts wav file to binned data, then converts it back to wav and writes it to output file."""

        _, header_info = extract_binned_data(read_wav_from_filename=read_wav_from_filename, write_wav_to_filename=f"{write_to_filename}.txt")
        vec_to_wav(write_to_filename, header_info)
        print()
        # to test/verify, listen to the wav file it writes, and see if it sounds like the input one

    def test_decode_generated_text(
        self,
        read_wav_from_filename: str = "/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_toy/violin_G4_phrase_forte_harmonic-glissando.wav",
        write_to_filename: str = "/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_output_toy/violin_G4_phrase_forte_harmonic-glissando_data_channels",
    ) -> None:
        """Converts wav file to binned data, then converts it back to wav and writes it to output file."""

        # quantized_data_channels, header_info = extract_binned_data(
        #     read_wav_from_filename=read_wav_from_filename,
        #     write_wav_to_filename=f"{write_to_filename}.txt",
        # )

        raw_data_channels, header_info = extract_data(read_wav_from_filename)
        print(raw_data_channels)
        quantized_data_channels = {ix: bin_data(data_channel) for ix, data_channel in raw_data_channels.items()}
        restored_wav = decode_generated_text(
            generated_text='-'.join(map(str, quantized_data_channels[0])),
            bytes_per_sample=header_info['bits_per_sample'] // 8,
        )

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

    def test_write_wav(
        self,
        read_wav_from_filename: str = "/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_toy/violin_G4_phrase_forte_harmonic-glissando.wav",
        write_to_filename: str = "/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_output_toy/violin_G4_phrase_forte_harmonic-glissando_test_write.wav",
    ) -> None:
        """Converts wav file to binned data, then converts it back to wav and writes it to output file."""

        raw_data_channels, header_info = extract_data(read_wav_from_filename)
        print(raw_data_channels)
        quantized_data_channels = {ix: bin_data(data_channel) for ix, data_channel in raw_data_channels.items()}
        restored_wav = decode_generated_text(
            generated_text='-'.join(map(str, quantized_data_channels[0])),
            bytes_per_sample=header_info['bits_per_sample'] // 8,
        )
        write_wav(raw_pressures=restored_wav, write_wav_to_filename=write_to_filename, header_info=header_info, n_pressure_samples=len(quantized_data_channels[0]))

    def test_wav_encoding(self):
        data = get_training_data("/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_toy", n_max_files=2)
        print(data)

    def test_wav_to_text(
        self,
        n_max_files: int = 1,
        in_wav_dir_name: str = "/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_toy",
        wav_str_filename: str = "/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_output_toy/sound.txt",
    ):
        convert_wav_to_text_file(
            in_wav_dir_name=in_wav_dir_name,
            out_text_filename=wav_str_filename,
            n_max_files=n_max_files,
        )

if __name__ == "__main__":
    unittest.main()
