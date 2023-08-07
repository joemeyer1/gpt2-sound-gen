#!usr/bin/env python3
# Copyright (c) Joseph Meyer (2020-2023). All rights reserved.


import unittest

from data_parsing_helpers.wav_to_vec import get_training_data
from data_parsing_helpers.file_helpers import bin_data
from data_parsing_helpers.wav_to_vec import format_data_for_training
from data_parsing_helpers.vec_to_wav import vec_to_wav
from data_parsing_helpers.wav_to_vec import extract_data, extract_binned_data
from generate_output import write_wav, decode_generated_text, get_clean_next_generated_text


class AudioEncodingTests(unittest.TestCase):
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

        raw_data_channels, header_info = extract_data(read_wav_from_filename)
        print(raw_data_channels)
        quantized_data_channels = {ix: bin_data(data_channel) for ix, data_channel in raw_data_channels.items()}
        restored_wav = decode_generated_text(
            generated_text='-'.join(map(str, quantized_data_channels[0])),
            bytes_per_sample=header_info['bits_per_sample'] // 8,
        )

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

    def test_convert_wav_to_text_file(
        self,
        n_max_files: int = 1,
        in_wav_dir_name: str = "/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_toy",
        wav_str_filename: str = "/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_output_toy/sound.txt",
    ):
        format_data_for_training(
            in_wav_dir_name=in_wav_dir_name,
            output_filename=wav_str_filename,
            n_max_files=n_max_files,
        )

    def test_get_clean_next_generated_text(self):
        assert get_clean_next_generated_text('-12142-1-32-') == '121'
        assert get_clean_next_generated_text('-12-1-32-') == '12'


if __name__ == "__main__":
    unittest.main()
