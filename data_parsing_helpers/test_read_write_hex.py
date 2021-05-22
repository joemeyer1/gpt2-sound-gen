#!usr/bin/env python3
# Copyright (c) Joe Meyer (2020). All rights reserved.


from data_parsing_helpers.vec_to_wav import vec_to_wav
from data_parsing_helpers.wav_to_vector import extract_data


def main(read_wav_from_filename):
    write_wav_to_filename = f"{read_wav_from_filename.split('.')[0]}_data_channels"
    _, header_info = extract_data(read_wav_from_filename=read_wav_from_filename, write_wav_to_filename=write_wav_to_filename)
    vec_to_wav(write_wav_to_filename, header_info)


if __name__ == "__main__":
    main(read_wav_from_filename="sound_files/violin_G4_phrase_forte_harmonic-glissando.wav")
