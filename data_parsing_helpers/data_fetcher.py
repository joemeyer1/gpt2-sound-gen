#!usr/bin/env python3
# Copyright (c) Joe Meyer (2020). All rights reserved.

import os

import numpy as np

from data_parsing_helpers.wav_to_vector import extract_binned_data


def get_training_data(read_wav_from_dir: str, n_max_files: int = 0) -> np.array:
    """Returns 2D array of 1D vectors of ints representing wav files in directory, up to n_max_files.

    Args:
        read_wav_from_dir: Directory to read wav files from.
        n_max_files: Max number of files to read. Reads all files in directory if set to 0.
    """

    vectors_list = []
    filenames = [f for f in os.listdir(read_wav_from_dir) if f != '.DS_Store']
    np.random.shuffle(filenames)
    for filename in filenames[-n_max_files:]:
        read_wav_from_path = f"{read_wav_from_dir}/{filename}"
        data_channels, _ = extract_binned_data(read_wav_from_filename=read_wav_from_path)
        flat_data_channels = np.array(list(data_channels.values())).flatten()
        vectors_list.append(flat_data_channels)
    biggest_vector_len = max([len(vec) for vec in vectors_list])
    for i in range(len(vectors_list)):
        padding = [0] * (biggest_vector_len - len(vectors_list[i]))
        vectors_list[i] = np.concatenate((vectors_list[i], padding))
    vectors = np.stack(vectors_list)
    return vectors




if __name__ == "__main__":
    # main("test.wav")
    get_training_data("sound_files")