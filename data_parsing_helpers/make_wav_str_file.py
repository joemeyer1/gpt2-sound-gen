#!usr/bin/env python3
# Copyright (c) Joe Meyer (2021). All rights reserved.

import fire
from tqdm import tqdm

from data_parsing_helpers.data_fetcher import get_training_data
from data_parsing_helpers.vec_to_wav import int_to_hex


def convert_wav_to_text_file(
        in_wav_dir_name: str = "sound_data",
        out_text_filename: str = "sound.txt",
        n_max_files: int = 1,
) -> None:

    def hex_to_tokens(hex):
        tokens = hex.hex()
        tokens_list = []
        chars_per_token = 1
        for i in range(0, len(tokens), chars_per_token):
            tokens_list.append(tokens[i: i + chars_per_token])
        return tokens_list

    ints_data = get_training_data(read_wav_from_dir=in_wav_dir_name, n_max_files=n_max_files)
    new_tokens = ['<endoftext>']
    with tqdm(total=len(ints_data), desc="Formatting training data files") as t:
        for file_tokens in ints_data:
            # new_tokens.append('<START>')
            for token in file_tokens:
                new_tokens += hex_to_tokens(int_to_hex(int_to_convert=token, bytes=4)) + ['-']
            new_tokens.append('<endoftext>')
            t.update()
    print(f"Writing formatted training file to {out_text_filename}")
    tokens_str = "".join(map(str, new_tokens))
    with open(out_text_filename, 'w') as f:
        f.write(tokens_str)

if __name__ == "__main__":
    fire.Fire(convert_wav_to_text_file)
