#!usr/bin/env python3
# Copyright (c) Joe Meyer (2021). All rights reserved.

import fire
from tqdm import tqdm

from data_parsing_helpers.data_fetcher import get_training_data


def convert_wav_to_text_file(
        in_wav_dir_name: str = "/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_toy",
        out_text_filename: str = "sound.txt",
        n_max_files: int = 1,
) -> None:

    ints_data = get_training_data(read_wav_from_dir=in_wav_dir_name, n_max_files=n_max_files)
    new_tokens = ['<endoftext>']
    with tqdm(total=len(ints_data), desc="Formatting training data files") as t:
        for file_tokens in ints_data:
            for token in file_tokens:
                new_tokens += [int(token), '-']
            new_tokens.append('<endoftext>')
            t.update()
    print(f"Converting {len(new_tokens)} tokens to str...")
    tokens_str = "".join(map(str, new_tokens))
    print(f"Writing {len(tokens_str)} chars to {out_text_filename}...")
    with open(out_text_filename, 'w') as f:
        f.write(tokens_str)

if __name__ == "__main__":
    fire.Fire(convert_wav_to_text_file)
