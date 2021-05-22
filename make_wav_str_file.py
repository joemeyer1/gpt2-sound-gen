#!usr/bin/env python3

# original code retrieved from https://pypi.org/project/keras-transformer/ under MIT license.
# If you want to use the original code under the MIT license, download it from the link above.

# Copyright (c) Joe Meyer (2021). All rights reserved.


from model_files.data_fetcher import get_training_data
from vec_to_wav import int_to_hex

def convert_wav_to_text_file(
        in_wav_dir_name: str = "sound_files",
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

    # vec_int_to_hex = np.vectorize(lambda i: int_to_hex(int_to_convert=i, bytes=4))
    ints_data = get_training_data(read_wav_from_dir=in_wav_dir_name, n_max_files=n_max_files)
    # tokens = tokens.flatten()
    # # bucketize tokens to avoid having like tens of thousands
    # mass, bins = np.histogram(a=tokens, bins=16)
    # tokens = vec_int_to_hex(ints_data)
    new_tokens = []
    for file_tokens in ints_data:
        new_tokens.append('<START>')
        for token in file_tokens:
            new_tokens += hex_to_tokens(int_to_hex(int_to_convert=token, bytes=4)) + ['-']
        new_tokens.append('<END>')
    tokens_str = "".join(map(str, new_tokens))
    with open(out_text_filename, 'w') as f:
        f.write(tokens_str)
