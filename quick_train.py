#!usr/bin/env python3
# Copyright (c) Joseph Meyer (2020-2023). All rights reserved.

from main.train_gpt2 import train_gpt2


def quick_train():
    train_gpt2(
        steps=100000,
        n_max_files=150,
        in_wav_dir_name="/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_strings",
        formatted_training_data_filename="/Users/joemeyer/Documents/gpt2-sound-gen/formatted_training_data_strings/generated_strings_short_model3.txt",
        output_dir="/Users/joemeyer/Documents/gpt2-sound-gen/trained_strings_model",
        tokenizer_name='strings_tokenizer_short0',
        use_previously_formatted_training_data=False,
        learning_rate=1e-5,
        load_model_from_chkpt="/Users/joemeyer/Documents/gpt2-sound-gen/trained_strings_model2",
        save_model_every_n_epochs=1000,
        overwrite_previous_model=False,
        block_size=1024,
    )


if __name__ == '__main__':
    quick_train()
