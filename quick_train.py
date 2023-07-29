#!usr/bin/env python3
# Copyright (c) Joseph Meyer. All rights reserved.

from train_gpt2 import train_gpt2 as train


def quick_train():
    train(
        steps=100,
        n_max_files=10000,
        in_wav_dir_name="/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_percussion",
        wav_str_filename="/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_output_percussion/generated_percussion.txt",
        output_dir="/Users/joemeyer/Documents/gpt2-sound-gen/trained_percussion_model",
        tokenizer_name='/Users/joemeyer/Documents/gpt2-sound-gen/percussion_tokenizer',
        use_previous_training_data=False,
        learning_rate=1e-3,
        load_model_from_chkpt="/Users/joemeyer/Documents/gpt2-sound-gen/trained_percussion_model",
        save_model_every_n_epochs=10,
        overwrite_previous_model=False,
    )


if __name__ == '__main__':
    quick_train()
