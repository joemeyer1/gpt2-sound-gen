#!usr/bin/env python3
# Copyright (c) Joseph Meyer. All rights reserved.

from train_gpt2 import train_gpt2 as train
from generate_gpt2_text import generate_wav


def test_train_gpt2():
    train(
        steps=1,
        n_max_files=1,
        in_wav_dir_name="/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_toy",
        wav_str_filename="/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_output_toy/sound.txt",
        use_previous_training_data=False,
        learning_rate=1e-3,
        load_model_from_chkpt=None,
        save_model_every_n_epochs=1,
        overwrite_previous_model=False,
    )


def test_generate_gpt2():
    generate_wav(
        model_folder="trained_model7",
        tokenizer_file="aitextgen000.tokenizer.json",
        prompt="",
        min_text_length=100,
        window_length=16,
        write_wav_to_filename="trash.wav",
        overwrite_previous_model_data=False,
        num_channels=1,
        sample_rate=48000,
        bits_per_sample=16,
    )
