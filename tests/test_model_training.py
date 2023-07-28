#!usr/bin/env python3
# Copyright (c) Joseph Meyer. All rights reserved.

from train_gpt2 import train_gpt2 as train


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
