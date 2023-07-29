#!usr/bin/env python3
# Copyright (c) Joseph Meyer. All rights reserved.

import unittest

from train_gpt2 import train_gpt2
from generate_gpt2_text import generate_wav


class GPTTests(unittest.TestCase):
    def test_train_gpt2(self):
        train_gpt2(
            steps=1,
            n_max_files=1,
            in_wav_dir_name="/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_toy",
            wav_str_filename="/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_output_toy/generated_sound_toy.txt",
            use_previous_training_data=False,
            learning_rate=1e-3,
            load_model_from_chkpt=None,
            save_model_every_n_epochs=1,
            overwrite_previous_model=False,
        )

    def test_generate_gpt2(self):
        generate_wav(
            model_folder="trained_model7",
            tokenizer_file="aitextgen000.tokenizer.json",
            prompt="<|endoftext|>",
            min_audio_length=100,
            window_length=16,
            write_wav_to_filename="/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_output_toy/test_generate.wav",
            overwrite_previous_model_data=False,
            num_channels=1,
            sample_rate=48000,
            bits_per_sample=16,
        )


if __name__ == "__main__":
    unittest.main()
