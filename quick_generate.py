#!usr/bin/env python3
# Copyright (c) Joseph Meyer. All rights reserved.

from generate_output import generate_wav


def test_generate_gpt2():
    generate_wav(
        model_folder="trained_percussion_model4",
        tokenizer_file="percussion_tokenizer0.tokenizer.json",
        prompt="<|endoftext|>",
        min_audio_samples=100000,
        window_length=32,
        write_wav_to_filename="/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_output_percussion/generated_percussion.wav",
        overwrite_previous_model_data=False,
        num_channels=1,
        sample_rate=48000,
        bits_per_sample=16,
    )


if __name__ == '__main__':
    test_generate_gpt2()
