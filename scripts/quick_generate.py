#!usr/bin/env python3
# Copyright (c) Joseph Meyer (2020-2023). All rights reserved.

from main.generate_output import generate_wav


def test_generate_gpt2():
    generate_wav(
        model_folder="/Users/joemeyer/Documents/gpt2-sound-gen/trained_strings_model3",
        tokenizer_file="/Users/joemeyer/Documents/gpt2-sound-gen/strings_tokenizer_short0.tokenizer.json",
        prompt="<|endoftext|>",
        min_audio_samples=10000,
        window_length=1000,
        write_wav_to_filename="/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_output_strings/generated_strings.wav",
        overwrite_previous_model_data=False,
        num_channels=1,
        sample_rate=48000,
        bits_per_sample=16,
    )


if __name__ == '__main__':
    test_generate_gpt2()
