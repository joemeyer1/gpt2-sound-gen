#!usr/bin/env python3
# Copyright (c) Joseph Meyer (2020-2023). All rights reserved.

from generate_output import generate_wav
from train_gpt2 import train_gpt2, ModelData


def test_cycle_train_and_generate(n_cycles: int = 4):

    model_data = ModelData(
        tokenizer_filename='strings_tokenizer_cycle',
        model_dir="/Users/joemeyer/Documents/gpt2-sound-gen/trained_strings_model_cycle",
    )
    load_model_from_chkpt = None
    use_previous_training_data = False

    for _ in range(n_cycles):
        model_data: ModelData = train_gpt2(
            steps=10,
            n_max_files=10,
            in_wav_dir_name="/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_strings",
            formatted_training_data_filename="/Users/joemeyer/Documents/gpt2-sound-gen/formatted_training_data_strings/generated_strings_short_cycle_model.txt",
            output_dir="/Users/joemeyer/Documents/gpt2-sound-gen/trained_strings_model",
            tokenizer_name=model_data.tokenizer_filename,
            use_previous_training_data=use_previous_training_data,
            learning_rate=1e-5,
            load_model_from_chkpt=load_model_from_chkpt,
            save_model_every_n_epochs=10,
            overwrite_previous_model=False,
            block_size=1024,
        )
        generate_wav(
            model_folder=model_data.model_dir,
            tokenizer_file=f"{model_data.tokenizer_filename}.tokenizer.json",
            prompt="<|endoftext|>",
            min_audio_samples=1000,
            window_length=1000,
            write_wav_to_filename="/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_output_strings/generated_strings_cycle.wav",
            overwrite_previous_model_data=False,
            num_channels=1,
            sample_rate=48000,
            bits_per_sample=16,
        )
        load_model_from_chkpt = model_data.model_dir
        use_previous_training_data = True


if __name__ == '__main__':
    test_cycle_train_and_generate()
