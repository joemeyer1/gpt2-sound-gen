#!usr/bin/env python3
# Copyright (c) Joseph Meyer (2020-2023). All rights reserved.

from main.generate_output import generate_wav
from main.train_gpt2 import train_gpt2, ModelData


def test_cycle_train_and_generate(n_cycles: int = 4):

    base_model_dir = "/Users/joemeyer/Documents/gpt2-sound-gen/trained_strings_model_cycle"
    model_data = ModelData(
        tokenizer_filename="/Users/joemeyer/Documents/gpt2-sound-gen/tokenizers/strings_tokenizer_cycle",
        model_dir=base_model_dir,
    )
    load_model_from_chkpt = None

    for i in range(n_cycles):
        print('\n\nTRAINING::\n')
        steps: int = 100 if i != 0 else 0
        print(f"model_data.model_dir: {model_data.model_dir}\n")
        model_data: ModelData = train_gpt2(
            steps=steps,
            n_max_files=2,
            in_wav_dir_name="/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_strings",
            formatted_training_data_filename="/Users/joemeyer/Documents/gpt2-sound-gen/formatted_training_data_strings/generated_strings_short_cycle_model.txt",
            output_dir=base_model_dir,
            tokenizer_name=model_data.tokenizer_filename,
            use_previous_training_data=False,
            learning_rate=1e-3,
            load_model_from_chkpt=load_model_from_chkpt,
            save_model_every_n_epochs=steps,
            overwrite_previous_model=False,
            block_size=1024,
        )
        print('\n\nGENERATING::\n')
        generate_wav(
            model_folder=model_data.model_dir,
            tokenizer_file=f"{model_data.tokenizer_filename}.tokenizer.json",
            prompt="<|endoftext|>",
            min_audio_samples=100,
            window_length=100,
            write_wav_to_filename="/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_output_strings/generated_strings_cycle.wav",
            overwrite_previous_model_data=False,
            num_channels=1,
            sample_rate=48000,
            bits_per_sample=16,
        )
        load_model_from_chkpt = model_data.model_dir


if __name__ == '__main__':
    test_cycle_train_and_generate()
