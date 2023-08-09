#!usr/bin/env python3
# Copyright (c) Joseph Meyer (2020-2023). All rights reserved.

from typing import Optional

from main.generate_output import generate_wav
from main.train_gpt2 import train_gpt2, ModelData


def test_cycle_train_and_generate(
        n_cycles: int = 4,
        n_max_files: int = 100,
        base_model_dir: str = "/Users/joemeyer/Documents/gpt2-sound-gen/trained_strings_model_cycle000",
        load_model_from_chkpt: Optional[str] = None,
        tokenizer_filename: str = "/Users/joemeyer/Documents/gpt2-sound-gen/tokenizers/strings_tokenizer_cycle000",
        refresh_data_every_n_epochs: int = 1000,
        gen_sample_every_n_epochs: int = 500,
        write_wav_to_filename: str = "/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_output_strings/generated_strings_cycle000.wav",
) -> None:

    model_data = ModelData(
        tokenizer_filename=tokenizer_filename,
        model_dir=base_model_dir,
    )

    n_epochs_until_sample_generation = 0  # save model and generate at epoch 0
    n_epochs_until_data_refresh = refresh_data_every_n_epochs
    use_previously_formatted_training_data = False
    for i in range(n_cycles):
        while n_epochs_until_data_refresh > 0 or n_epochs_until_sample_generation > 0:

            if n_epochs_until_data_refresh <= 0:
                use_previously_formatted_training_data = False
                n_epochs_until_data_refresh = refresh_data_every_n_epochs

            print('\n\nTRAINING::\n')
            steps: int = min(gen_sample_every_n_epochs, n_epochs_until_data_refresh)
            print(f"model_data.model_dir: {model_data.model_dir}\n")
            model_data: ModelData = train_gpt2(
                steps=steps,
                n_max_files=n_max_files,
                in_wav_dir_name="/Users/joemeyer/Documents/gpt2-sound-gen/sound_data_strings",
                formatted_training_data_filename="/Users/joemeyer/Documents/gpt2-sound-gen/formatted_training_data_strings/generated_strings_short_cycle_model.txt",
                output_dir=base_model_dir,
                tokenizer_name=model_data.tokenizer_filename,
                use_previously_formatted_training_data=use_previously_formatted_training_data,
                learning_rate=1e-2,
                load_model_from_chkpt=load_model_from_chkpt,
                save_model_every_n_epochs=steps + 1,  # model will save after steps are done, so avoid writing twice
                overwrite_previous_model=False,
                block_size=1024,
            )
            load_model_from_chkpt = model_data.model_dir

            use_previously_formatted_training_data = True
            n_epochs_until_data_refresh -= steps
            n_epochs_until_sample_generation -= steps

            if n_epochs_until_sample_generation <= 0:
                print('\n\nGENERATING::\n')
                generate_wav(
                    model_folder=model_data.model_dir,
                    tokenizer_file=f"{model_data.tokenizer_filename}.tokenizer.json",
                    prompt="<|endoftext|>",
                    min_audio_samples=1000,
                    window_length=1000,
                    write_wav_to_filename=write_wav_to_filename,
                    overwrite_previous_model_data=False,
                    num_channels=1,
                    sample_rate=48000,
                    bits_per_sample=16,
                )
                n_epochs_until_sample_generation = gen_sample_every_n_epochs


if __name__ == '__main__':
    test_cycle_train_and_generate()
