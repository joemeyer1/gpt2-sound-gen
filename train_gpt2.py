#!usr/bin/env python3
# Copyright (c) Joe Meyer (2021). All rights reserved.

import os
import fire

from aitextgen.aitextgen.TokenDataset import TokenDataset
from aitextgen.aitextgen.tokenizers import train_tokenizer
from aitextgen.aitextgen.utils import build_gpt2_config
from aitextgen.aitextgen import aitextgen

from data_parsing_helpers.make_wav_str_file import convert_wav_to_text_file
from generate_output import generate_wav, make_name_unique

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_gpt2(
    steps: int = 100,
    n_max_files: int = 1,
    in_wav_dir_name: str = "sound_data",
    wav_str_filename: str = "sound.txt",
    output_dir: str = "trained_model",
    tokenizer_name: str = "aitextgen00",
    use_previous_training_data: bool = True,
    learning_rate=1e-3,
    load_model_from_chkpt=None,
    save_model_every_n_epochs=1000,
    overwrite_previous_model=False,
) -> None:

    if use_previous_training_data:
        assert os.path.exists(wav_str_filename), f"training data {wav_str_filename} not found"
        print(f"using previous training data file: {wav_str_filename}")
    else:
        print(f"Creating file {wav_str_filename} from dir {in_wav_dir_name}\n")
        convert_wav_to_text_file(
            in_wav_dir_name=in_wav_dir_name,
            out_text_filename=wav_str_filename,
            n_max_files=n_max_files,
        )

    if not load_model_from_chkpt:
        block_size = 64
        n_tokens = len(set(TokenDataset(wav_str_filename, block_size=block_size).tokens))  # can also be arbitrary int, e.g. 1000
        print(f"Found vocab of size {n_tokens}\n")
        if not overwrite_previous_model:
            tokenizer_name = make_name_unique(tokenizer_name)
        train_tokenizer(wav_str_filename, vocab_size=n_tokens, prefix=tokenizer_name)

        config = build_gpt2_config(
            vocab_size=n_tokens,
            max_length=block_size,
            dropout=0.0,
            n_embd=256,
            n_layer=8,
            n_head=8,
        )
        print(f"building model with config {config}")
        ai = aitextgen(tokenizer_file=f"{tokenizer_name}.tokenizer.json", config=config)
    else:
        print(f"Loading gpt2 model from chkpt: '{load_model_from_chkpt}' with tokenizer: '{tokenizer_name}.tokenizer.json'")
        ai = aitextgen(
            model_folder=load_model_from_chkpt,
            tokenizer_file=f"{tokenizer_name}.tokenizer.json",
            to_gpu=False,
        )

    if not overwrite_previous_model:
        output_dir = make_name_unique(output_dir)
    print(f"Training model {output_dir} for {steps} epochs with learning rate {learning_rate}\n")
    ai.train(
        train_data=wav_str_filename,
        num_steps=steps,
        learning_rate=learning_rate,
        batch_size=16,
        output_dir=output_dir,
        save_every=save_model_every_n_epochs,
    )


if __name__ == "__main__":
    fire.Fire(train_gpt2)
