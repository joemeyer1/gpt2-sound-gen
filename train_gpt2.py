#!usr/bin/env python3
# Copyright (c) Joe Meyer (2021). All rights reserved.

import os
from typing import Optional

import fire

from aitextgen.aitextgen import aitextgen
from aitextgen.aitextgen.TokenDataset import TokenDataset
from aitextgen.aitextgen.tokenizers import train_tokenizer
from aitextgen.aitextgen.utils import build_gpt2_config
from data_parsing_helpers.wav_to_vec import format_data_for_training
from generate_output import make_name_unique

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_gpt2(
    steps: int,
    n_max_files: int,
    in_wav_dir_name: str,
    formatted_training_data_filename: str,
    output_dir: str,
    tokenizer_name: str,
    use_previous_training_data: bool,
    learning_rate: float,
    save_model_every_n_epochs: int,
    overwrite_previous_model: bool,
    load_model_from_chkpt: Optional[str] = None,
    block_size: Optional[int] = None,  # only relevant if training new model
) -> None:

    if block_size:
        _MAX_BLOCK_SIZE = 1024
        assert block_size <= _MAX_BLOCK_SIZE, f"block_size: {block_size} cannot exceed max_block_size: {_MAX_BLOCK_SIZE} -" \
                                          f" pre-trained GPT-2 cannot handle more tokens due to limited receptive field"

    if use_previous_training_data:
        assert os.path.exists(formatted_training_data_filename), f"training data {formatted_training_data_filename} not found"
        print(f"using previous training data file: {formatted_training_data_filename}")
    else:
        print(f"Creating file {formatted_training_data_filename} from dir {in_wav_dir_name}\n")
        format_data_for_training(
            in_wav_dir_name=in_wav_dir_name,
            output_filename=formatted_training_data_filename,
            n_max_files=n_max_files,
        )

    if not load_model_from_chkpt:
        n_tokens = len(set(TokenDataset(formatted_training_data_filename, block_size=block_size).tokens))  # can also be arbitrary int, e.g. 1000
        print(f"Found vocab of size {n_tokens}\n")
        if not overwrite_previous_model:
            tokenizer_name = make_name_unique(tokenizer_name)
        train_tokenizer(formatted_training_data_filename, vocab_size=n_tokens, prefix=tokenizer_name)

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
        train_data=formatted_training_data_filename,
        num_steps=steps,
        learning_rate=learning_rate,
        batch_size=16,
        output_dir=output_dir,
        save_every=save_model_every_n_epochs,
    )


if __name__ == "__main__":
    fire.Fire(train_gpt2)
