
import os
import fire

from aitextgen import aitextgen
from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import build_gpt2_config

from data_parsing_helpers.make_wav_str_file import convert_wav_to_text_file
from generate_gpt2_text import generate_text, clean_model_output

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_gpt2(
    steps: int = 100,
    n_max_files: int = 1,
    in_wav_dir_name: str = "sound_data",
    wav_str_filename: str = "sound.txt",
    learning_rate=1e-3,
) -> None:

    print(f"Creating file {wav_str_filename} from dir {in_wav_dir_name}\n")
    convert_wav_to_text_file(
        in_wav_dir_name=in_wav_dir_name,
        out_text_filename=wav_str_filename,
        n_max_files=n_max_files,
    )

    n_tokens = len(set(TokenDataset(wav_str_filename, block_size=32).tokens))  # can also be arbitrary int, e.g. 1000
    print(f"Found vocab of size {n_tokens}\n")
    train_tokenizer(wav_str_filename, vocab_size=n_tokens)

    config = build_gpt2_config(
        vocab_size=n_tokens,
        max_length=32,
        dropout=0.0,
        n_embd=256,
        n_layer=8,
        n_head=8,
    )
    ai = aitextgen(tokenizer_file="aitextgen.tokenizer.json", config=config)

    print(f"Training ({steps} epochs) with learning rate {learning_rate}\n")
    ai.train(
        train_data=wav_str_filename,
        num_steps=steps,
        learning_rate=learning_rate,
        batch_size=16,
        output_dir="trained_model",
    )

    raw_generated_texts = ai.generate(n=5, max_length=1000, prompt="", return_as_list=True)
    print("RAW:\n", *raw_generated_texts, sep="\n" + "=" * 10 + "\n")

    print("\nCLEAN:")
    for raw_generated_text in raw_generated_texts:
        print(f"{clean_model_output(raw_generated_text)}\n", end="="*10 + "\n")

    generate_text(
        model_folder="trained_model",
        tokenizer_file="aitextgen.tokenizer.json",
        prompt="",
        min_text_length=10000,
        write_raw_output_to_filename="raw_generated_unformatted_wav.txt",
        write_clean_output_to_filename="clean_generated_formatted_hex_str.txt"
    )


if __name__ == "__main__":
    fire.Fire(train_gpt2)
