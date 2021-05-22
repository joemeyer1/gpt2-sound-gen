
import os
import fire

from aitextgen import aitextgen
from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import build_gpt2_config

from make_wav_str_file import convert_wav_to_text_file

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_gpt2_script(
    steps: int = 100,
    n_max_files: int = 1,
    in_wav_dir_name: str = "sound_files",
    wav_str_filename: str = "sound.txt",
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

    print(f"Training ({steps} epochs)\n")
    ai.train(wav_str_filename, batch_size=16, num_steps=steps)
    ai.generate(n=100, prompt="")


if __name__ == "__main__":
    fire.Fire(run_gpt2_script)
