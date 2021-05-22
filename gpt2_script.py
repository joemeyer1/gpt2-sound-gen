
from aitextgen import aitextgen
from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import build_gpt2_config

from make_wav_str_file import convert_wav_to_text_file


def run_gpt2_script(
    steps: int = 100,
    wav_str_filename: str = "sound.txt"
) -> None:

    convert_wav_to_text_file(
        in_wav_dir_name="sound_files",
        out_text_filename=wav_str_filename,
        n_max_files=1,
    )

    print(f"Tokenizing file name {wav_str_filename}")
    train_tokenizer(wav_str_filename)

    config = build_gpt2_config(
        vocab_size=len(set(TokenDataset(wav_str_filename, block_size=32).tokens)),
        max_length=32,
        dropout=0.0,
        n_embd=256,
        n_layer=8,
        n_head=8,
    )
    ai = aitextgen(tokenizer_file="aitextgen.tokenizer.json", config=config)

    print(f"Training ({steps} epochs}")
    ai.train(wav_str_filename, batch_size=16, num_steps=steps)
    ai.generate(n=100, prompt="")


if __name__ == "__main__":
    run_gpt2_script()
