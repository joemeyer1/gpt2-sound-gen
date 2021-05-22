
import os
import fire

from aitextgen import aitextgen
from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import build_gpt2_config

from data_parsing_helpers.make_wav_str_file import convert_wav_to_text_file

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
    print("RAW:")
    print(*raw_generated_texts, sep="\n" + "=" * 10 + "\n")

    def clean_model_output(model_output: str, bits_per_word=8) -> str:
        clean_output = ""
        model_output_words = model_output.split('-')
        for model_output_word in model_output_words:
            truncated_word = model_output_word[:bits_per_word]
            padding = "0" * (bits_per_word - len(truncated_word))
            clean_model_output_item = padding + truncated_word + '-'
            clean_output += clean_model_output_item
        return clean_output

    print("\nCLEAN:")
    for raw_generated_text in raw_generated_texts:
        print(f"{clean_model_output(raw_generated_text)}\n", end="="*10 + "\n")

    raw_generated_wav_txt = ""
    while len(raw_generated_wav_txt) < 10000:
        raw_generated_wav_txt = raw_generated_wav_txt[:-16] + ai.generate(n=1, max_length=512, batch_size=100, prompt=raw_generated_wav_txt[-16:], return_as_list=True)[0]#.split('-')[0]
    with open("raw_generated_unformatted_wav.txt", 'w') as f:
        f.write(raw_generated_wav_txt)
    clean_generated_wav_txt = clean_model_output(raw_generated_wav_txt)
    word_list = clean_generated_wav_txt.split('-')
    hex_str = ""
    i = 0
    for word in word_list:
        worda, wordb = word[:4], word[4:]
        hex_str += worda + " " + wordb
        i += 1
        if not i % 4:
            hex_str += '\n'
        else:
            hex_str += ' '
    # TODO: write this sound data as a .wav file - figure out how to find + write header info
    with open("clean_generated_formatted_hex_str.txt", 'w') as f:
        f.write(hex_str)


if __name__ == "__main__":
    fire.Fire(train_gpt2)
