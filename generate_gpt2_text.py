
import os

import fire
from aitextgen import aitextgen


def generate_text(
        model_folder="trained_model",
        tokenizer_file="aitextgen.tokenizer.json",
        prompt="",
        min_text_length=10000,
        window_length=16,
        write_raw_output_to_filename=None,
        write_clean_output_to_filename=None,
        overwrite_previous_model_data=False,
) -> None:

    if overwrite_previous_model_data:
        if write_raw_output_to_filename:
            write_raw_output_to_filename = make_name_unique(write_raw_output_to_filename)
        if write_clean_output_to_filename:
            write_clean_output_to_filename = make_name_unique(write_clean_output_to_filename)

    ai = aitextgen(model_folder=model_folder, tokenizer_file=tokenizer_file,)
    raw_generated_wav_txt = prompt
    while len(raw_generated_wav_txt) < min_text_length:
        raw_generated_wav_txt = raw_generated_wav_txt[:-window_length] + ai.generate(n=1, max_length=512, batch_size=100, prompt=raw_generated_wav_txt[-window_length:], return_as_list=True)[0]#.split('-')[0]
    if write_raw_output_to_filename:
        with open(write_raw_output_to_filename, 'w') as f:
            f.write(raw_generated_wav_txt)
    else:
        print(f"RAW:\n{raw_generated_wav_txt}\n")
    clean_generated_wav_txt = clean_model_output(raw_generated_wav_txt)

    clean_generated_wav_txt = format_wav_body(hex_text=clean_generated_wav_txt)
    # TODO: write this sound data as a .wav file - figure out how to find + write header info
    if write_clean_output_to_filename:
        with open(write_clean_output_to_filename, 'w') as f:
            f.write(clean_generated_wav_txt)
    else:
        print(f"CLEAN:\n{clean_generated_wav_txt}\n")

def clean_model_output(model_output: str, bits_per_word=8) -> str:
    clean_output = ""
    model_output_words = model_output.split('-')
    for model_output_word in model_output_words:
        truncated_word = model_output_word[:bits_per_word]
        padding = "0" * (bits_per_word - len(truncated_word))
        clean_model_output_item = padding + truncated_word + '-'
        clean_output += clean_model_output_item
    return clean_output


def format_wav_body(hex_text: str) -> str:
    """Returns a formatted wav body given hex text like expected gpt2 output."""

    word_list = hex_text.split('-')
    wav_body = ""
    i = 0
    for word in word_list:
        worda, wordb = word[:4], word[4:]
        wav_body += worda + " " + wordb
        i += 1
        if not i % 4:
            wav_body += '\n'
        else:
            wav_body += ' '
    return wav_body

def make_name_unique(name: str) -> str:
    i = 0
    while os.path.exists(name):
        i += 1
    return name + str(i)


if __name__ == "__main__":
    fire.Fire(generate_text)
