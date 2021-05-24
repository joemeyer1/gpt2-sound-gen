#!usr/bin/env python3
# Copyright (c) Joe Meyer (2021). All rights reserved.

import os
from tqdm import tqdm

from typing import Optional

import fire
from aitextgen import aitextgen

from data_parsing_helpers.vec_to_wav import write_header, str_to_hex


def generate_wav(
        model_folder="trained_model_10k_epochs",
        tokenizer_file="aitextgen.tokenizer.json",
        prompt="",
        min_text_length=10000,
        window_length=16,
        write_wav_to_filename="trash.wav",
        overwrite_previous_model_data=False,
        num_channels=1,
        sample_rate=48000,
        bits_per_sample=16,
) -> None:
    clean_generated_wav_txt = generate_text(
        model_folder=model_folder,
        tokenizer_file=tokenizer_file,
        prompt=prompt,
        min_text_length=min_text_length,
        window_length=window_length,
        overwrite_previous_model_data=overwrite_previous_model_data,
    )
    header_info = {
        "num_channels": num_channels,
        "sample_rate": sample_rate,
        "bits_per_sample": bits_per_sample,
    }
    if not overwrite_previous_model_data:
        write_wav_to_filename = make_name_unique(write_wav_to_filename)
    write_wav(wav_txt=clean_generated_wav_txt, write_wav_to_filename=write_wav_to_filename, header_info=header_info)


def generate_text(
        model_folder="trained_model_10k_epochs",
        tokenizer_file="aitextgen.tokenizer.json",
        prompt="",
        min_text_length=10000,
        window_length=16,
        write_raw_output_to_filename=None,
        write_clean_output_to_filename=None,
        overwrite_previous_model_data=False,
) -> str:

    if overwrite_previous_model_data:
        if write_raw_output_to_filename:
            write_raw_output_to_filename = make_name_unique(write_raw_output_to_filename)
        if write_clean_output_to_filename:
            write_clean_output_to_filename = make_name_unique(write_clean_output_to_filename)

    ai = aitextgen(model_folder=model_folder, tokenizer_file=tokenizer_file,)
    raw_generated_wav_txt = prompt
    t = tqdm(total=min_text_length)
    while len(raw_generated_wav_txt) < min_text_length:
        generated_text_up_to_prompt = raw_generated_wav_txt[:-window_length]
        next_generated_text_prompt = raw_generated_wav_txt[-window_length:]
        next_generated_text = ai.generate(n=1, max_length=512, batch_size=100, prompt=next_generated_text_prompt, return_as_list=True)[0]#.split('-')[0]
        raw_generated_wav_txt = generated_text_up_to_prompt + next_generated_text
        t.update(len(next_generated_text) - len(next_generated_text_prompt))
    t.close()
    if write_raw_output_to_filename:
        with open(write_raw_output_to_filename, 'w') as f:
            f.write(raw_generated_wav_txt)
    else:
        print(f"RAW:\n{raw_generated_wav_txt}\n")
    clean_generated_wav_txt = clean_model_output(raw_generated_wav_txt)

    clean_generated_wav_txt = format_wav_body(hex_text=clean_generated_wav_txt)
    if write_clean_output_to_filename:
        with open(write_clean_output_to_filename, 'w') as f:
            f.write(clean_generated_wav_txt)
    else:
        print(f"CLEAN:\n{clean_generated_wav_txt}\n")
    return clean_generated_wav_txt

def write_wav(wav_txt: str, write_wav_to_filename: str, header_info=None):
    if not header_info:
        header_info = {
            "num_channels": 1,
            "sample_rate": 48000,
            "bits_per_sample": 16,
        }
    wav_txt = wav_txt.replace(' ', '').replace('\n', '')
    len_txt = len(wav_txt)
    header_info['chunk_size'] = (len_txt // 2) + 36
    header_info['subchunk2size'] = len_txt // 2
    header_str = write_header(header_info)
    print(header_str)
    body_str = str_to_hex(wav_txt)
    whole_str = header_str + body_str
    print(whole_str)
    with open(write_wav_to_filename, 'wb') as f:
        f.write(whole_str)

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

    def get_last_dot_i(name) -> Optional[int]:
        j_range = list(range(1, len(name)))
        j_range.reverse()
        for j in j_range:
            if name[j] == '.':
                return j
        return None

    last_dot_i = get_last_dot_i(name)
    if last_dot_i:
        raw_name, ext = name[:last_dot_i], name[last_dot_i:]
    else:
        raw_name, ext = name, ''
    i = 0
    while os.path.exists(raw_name + str(i) + ext):
        i += 1
    return raw_name + str(i) + ext


if __name__ == "__main__":
    fire.Fire(generate_wav)
