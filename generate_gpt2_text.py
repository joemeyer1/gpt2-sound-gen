#!usr/bin/env python3
# Copyright (c) Joe Meyer (2021). All rights reserved.

import os
from tqdm import tqdm

from typing import Optional, List

import fire
from aitextgen.aitextgen import aitextgen

from data_parsing_helpers.vec_to_wav import write_header, str_to_hex, int_to_hex
from data_parsing_helpers.file_helpers import unbin_data


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
    header_info = {
        "num_channels": num_channels,
        "sample_rate": sample_rate,
        "bits_per_sample": bits_per_sample,
    }
    clean_generated_wav_txt = generate_text(
        model_folder=model_folder,
        tokenizer_file=tokenizer_file,
        prompt=prompt,
        min_text_length=min_text_length,
        window_length=window_length,
        overwrite_previous_model_data=overwrite_previous_model_data,
    )
    n_pressure_samples = clean_generated_wav_txt.count('-')
    restored_wav = decode_generated_text(
        generated_text=clean_generated_wav_txt,
        bytes_per_sample=header_info['bits_per_sample'] // 8,
    )
    if not overwrite_previous_model_data:
        write_wav_to_filename = make_name_unique(write_wav_to_filename)
    print(f"writing wav file '{write_wav_to_filename}'")
    write_wav(
        raw_pressures=restored_wav,
        write_wav_to_filename=write_wav_to_filename,
        header_info=header_info,
        n_pressure_samples=n_pressure_samples,
    )


def generate_text(
        model_folder="trained_model_10k_epochs",
        tokenizer_file="aitextgen.tokenizer.json",
        prompt="",
        min_text_length=10000,
        window_length=16,
        write_raw_output_to_filename=None,
        overwrite_previous_model_data=True,
) -> str:
    """Returns clean model output, consisting of integers in range(0, 255) corresponding to pressure bins."""

    if not overwrite_previous_model_data:
        if write_raw_output_to_filename:
            write_raw_output_to_filename = make_name_unique(write_raw_output_to_filename)

    ai = aitextgen(model_folder=model_folder, tokenizer_file=tokenizer_file,)
    raw_generated_wav_txt = prompt
    with tqdm(total=min_text_length, desc="generating gpt2 output tokens") as t:
        while len(raw_generated_wav_txt) < min_text_length:
            generated_text_up_to_prompt = raw_generated_wav_txt[:-window_length]
            next_generated_text_prompt = raw_generated_wav_txt[-window_length:]
            next_generated_text = ai.generate(
                n=1,
                max_length=512,
                min_length=4,
                # batch_size=100,
                prompt=next_generated_text_prompt,
                return_as_list=True
            )[0][len(next_generated_text_prompt):]#.split('-')[0]]
            clean_next_generated_text = get_clean_next_generated_text(next_generated_text)
            raw_generated_wav_txt = generated_text_up_to_prompt + next_generated_text_prompt + clean_next_generated_text + '-'
            t.update(len(clean_next_generated_text) - len(next_generated_text_prompt))
    if write_raw_output_to_filename:
        print(f"writing raw output to file '{write_raw_output_to_filename}'")
        with open(write_raw_output_to_filename, 'w') as f:
            f.write(raw_generated_wav_txt)
    else:
        print(f"RAW:\n{raw_generated_wav_txt}\n")
    return raw_generated_wav_txt


def decode_generated_text(
        generated_text: str,
        bytes_per_sample: int,
        write_clean_output_to_filename: Optional[str] = None,
        overwrite_previous_model_data: bool = True,
) -> bytes:
    """Restores binned audio data back to hexadecimal bytes."""

    restored_audio_pressures = restore_audio_pressures(generated_text=generated_text, bytes_per_sample=bytes_per_sample)
    if write_clean_output_to_filename:
        if not overwrite_previous_model_data:
            write_clean_output_to_filename = make_name_unique(write_clean_output_to_filename)
        with open(write_clean_output_to_filename, 'w') as f:
            print(f"writing clean output to file '{write_clean_output_to_filename}'")
            f.write(restored_audio_pressures)
    else:
        print(f"CLEAN:\n{restored_audio_pressures}\n")
    return restored_audio_pressures


def write_wav(
        raw_pressures: bytes,
        write_wav_to_filename: str,
        n_pressure_samples: int,
        header_info=None,
):
    if not header_info:
        header_info = {
            "num_channels": 1,
            "sample_rate": 48000,
            "bits_per_sample": 16,
        }
    # wav_txt = wav_txt.replace(' ', '').replace('\n', '')
    # len_txt = len(wav_txt)
    header_info['chunk_size'] = (n_pressure_samples * 4) + 36
    header_info['subchunk2size'] = n_pressure_samples * 4
    header = write_header(header_info)
    header_str = header.hex()
    print(header_str)
    whole_str = header_str + raw_pressures
    print(whole_str)
    # pretty_wav = add_spaces_and_linebreaks(whole_str)
    # print(pretty_wav)
    with open(write_wav_to_filename, 'wb') as f:
        f.write(whole_str)


def get_clean_next_generated_text(generated_text: str) -> str:

    def get_first_ints_chunk(text: str) -> str:
        chunk = ''
        for i in range(len(text)):
            if text[i] == '-' and len(chunk) > 0:
                return chunk
            elif text[i].isdigit():
                chunk += text[i]
                if len(chunk) == 3:
                    return chunk

    first_ints_chunk = get_first_ints_chunk(generated_text)
    return str(min(int(first_ints_chunk), 255))


def clean_model_output(model_output: str, bits_per_word=8) -> str:
    clean_output = ""
    model_output_words = model_output.split('-')
    for model_output_word in model_output_words:
        truncated_word = model_output_word[:bits_per_word]
        padding = "0" * (bits_per_word - len(truncated_word))
        clean_model_output_item = padding + truncated_word + '-'
        clean_output += clean_model_output_item
    return clean_output


def restore_audio_pressures(generated_text: str, bytes_per_sample: int) -> bytes:
    """Returns a formatted wav body given hex text like expected gpt2 output."""

    generated_pressures = list(map(int, generated_text.split('-')[:-1]))
    unbinned_pressures = unbin_data(generated_pressures)
    hex_pressures = b''.join(map(
        lambda int_to_convert: int_to_hex(int_to_convert=int_to_convert, bytes=bytes_per_sample, signed=True),
        unbinned_pressures,
    ))
    return hex_pressures


def add_spaces_and_linebreaks(audio_pressures: bytes) -> str:
    audio_with_spaces_and_linebreaks: str = ''
    for i, ch in enumerate(audio_pressures):
        if i != 0:
            if (i % 32) == 0:
                audio_with_spaces_and_linebreaks += '\n'
            elif (i % 4) == 0:
                audio_with_spaces_and_linebreaks += ' '

        audio_with_spaces_and_linebreaks += ch

    return audio_with_spaces_and_linebreaks
    # ugly_str = str(audio_pressures).replace('\\', '').replace('b', '').replace("\'", '').replace('x', '')
    # wav_body = ""
    # return hex_pressures
    # i = 0
    # for hex_pressure in hex_pressures:
    #     int_to_hex(int_to_convert=word, bytes=bytes_per_sample, signed=True)
    #     worda, wordb = word[:4], word[4:]
    #     wav_body += hex_pressure + " "
    #     i += 1
    #     if not i % 4:
    #         wav_body += '\n'
    #     else:
    #         wav_body += ' '
    # return wav_body

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
