#!usr/bin/env python3
# Copyright (c) Joe Meyer (2021). All rights reserved.

import os
from typing import Optional

import fire
from tqdm import tqdm

from aitextgen.aitextgen.utils import model_max_length
from aitextgen.aitextgen import aitextgen

from data_parsing_helpers.file_helpers import unbin_data
from data_parsing_helpers.vec_to_wav import write_header, int_to_hex


def generate_wav(
        model_folder: str,  # e.g. 'trained_model'
        tokenizer_file: str,  # e.g. 'aitextgen.tokenizer.json'
        write_wav_to_filename: str,  # e.g. 'generated_sound.wav'
        min_audio_samples: int,
        window_length: int,
        num_channels: int,
        sample_rate: int,
        bits_per_sample: int,
        prompt: str = "",
        overwrite_previous_model_data: bool = False,
) -> None:
    header_info = {
        "num_channels": num_channels,
        "sample_rate": sample_rate,
        "bits_per_sample": bits_per_sample,
    }
    clean_generated_wav_txt = _generate_raw(
        model_folder=model_folder,
        tokenizer_file=tokenizer_file,
        prompt=prompt,
        min_audio_length=min_audio_samples,
        window_length=window_length,
        overwrite_previous_model_data=overwrite_previous_model_data,
    )
    if prompt == '<|endoftext|>':
        clean_generated_wav_txt = clean_generated_wav_txt[len(prompt):]
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


def _generate_raw(
        model_folder: str,
        tokenizer_file: str,
        prompt: str,
        min_audio_length: int,
        window_length: int,
        overwrite_previous_model_data: bool,
        write_raw_output_to_filename=None,
) -> str:
    """Returns clean model output, consisting of integers in range(0, 255) corresponding to pressure bins."""

    if not overwrite_previous_model_data:
        if write_raw_output_to_filename:
            write_raw_output_to_filename = make_name_unique(write_raw_output_to_filename)

    ai = aitextgen(model_folder=model_folder, tokenizer_file=tokenizer_file,)

    max_block_size = model_max_length(ai.model.config)
    assert window_length + 4 < max_block_size, f"window_length + 4 generated chars: {window_length + 4} " \
                                               f"cannot be greater than the max tokens model can handle: {max_block_size}"

    generated_data = prompt
    audio_length = 0
    with tqdm(total=min_audio_length, desc="generating output tokens") as t:
        while audio_length < min_audio_length:
            generated_text_up_to_prompt = generated_data[:-window_length]
            next_generated_text_prompt = generated_data[-window_length:]
            next_generated_text = ai.generate(
                n=1,
                max_length=512,
                min_length=4,
                # batch_size=100,
                prompt=next_generated_text_prompt,
                return_as_list=True,
                skip_special_tokens=False,
            )[0][len(next_generated_text_prompt):]
            clean_next_generated_bin = get_clean_next_generated_text(next_generated_text)
            if clean_next_generated_bin != '':
                generated_data = generated_text_up_to_prompt + next_generated_text_prompt + clean_next_generated_bin + '-'
                audio_length += 1
                t.update()
    if write_raw_output_to_filename:
        print(f"writing raw output to file '{write_raw_output_to_filename}'")
        with open(write_raw_output_to_filename, 'w') as f:
            f.write(generated_data)
    else:
        print(f"RAW:\n{generated_data}\n")
    return generated_data


def decode_generated_text(
        generated_text: str,
        bytes_per_sample: int,
        write_clean_output_to_filename: Optional[str] = None,
        overwrite_previous_model_data: bool = False,
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

    header_info['chunk_size'] = (n_pressure_samples * 4) + 36
    header_info['subchunk2size'] = n_pressure_samples * 4
    header = write_header(header_info)
    whole_str = header + raw_pressures
    print(whole_str)
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
        return chunk

    first_ints_chunk = get_first_ints_chunk(generated_text)
    if first_ints_chunk == '':
        return first_ints_chunk
    else:
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
    """Returns a formatted wav body given hex text like expected generated output."""

    generated_pressures = list(map(int, generated_text.split('-')[:-1]))
    unbinned_pressures = unbin_data(generated_pressures)
    hex_pressures = b''.join(map(
        lambda int_to_convert: int_to_hex(int_to_convert=int_to_convert, n_bytes=bytes_per_sample, signed=True),
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
