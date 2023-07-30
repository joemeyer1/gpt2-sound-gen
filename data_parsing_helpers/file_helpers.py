#!usr/bin/env python3
# Copyright (c) Joe Meyer (2020). All rights reserved.

from typing import List, Tuple

import numpy as np


def file_to_hex_ls(filename: str) -> List[str]:
    """Converts file to a list of hexadecimal-encoded bytes.

    e.g. a file with bytes '5249 4646' -> ['52', '49', '46', '46']
    """

    def _file_to_hex_str(filename: str):
        with open(filename, 'rb') as f:
            words = f.read()
        return words.hex()

    def _hex_str_to_ls(hex_str: str) -> List[str]:
        i = 0
        hex_ls = []
        while i < len(hex_str):
            hex_ls.append(hex_str[i:i+2])
            i += 2
        return hex_ls

    hex_str = _file_to_hex_str(filename)
    hex_ls = _hex_str_to_ls(hex_str)
    return hex_ls


def get_n_bytes(n: int, hex_ls: List[str], i: int) -> Tuple[List[str], int]:
    """Returns next n_bytes from hex_ls.

    e.g. get_n_bytes(2, ['52', '49', '46', '46'] , 1) -> (['49', '46'], 3)
    """

    n_bytes = hex_ls[i: i + n]
    i = i + n
    return n_bytes, i


def get_n_bytes_int(n: int, hex_ls: List[str], i: int) -> Tuple[int, int]:
    """Converts next n_bytes from hex_ls to an integer."""

    def _decode_endian_and_twos_comp(hex_ls: List[str]) -> int:
        # little endian (ones place at start)
        # e.g. ['10', '01'] = x0110 = 256 + 16 = 272
        hex_str = ''
        for h in hex_ls:
            hex_str = h + hex_str
        return _twos_comp(int(hex_str, 16), len(hex_str) * 4)

    n_bytes, i = get_n_bytes(n, hex_ls, i)
    decoded_bytes = _decode_endian_and_twos_comp(n_bytes)
    return decoded_bytes, i


def get_n_bytes_str(n: int, hex_ls: List[str], i: int) -> Tuple[str, int]:

    def _read_hex_ls(hex_ls: List[str]) -> str:
        chars = ''
        for h in hex_ls:
            chars += chr(int(h, 16))
        return chars

    n_bytes, i = get_n_bytes(n, hex_ls, i)
    return _read_hex_ls(n_bytes), i


def bin_data(wav_data: List[int], n_bins: int = 256, n_bits: int = 16) -> List[int]:
    """Takes wav data with real integer values and returns binned/simplified representation."""

    max_pressure: int = 2 ** (n_bits - 1)

    def standardize_pressure(pressure: int) -> float:
        return pressure / max_pressure

    def mus_law(standardized_pressure: float) -> float:
        return _mus_law(standardized_pressure, n_bins=n_bins)

    def get_bin(mu_value: float) -> int:
        return min(int(((mu_value + 1) / 2) * n_bins), n_bins - 1)

    standardized_wav_data = list(map(standardize_pressure, wav_data))
    mus_law_wav_data = list(map(mus_law, standardized_wav_data))
    binned_data = list(map(get_bin, mus_law_wav_data))
    return binned_data


def unbin_data(binned_data: List[int], n_bins: int = 256, n_bits: int = 16) -> List[int]:

    max_pressure: int = 2**(n_bits - 1)

    def unbin(binned: int) -> float:
        """Maps a binned value to [-1, 1]"""
        return ((binned / n_bins) * 2) - 1

    def reverse_mus_law(mu_transformed_pressure: float) -> float:
        return _reverse_mus_law(mu_transformed_pressure, n_bins=n_bins)

    def unstandardize_pressure(standardized_pressure: float) -> int:
        return int(standardized_pressure * max_pressure)

    unbinned_data = list(map(unbin, binned_data))
    reverse_mu_data = list(map(reverse_mus_law, unbinned_data))
    unstandardized_data = list(map(unstandardize_pressure, reverse_mu_data))
    return unstandardized_data


# Low-level helpers
def _twos_comp(val: int, bits: int) -> int:
    """compute the 2's complement of int value val"""
    if (val & (1 << (bits - 1))) != 0:  # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val


def _mus_law(raw_pressure: float, n_bins: int) -> float:
    mu = n_bins - 1
    mu_transformed_pressure = np.sign(raw_pressure) * np.log(1 + mu * np.abs(raw_pressure)) / np.log(mu + 1)
    return mu_transformed_pressure


def _reverse_mus_law(mu_transformed_pressure: float, n_bins: int) -> float:
    mu = n_bins - 1
    unmud_pressure = np.sign(mu_transformed_pressure) * (np.exp(np.abs(mu_transformed_pressure) * np.log(mu + 1)) - 1) / mu
    return unmud_pressure
