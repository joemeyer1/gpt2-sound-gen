#!usr/bin/env python3
# Copyright (c) Joe Meyer (2020). All rights reserved.

from typing import List

import numpy as np


def file_to_hex_ls(filename):
    hex_str = file_to_hex_str(filename)
    hex_ls = hex_str_to_ls(hex_str)
    return hex_ls


def get_n_bytes(n, hex_ls, i):
    # get next n bytes after index in in hex_ls
    n_bytes = hex_ls[i: i + n]
    i = i + n
    return n_bytes, i


def get_n_bytes_int(n, hex_ls, i):
    n_bytes, i = get_n_bytes(n, hex_ls, i)
    return _decode_endian_and_twos_comp(n_bytes), i


def get_n_bytes_str(n, hex_ls, i):
    n_bytes, i = get_n_bytes(n, hex_ls, i)
    return read_hex_ls(n_bytes), i


# helpers for file_to_hex_ls()
def file_to_hex_str(filename):
    with open(filename, 'rb') as f:
        words = f.read()
    return words.hex()


def hex_str_to_ls(hex_str):
    i = 0
    hex_ls = []
    while i < len(hex_str):
        hex_ls.append(hex_str[i:i+2])
        i += 2
    return hex_ls


# helper for get_n_bytes_int()
def _decode_endian_and_twos_comp(hex_ls):
    # little endian (ones place at start)
    # e.g. ['10', '01'] = x0110 = 256 + 16 = 272
    hex_str = ''
    for h in hex_ls:
        hex_str = h + hex_str
    return twos_comp(int(hex_str, 16), len(hex_str) * 4)


def twos_comp(val, bits):
    """compute the 2's complement of int value val"""
    if (val & (1 << (bits - 1))) != 0:  # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val


# helper for get_n_bytes_str()

def read_hex_ls(hex_str):
    chars = ''
    for h in hex_str:
        chars += chr(int(h, 16))
    return chars


def bin_data(wav_data: List[int], n_bins: int = 256, n_bits: int = 16) -> List[int]:
    """Takes wav data with real integer values and returns binned/simplified representation."""

    max_pressure: int = 2**(n_bits - 1)

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

    def _reverse_mus_law(mu_transformed_pressure):
        mu = n_bins - 1
        unmud_pressure = np.sign(mu_transformed_pressure) * (np.exp(np.abs(mu_transformed_pressure) * np.log(mu + 1)) - 1) / mu
        return unmud_pressure

    def unstandardize_pressure(standardized_pressure: float) -> int:
        return int(standardized_pressure * max_pressure)

    unbinned_data = list(map(unbin, binned_data))
    reverse_mu_data = list(map(_reverse_mus_law, unbinned_data))
    unstandardized_data = list(map(unstandardize_pressure, reverse_mu_data))
    return unstandardized_data


def _mus_law(pressure: float, n_bins: int = 256) -> int:
    mu = n_bins - 1
    quantized_pressure = np.sign(pressure) * np.log(1 + mu * np.abs(pressure)) / np.log(mu + 1)
    return quantized_pressure

# def _reverse_mus_law(quantized_pressure, n_bins: int = 256, n_bits: int = 16) -> float:
#
#     max_pressure: int = 16**n_bits - 1
#
#     def reverse_standardize_pressure(quantized_pressure: float) -> float:
#         return (quantized_pressure * (max_pressure / 2)) + (max_pressure / 2)
#
#     mu = n_bins - 1
#     scaled_pressure = np.sign(quantized_pressure) * (np.exp(np.abs(quantized_pressure) * np.log(mu + 1)) - 1) / mu
#     raw_pressure = reverse_standardize_pressure(scaled_pressure)
#     return raw_pressure
