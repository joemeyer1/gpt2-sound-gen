#!usr/bin/env python3
# Copyright (c) Joe Meyer (2020). All rights reserved.

def file_to_hex_ls(filename):
    hex_str = file_to_hex_str(filename)
    hex_ls = hex_str_to_ls(hex_str)
    return hex_ls

def get_n_bytes(n, hex_ls, i):
    # get next n bytes after index in in hex_ls
    n_bytes = hex_ls[i:i+n]
    i=i+n
    return n_bytes, i

def get_n_bytes_int(n, hex_ls, i):
    n_bytes, i = get_n_bytes(n, hex_ls, i)
    return get_number(n_bytes), i

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
        i+=2
    return hex_ls

# helper for get_n_bytes_int()
def get_number(hex_ls):
    # little endian (ones place at start)
    # e.g. ['10', '01'] = x0110 = 256 + 16 = 272
    hex_str = ''
    for h in hex_ls:
        hex_str = h + hex_str
    return int(hex_str, 16)

# helper for get_n_bytes_str()

def read_hex_ls(hex_str):
    chars = ''
    for h in hex_str:
        chars += chr(int(h, 16))
    return chars
