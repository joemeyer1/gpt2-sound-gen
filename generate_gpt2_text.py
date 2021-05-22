
from aitextgen import aitextgen

def generate_text(
        model_folder="trained_model",
        tokenizer_file="aitextgen.tokenizer.json",
        prompt="",
        write_raw_output_to_filename="raw_loaded_generated_unformatted_wav.txt",
        write_clean_output_to_filename="clean_loaded_generated_formatted_hex_str.txt"
):
    ai = aitextgen(model_folder=model_folder, tokenizer_file=tokenizer_file,)
    raw_generated_wav_txt = prompt
    while len(raw_generated_wav_txt) < 10000:
        raw_generated_wav_txt = raw_generated_wav_txt[:-16] + ai.generate(n=1, max_length=512, batch_size=100, prompt=raw_generated_wav_txt[-16:], return_as_list=True)[0]#.split('-')[0]
    if write_raw_output_to_filename:
        with open(write_raw_output_to_filename, 'w') as f:
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
    if write_clean_output_to_filename:
        with open(write_clean_output_to_filename, 'w') as f:
            f.write(hex_str)

def clean_model_output(model_output: str, bits_per_word=8) -> str:
    clean_output = ""
    model_output_words = model_output.split('-')
    for model_output_word in model_output_words:
        truncated_word = model_output_word[:bits_per_word]
        padding = "0" * (bits_per_word - len(truncated_word))
        clean_model_output_item = padding + truncated_word + '-'
        clean_output += clean_model_output_item
    return clean_output


if __name__ == "__main__":
    generate_text()
