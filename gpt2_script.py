import gpt_2_simple as gpt2
import os
from make_wav_str_file import convert_wav_to_text_file


def run_gpt2_script(
    model_name: str = "124M",
    steps: int = 10,
    wav_str_filename: str = "sound.txt"
) -> None:

    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/

    convert_wav_to_text_file(
        in_wav_dir_name="sound_files",
        out_text_filename=wav_str_filename,
        n_max_files=1,
    )

    print(f"file_name: {wav_str_filename}")
    print(f"steps: {steps}")
    print(f"model_name: {model_name}")

    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess,
                  wav_str_filename,
                  model_name=model_name,
                  steps=steps)   # steps is max number of training steps

    gpt2.generate(sess)


if __name__ == "__main__":
    run_gpt2_script()
