import gpt_2_simple as gpt2
import os
from make_wav_str_file import write_file_from_wav_str

model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/



# # in_file_names = [fname for fname in os.listdir("sound_files") if fname]
# in_file_name = "sound_files/violin_G4_phrase_forte_harmonic-glissando"
# out_file_name = "sound.txt"
# if not os.path.isfile(file_name):
#   data = "sound_files/violin_G4_phrase_forte_harmonic-glissando.wav"
#   with open(in_file_name, 'rb') as f:
#       text = f.read()
#
#   with open(out_file_name, 'wb') as f:
#       f.write(data)

wav_str_filename = "sound.txt"
write_file_from_wav_str(
    read_wav_from_dir="sound_files",
    write_wav_str_to_file=wav_str_filename,
    n_max_files=1,
)

steps = 10

print(f"file_name: {wav_str_filename}")
print(f"steps: {steps}")
print(f"model_name: {model_name}")

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              wav_str_filename,
              model_name=model_name,
              steps=steps)   # steps is max number of training steps

gpt2.generate(sess)
