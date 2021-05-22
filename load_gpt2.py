
from aitextgen import aitextgen

def load_gpt2(model_folderr="trained_model", tokenizer_file="aitextgen.tokenizer.json", prompt=""):
    ai = aitextgen(model_folder=model_folderr, tokenizer_file=tokenizer_file,)
    ai.generate(n=1, batch_size=100, max_length=1000, prompt=prompt, return_as_list=False)

if __name__ == "__main__":
    load_gpt2()
