from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import os 
from tqdm import tqdm

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Explicitly set pad_token_id to eos_token_id
tokenizer.pad_token = tokenizer.eos_token

def generate_prompt(in_promt):
    """ Function to refine the initial set of key-words 
        
    input  : in_promt (str) an initial set of key-words
    output : refine_prompt (str) a set of refine descriptions  """

    # Encode input text
    input_ids = tokenizer.encode(in_promt, return_tensors="pt", padding=True, truncation=True)

    # Generate attention mask
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    # Generate text
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=200,  # Maximum length of generated text
        num_return_sequences=1,  # Number of generated sequences
        no_repeat_ngram_size=2,  # Avoid repetition
        top_k=50,  # Filter top-k tokens
        top_p=0.95,  # Nucleus sampling
        temperature=1.0,  # Sampling temperature
        do_sample=True, 
    )

    # Decode and print generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text 

def prompt_refinement(input_dir):
    # read the json input file
    with open(input_dir, "r") as file:
        data = json.load(file) 

    refine_prompts = {}
    load_bar = tqdm(data)
    for i, concept in enumerate(load_bar):
        key_words_wighted = data[concept]["keywords"]
        key_words_list = []
        for key_word_wighted in key_words_wighted:
            key_words_list += list(key_word_wighted.keys()) 
        key_words = ", ".join(key_words_list)

        in_promt = "create a coherent description of a scene with the words: " + key_words
        refine_prompts[concept] = generate_prompt(in_promt)[50:]
        load_bar.update(1)
    load_bar.close()
    return refine_prompts

def save_prompts(out_dirs, data):
    out_file = os.path.join(out_dirs, "prompts.json")
    assert os.path.isdir(out_dirs), f"directory: {out_dirs} not found"
    with open(out_file, "w") as file:
        json.dump(data, file, indent=4)
    print(f"data store in {out_file}")
    

if __name__ == "__main__":

    input_dir = "data/keywords.json"
    output_dir = "experiments/experiment_gpt2_sdxl"
    os.makedirs(output_dir, exist_ok=True)

    data = prompt_refinement(input_dir)
    save_prompts(output_dir, data)

    