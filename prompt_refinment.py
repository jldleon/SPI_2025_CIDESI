from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Input prompt
input_text = """create a coherent description of a scene with the words: sea, woman, fish 
exaples:
1. a mermaid swimming in the sea, """

# Encode input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate text
output = model.generate(
    input_ids,
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
print(generated_text)
