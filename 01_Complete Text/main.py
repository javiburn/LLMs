import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate(text, max_length=100):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    input_ids = tokenizer.encode(text, return_tensors='pt')
    attention_mask = torch.ones_like(input_ids)
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    text = ""
    text = input("Enter a sentence: ")
    generated_text = generate(text, max_length=100)
    print(generated_text)

if __name__ == "__main__":
    main()