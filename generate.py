from transformers import (AutoModelForCausalLM, AutoTokenizer, set_seed)
BASE_PATH = "./model/pretrained_distilgpt2"
TUNED_PATH = "./model/fine_tuned_quote_generator"
prompts = ["Success comes from", "Your dreams are worth", "The secret of happiness is"]

def generate(model_path, prompt):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)
for prompt in prompts:
    print("\nPROMPT:", prompt)
    set_seed(42)
    print("  BEFORE (original DistilGPT2):", generate(BASE_PATH, prompt))
    set_seed(42)
    print("  AFTER  (fine-tuned)         :", generate(TUNED_PATH, prompt))