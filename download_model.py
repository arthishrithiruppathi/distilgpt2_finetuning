import os
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"
save_directory = "./model/pretrained_distilgpt2"

os.makedirs(save_directory, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print("DistilGPT2 downloaded successfully!")
print("Model saved at:", save_directory)