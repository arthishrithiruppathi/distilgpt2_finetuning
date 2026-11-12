import math
import re
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForCausalLM, AutoTokenizer,DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments, set_seed)

MODEL_PATH = "./model/pretrained_distilgpt2"
SAVE_PATH = "./model/fine_tuned_quote_generator"

set_seed(42)

# 1. Load the quotes and clean them
with open("dataset/quotes.txt", "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

quotes = []
for line in lines:
    line = re.sub(r"^\s*\d+[\.\)]\s*", "", line) 
    line = line.strip().strip('"').strip()          
    if line:
        quotes.append(line)

quotes = list(dict.fromkeys(quotes))               
print("Total unique quotes:", len(quotes))

# 2. 90% training, 10% validation
train_quotes, val_quotes = train_test_split(quotes, test_size=0.1, random_state=42)
print("Training / validation:", len(train_quotes), len(val_quotes))

# 3. Tokenize 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

def to_dataset(texts):
    ds = Dataset.from_dict({"text": [t + tokenizer.eos_token for t in texts]})
    return ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=64),batched=True, remove_columns=["text"])

train_dataset = to_dataset(train_quotes)
val_dataset = to_dataset(val_quotes)

# 4. Load the pretrained text generation model
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# 5. The collator builds the labels
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(output_dir="./results",eval_strategy="epoch",save_strategy="epoch",
    save_total_limit=1,learning_rate=5e-5,per_device_train_batch_size=8,per_device_eval_batch_size=8,
    num_train_epochs=3,weight_decay=0.01,logging_steps=10,load_best_model_at_end=True,
    metric_for_best_model="eval_loss",report_to="none",)

trainer = Trainer(model=model, args=args, train_dataset=train_dataset,eval_dataset=val_dataset, 
                  processing_class=tokenizer,data_collator=collator)

# 6. Score BEFORE fine-tuning 
before = trainer.evaluate()
print("\nPERPLEXITY BEFORE FINE-TUNING:", round(math.exp(before["eval_loss"]), 2))

# 7. Fine-tune
trainer.train()

# 8. Score AFTER fine-tuning
after = trainer.evaluate()
print("\nPERPLEXITY AFTER FINE-TUNING:", round(math.exp(after["eval_loss"]), 2))

# 9. Save
trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print("Fine-tuned model saved.")