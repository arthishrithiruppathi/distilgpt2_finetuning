# DistilGPT2 Fine-Tuning: Motivational Quote Generator

Fine-tuning `distilgpt2` (Hugging Face Transformers) on a dataset of motivational quotes so the model generates new motivational text from a short prompt.

## Overview

| Item | Details |
|---|---|
| Base model | `distilgpt2` |
| Task | Text generation (causal language modeling) |
| Framework | Hugging Face Transformers, PyTorch |
| Dataset | Motivational quotes (text file / CSV) |
| Environment | Python virtual environment (venv) |

## Project Structure

```
distilgpt2_quotes/
├── data/
│   └── quotes.csv
├── prepare_data.py
├── train.py                 
├── generate.py            
├── requirements.txt      
└── README.md
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate       
```

2. Install dependencies:

```bash
pip install transformers datasets torch accelerate pandas
```

## How to Run

```bash
python prepare_data.py   
python train.py          
python generate.py        
```

## Workflow

1. **Prepare data**: Load the quotes, remove duplicates and empty lines, and keep one quote per row.
2. **Tokenize**: Convert each quote into token IDs using the GPT-2 tokenizer. GPT-2 has no padding token, so the end-of-sequence token is used for padding.
3. **Train**: Fine-tune `distilgpt2` with a causal language modeling objective, where the model learns to predict the next word in a quote.
4. **Generate**: Give the model a starting phrase and let it complete the quote.

## What Is Fine-Tuning?

`distilgpt2` is a smaller, faster version of GPT-2 that is already pre-trained on a large amount of general English text. Fine-tuning continues training on a small, specific dataset (motivational quotes) so the model picks up the tone and style of that data. This needs far less data and compute than training from scratch.

## Why distilgpt2?

- Much smaller than GPT-2 and larger models, so it trains quickly on a normal laptop.
- Easy to run and explain, which makes it a good model for learning how fine-tuning works.

## Example Usage

```python
from transformers import pipeline

generator = pipeline("text-generation", model="./distilgpt2_quotes_model")

result = generator(
    "Success comes to those who",
    max_length=40,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.8,
)
print(result[0]["generated_text"])
```

## Results

| Prompt | Generated output |
|---|---|
| _add your prompt_ | _add the generated quote_ |

## Limitations

- distilgpt2 is a small model, so some generated quotes may be repetitive or not fully make sense.
- Output quality depends on the size and quality of the quotes dataset.
- Generated text is random by nature, so results change on every run unless a seed is set.

## Author

**ArthiShri T.**
LinkedIn: [linkedin.com/in/arthishri-thiruppathi](https://linkedin.com/in/arthishri-thiruppathi)
