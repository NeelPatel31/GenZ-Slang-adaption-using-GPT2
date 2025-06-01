# Gen-Z Slang Adaptation using GPT2

This project implements a two-phase approach to adapt GPT2 for understanding and generating Gen-Z slang:

1. Fine-tuning GPT2 on dictionary word-definition pairs
2. Expanding vocabulary with Gen-Z slang terms

## Project Overview

The model is trained to understand the relationship between words and their definitions, then expanded to include modern Gen-Z slang vocabulary. This approach allows the model to better understand and generate contemporary internet language and slang terms.

## Architecture

- Base Model: GPT2
- Training Phases:
  - Phase 1: Dictionary fine-tuning
  - Phase 2: Vocabulary expansion with Gen-Z slang

## Dataset

The project uses two main datasets:
- Dictionary word-definition pairs
- Gen-Z slang dataset with terms, definitions, and contextual examples

## Training Process

### Phase 1: Dictionary Fine-tuning
- Fine-tunes GPT2 on word-definition pairs
- Uses MSE Loss on hidden states
- Implements gradient accumulation for stable training
- Learning rate: 3e-5
- Batch size: 8 with 32 gradient accumulation steps

### Phase 2: Vocabulary Expansion
- Expands model vocabulary with Gen-Z slang terms
- Derives embeddings from contextual definitions
- Preserves original model knowledge while adding new terms

## Evaluation

The model is evaluated on:
- Word prediction accuracy
- Top-k accuracy metrics (k = 1, 5, 10, 25)
- Average rank of correct predictions

## Requirements

- Python 3.11+
- PyTorch
- Transformers
- Pandas
- NumPy
- Matplotlib
- CUDA-capable GPU (recommended)

## Usage

1. Training:
```python
# Run the notebook gpt2-genz-adaptation.ipynb
```

2. Inference:
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('path/to/model')
tokenizer = GPT2Tokenizer.from_pretrained('path/to/model')

# Generate text
text = "This is so"
inputs = tokenizer(text, return_tensors='pt')
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

## Model Outputs

The project produces two main model variants:
1. GPT2-finetuned-dictionary: Base model fine-tuned on dictionary data
   - Available at: [Kaggle Model Hub](https://www.kaggle.com/models/neelpatel31/gpt2_definition_finetuned)
2. GPT2-GenZ-slang: Final model with expanded Gen-Z vocabulary
   - Available at: [Kaggle Model Hub](https://www.kaggle.com/models/neelpatel31/gpt2_genz_slang)