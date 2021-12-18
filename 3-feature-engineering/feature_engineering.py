import numpy as np
import pandas as pd
import torch
import transformers as ppb

def build_tensors_fn(descriptions, tokenizer, max_len=128):
    # tokenization. 
    sentences = descriptions['long_description'].apply((lambda s: ' '.join(s.split()[:max_len])))
    tokenized = sentences.apply((lambda s: tokenizer.encode(s, add_special_tokens=True)))

    # padding
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

    # masking 
    attention_mask = np.where(padded != 0, 1, 0)

    # model#1
    input_ids = torch.tensor(padded)
    attention_mask =  torch.tensor(attention_mask)

    return (input_ids, attention_mask)

def extract_features_fn(dataset, model, tokenizer):
    input_ids, attention_mask = build_tensors_fn(dataset, tokenizer)
 
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:,0,:].numpy()
    labels   = dataset['severity_code']

    return (features, labels)