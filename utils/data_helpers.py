from copy import deepcopy

from config import START_TOKEN, END_KEY, RESPONSE_KEY_NL

def adding_starting_ending_token(rec):
    rec["text"] = START_TOKEN + rec["text"] + f"\n\n{END_KEY}"
    return rec

def tokenize_batch(batch, tokenizer, max_length: int) -> dict:
    
    return tokenizer(
        batch["text"],
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

def creating_masking_labels_inputs(rec, tokenizer):
    
    tokenized_ = tokenizer(
        RESPONSE_KEY_NL
    )['input_ids'][0]
    
    labels = deepcopy(rec['input_ids'])
    
    for i in range(len(labels)):
        if labels[i]!=tokenized_:
            labels[i] = -100
        else:
            labels[i] = -100
            break
    
    tokenized_ = tokenizer(
        END_KEY
    )['input_ids'][0]

    for i in range(len(labels)-1, 0, -1):
        if labels[i]==tokenized_:
            labels[i] = -100
        else:
            break
    
    rec['labels'] = labels
    
    return rec
