from tqdm import tqdm

from functools import partial

from datasets import load_dataset

import torch
import torch.nn.functional as F

from config import LR, DEVICE, END_KEY, RESPONSE_KEY_NL, INSTRUCTION_KEY, BATCH_SIZE, EPOCH

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer


model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-neo-2.7B",
    low_cpu_mem_usage=True,
    cache_dir = "/home/sarabjot/storage/backup2/pathfactory_models"
)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})

model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

dataset = load_dataset("tatsu-lab/alpaca")['train']

response_key_stripped = RESPONSE_KEY_NL.strip()
dataset = dataset.filter(lambda rec: not rec["text"].strip().endswith(response_key_stripped))

def _func(rec):
    rec["text"] += f"\n\n{END_KEY}"
    rec['label'] = RESPONSE_KEY_NL + rec['text'].split(RESPONSE_KEY_NL)[1]
    rec['text'] = rec['text'].split(RESPONSE_KEY_NL)[0]
    return rec

dataset = dataset.map(_func)

def preprocess_batch(batch, tokenizer, max_length: int) -> dict:
    
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

_preprocessing_function = partial(preprocess_batch, max_length=128, tokenizer=tokenizer)
dataset = dataset.map(
    _preprocessing_function,
    batched=True,
    remove_columns=["instruction", "input", "output", "text"],
)

def _fun_tokenize_labels(rec):
    tokenized_ = tokenizer(
        rec["label"],
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    rec['label'] = tokenized_['input_ids']
    return rec
    
dataset = dataset.map(_fun_tokenize_labels)

dataset_split = dataset.train_test_split(test_size=1000)

optimizer = torch.optim.SGD(model.parameters(), lr=LR)

count = 0
step_at = 1000
with torch.cuda.amp.autocast():
    for ep in range(EPOCH):
        for row in tqdm(dataset_split['train']):
            optimizer.zero_grad()
            if len(row["input_ids"]) <= 1:
                continue
            batch_in = {
                "input_ids": torch.tensor(row['input_ids']),
                "attention_mask": torch.tensor(row['attention_mask'])
            }
            out = model.forward(**batch_in,)
            out = out.logits
            y_ = torch.tensor(row['label']).flatten()
            loss = F.cross_entropy(out, y_)
            print(loss)
            loss.backward()
            optimizer.step()
            
            count += 1
            
            if count % step_at == 0 or loss < 0.8:
                print("*"*100)
                print(tokenizer.decode(torch.argmax(out, dim=1)))
                print("-"*100)
                print(tokenizer.decode(torch.tensor(row['input_ids'])))
                print("-"*100)
                print(tokenizer.decode(torch.tensor(row['label']).flatten()))
            
print("yolo")