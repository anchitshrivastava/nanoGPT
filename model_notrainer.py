from tqdm import tqdm

from functools import partial

from datasets import load_dataset

import torch
import torch.nn.functional as F

from config import LR, DEVICE, END_KEY, RESPONSE_KEY_NL, INSTRUCTION_KEY, BATCH_SIZE, EPOCH, START_TOKEN

from transformers import AutoTokenizer, AutoModelForCausalLM


model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-neo-2.7B",
    low_cpu_mem_usage=True,
    cache_dir = "/home/sarabjot/storage/backup2/pathfactory_models"
)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
tokenizer.bos_token = START_TOKEN
tokenizer.pad_token = '<|pad|>'
tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})

model.to(DEVICE)

model.resize_token_embeddings(len(tokenizer))

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

dataset = load_dataset("tatsu-lab/alpaca")['train']

response_key_stripped = RESPONSE_KEY_NL.strip()
dataset = dataset.filter(lambda rec: not rec["text"].strip().endswith(response_key_stripped))

def _func(rec):
    rec["text"] = START_TOKEN + rec["text"] + f"\n\n{END_KEY}"
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
        RESPONSE_KEY_NL
    )['input_ids'][0]
    # seggregate input_ids and label
    input_ids = []
    attention_mask = []
    label = []
    for i in range(len(rec['input_ids'])):
        if rec['input_ids'][i] != tokenized_:
            input_ids.append(rec['input_ids'][i])
            attention_mask.append(rec['attention_mask'][i])
        else:
            label += rec['input_ids'][i:]
            break
    rec = {}
    rec['input_ids'] = torch.tensor(input_ids, dtype=int)
    rec['attention_mask'] = torch.tensor(attention_mask, dtype=int)
    rec['label'] = torch.tensor(label, dtype=int)
    
    return rec
    
dataset = dataset.map(_fun_tokenize_labels)

dataset_split = dataset.train_test_split(test_size=1000)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def get_random_batch(dataset, batch_size=BATCH_SIZE):
    random_indexes = torch.randint(0, len(dataset), batch_size)
    return dataset[random_indexes]

count = 0
step_at = 10
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