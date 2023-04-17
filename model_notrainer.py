from tqdm import tqdm
from copy import deepcopy

from functools import partial

from datasets import load_dataset

import torch
import torch.nn.functional as F

from config import LR, DEVICE, END_KEY, RESPONSE_KEY_NL, INSTRUCTION_KEY, BATCH_SIZE, EPOCH, START_TOKEN, INPUT_KEY

from transformers import AutoTokenizer, AutoModelForCausalLM


model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-neo-2.7B",
    low_cpu_mem_usage=True,
    cache_dir = "/home/sarabjot/storage/backup2/pathfactory_models"
)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
tokenizer.bos_token = START_TOKEN
tokenizer.pad_token = END_KEY
tokenizer.eos_token = END_KEY
tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL, INPUT_KEY]})

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
    
    labels = deepcopy(rec['input_ids'])
    
    for i in range(len(labels)):
        if labels[i]!=tokenized_:
            labels[i] = -100
        else:
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
    
dataset = dataset.map(_fun_tokenize_labels)

dataset_split = dataset.train_test_split(test_size=1000)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

def get_random_batch(dataset, batch_size=BATCH_SIZE):
    random_indexes = torch.randint(0, len(dataset), (batch_size, ))
    return dataset[random_indexes]

count = 0
step_at = 10
with torch.cuda.amp.autocast():
    for ep in range(EPOCH):
        for _ in tqdm(range(len(dataset_split['train'])//BATCH_SIZE)):
            row = get_random_batch(dataset_split['train'], batch_size=BATCH_SIZE)
            optimizer.zero_grad()
            batch_in = {
                "input_ids": torch.tensor(row['input_ids']),
                "attention_mask": torch.tensor(row['attention_mask'])
            }
            out = model.forward(**batch_in,)
            out = out.logits
            b, t, c = out.shape
            out = out.view(b*t, c)
            y_ = torch.tensor(row['labels']).flatten()
            loss = F.cross_entropy(out, y_)
            print(loss)
            loss.backward()
            optimizer.step()
            
            count += 1
            
            if count % step_at == 0 or loss < 0.8:
                print("*"*100)
                print(tokenizer.decode(torch.argmax(out, dim=1)))

torch.save(model, "/home/sarabjot/PathFactory/GPT-j/saved_hf_model/saved_model")
print("yolo")