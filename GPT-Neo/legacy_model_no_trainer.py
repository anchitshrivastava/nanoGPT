import os
from tqdm import tqdm

from copy import deepcopy

from functools import partial

import numpy as np

from datasets import load_dataset

import torch
import torch.nn.functional as F

from config import LR, DEVICE, END_KEY, RESPONSE_KEY_NL, INSTRUCTION_KEY, BATCH_SIZE, \
    EPOCH, START_TOKEN, INPUT_KEY, PROMPT_FORMAT, MAX_TOKENIZE_LEN

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-2.7B")
config.max_position_embeddings = 2048

model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-neo-2.7B",
    low_cpu_mem_usage=True,
    cache_dir = "/home/sarabjot/storage/backup2/pathfactory_models",
    config = config
)


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
tokenizer.bos_token = START_TOKEN
tokenizer.pad_token = END_KEY
tokenizer.eos_token = END_KEY
tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL, INPUT_KEY]})

model.to(DEVICE)

model.resize_token_embeddings(len(tokenizer))

def generate_response(instruction: str, *, model, tokenizer, 
                      do_sample: bool = True, max_new_tokens: int = 256, top_p: float = 0.92, top_k: int = 0, **kwargs) -> str:
    input_ids = tokenizer(PROMPT_FORMAT.format(instruction=instruction), return_tensors="pt").input_ids.to(DEVICE)

    # each of these is encoded to a single token
    response_key_token_id = tokenizer.encode("### Response:")[0]
    end_key_token_id = tokenizer.encode("### End")[0]

    gen_tokens = model.generate(input_ids, pad_token_id=tokenizer.pad_token_id, eos_token_id=end_key_token_id,
                                do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k, **kwargs)[0].cpu()

    # find where the response begins
    response_positions = np.where(gen_tokens == response_key_token_id)[0]

    if len(response_positions) >= 0:
        response_pos = response_positions[0]
        
        # find where the response ends
        end_pos = None
        end_positions = np.where(gen_tokens == end_key_token_id)[0]
        if len(end_positions) > 0:
            end_pos = end_positions[0]

        return tokenizer.decode(gen_tokens[response_pos + 1 : end_pos]).strip()

    return None

if os.path.exists("/home/sarabjot/PathFactory/GPT-j/saved_hf_model/saved_model.pth"):
    model.load_state_dict(torch.load("/home/sarabjot/PathFactory/GPT-j/saved_hf_model/saved_model.pth"))

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
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

_preprocessing_function = partial(preprocess_batch, max_length=MAX_TOKENIZE_LEN, tokenizer=tokenizer)

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
    
dataset = dataset.map(_fun_tokenize_labels)

print(dataset)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

def get_random_batch(dataset, batch_size=BATCH_SIZE):
    random_indexes = torch.randint(0, len(dataset), (batch_size, ))
    return dataset[random_indexes]

count = 0
step_at = 10
for ep in range(EPOCH):
    for _ in tqdm(range(len(dataset)// BATCH_SIZE)):
        row = get_random_batch(dataset, batch_size=BATCH_SIZE)
        optimizer.zero_grad()
        batch_in = {
            "input_ids": torch.tensor(row['input_ids']).to(DEVICE),
            "attention_mask": torch.tensor(row['attention_mask']).to(DEVICE)
        }
        out_ = model.forward(**batch_in,)
        out_ = out_.logits
        b, t, c = out_.shape
        out_ = out_.view(b*t, c)
        y_ = torch.tensor(row['labels']).flatten()
        loss = F.cross_entropy(out_, y_)
        print(loss)
        loss.backward()
        optimizer.step()
        
        count += 1
        
        if count % step_at == 0 or loss < 0.8:
            torch.save(model.state_dict(), "/home/sarabjot/PathFactory/GPT-j/saved_hf_model/saved_model.pth")
            print("*"*100, flush=True)
            print(tokenizer.decode(torch.argmax(out_, dim=1)))

torch.save(model.state_dict(), "/home/sarabjot/PathFactory/GPT-j/saved_hf_model/saved_model.pth")
print("yolo")