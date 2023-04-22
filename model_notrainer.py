import os
from tqdm import tqdm
from copy import deepcopy

from functools import partial

import numpy as np

from datasets import load_dataset

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import LR, DEVICE, END_KEY, RESPONSE_KEY_NL, INSTRUCTION_KEY, BATCH_SIZE, EPOCH, START_TOKEN, INPUT_KEY

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling


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
        truncation=True,
    )

_preprocessing_function = partial(preprocess_batch, max_length=1024, tokenizer=tokenizer)

dataset = dataset.map(
    _preprocessing_function,
    batched=True,
    remove_columns=["instruction", "input", "output", "text"],
)

dataset_split = dataset.train_test_split(test_size=1000)

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)

        # The prompt ends with the response key plus a newline.  We encode this and then try to find it in the
        # sequence of tokens.  This should just be a single token.
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)

        labels = batch["labels"].clone()

        for i in range(len(examples)):

            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch

data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
    )

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

train_dataloader = DataLoader(dataset_split['train'], batch_size=BATCH_SIZE, collate_fn=data_collator)

count = 0
step_at = 10
with torch.cuda.amp.autocast():
    for ep in range(EPOCH):
        for row in tqdm(train_dataloader):
            optimizer.zero_grad()
            outputs = model(input_ids=row['input_ids'], attention_mask=row['attention_mask'], labels=row['labels'])
            loss = outputs.loss
            print(loss)
            loss.backward()
            optimizer.step()
            
            count += 1
            
            if count % step_at == 0 or loss < 0.8:
                torch.save(model.state_dict(), "/home/sarabjot/PathFactory/GPT-j/saved_hf_model/saved_model.pth")
                print("*"*100)
                out = outputs.logits
                print(tokenizer.decode(torch.argmax(out[0], dim=1)))

torch.save(model.state_dict(), "/home/sarabjot/PathFactory/GPT-j/saved_hf_model/saved_model.pth")
print("yolo")