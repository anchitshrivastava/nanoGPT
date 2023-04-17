from functools import partial

from datasets import load_dataset

import torch
import torch.nn.functional as F

from config import LR, DEVICE, END_KEY, RESPONSE_KEY_NL, INSTRUCTION_KEY, BATCH_SIZE, EPOCH, START_TOKEN, INPUT_KEY

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments


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
    rec["text"] += f"\n\n{END_KEY}"
    rec['label'] = RESPONSE_KEY_NL + rec['text'].split(RESPONSE_KEY_NL)[1]
    rec['text'] = rec['text'].split(RESPONSE_KEY_NL)[0]
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
        rec["label"],
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    rec['label'] = tokenized_['input_ids'][0]
    return rec
    
dataset = dataset.map(_fun_tokenize_labels)

dataset_split = dataset.train_test_split(test_size=1000)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def get_random_batch(dataset, batch_size=BATCH_SIZE):
    random_indexes = torch.randint(0, len(dataset), (batch_size, ))
    return dataset[random_indexes]


training_args = TrainingArguments(
        output_dir="./saved_hf_model",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        fp16=False,
        bf16=True,
        learning_rate=LR,
        num_train_epochs=EPOCH,
        evaluation_strategy="steps",
        eval_steps=100,
        remove_unused_columns=False,
        logging_dir=f"./saved_hf_model/runs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,
        load_best_model_at_end=True,
    )

optimizer=transformers.AdamW(model.parameters(),
lr=0.00025)

trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset_split["train"],
        eval_dataset=dataset_split["test"],
        optimizers=(optimizer, )
    )

trainer.train()

print("yolo")