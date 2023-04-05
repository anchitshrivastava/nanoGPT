from tqdm import tqdm

from datasets import load_dataset

import torch
import torch.nn.functional as F

from utils import get_train_batch, get_val_batch

from config import LR, DEVICE, END_KEY, RESPONSE_KEY_NL, INSTRUCTION_KEY

from transformers import AutoTokenizer, GPTJForCausalLM


model = GPTJForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B",
    revision="float16",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    cache_dir = "/home/sarabjot/storage/backup2/pathfactory_models"
)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})

model.to(DEVICE)

# prompt = "The Belgian national football team "
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

# generated_ids = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=200)

# generated_text = tokenizer.decode(generated_ids[0])
# print(generated_text)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

dataset = load_dataset("tatsu-lab/alpaca", streaming=True)['train']

with torch.cuda.amp.autocast():
    for row in tqdm(dataset):
        optimizer.zero_grad()
        if len(row["text"]) <= 1:
            continue
        
        
        prompt_format = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

{INSTRUCTION_KEY}
{row['instruction']}

{RESPONSE_KEY_NL}
"""
        
        
        batch_in = tokenizer(prompt_format, truncation=True, max_length=128, return_tensors='pt', padding="max_length")
        batch_out = tokenizer(row['text'], truncation=True, max_length=128, return_tensors='pt', padding="max_length")
        batch_in = {k: v.to(DEVICE) for k, v in batch_in.items()}
        batch_out = {k: v.to(DEVICE) for k, v in batch_out.items()}
        out = model.forward(**batch_in,)
        loss = F.cross_entropy(out.logits[:, :-1, :].flatten(0, -2), batch_out['input_ids'][:, 1:].flatten(),
                               reduction='mean')
        print(loss)
        loss.backward()
        optimizer.step()
