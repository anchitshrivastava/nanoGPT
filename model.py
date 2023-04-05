from tqdm import tqdm

from datasets import load_dataset

import torch
import torch.functional as F

from utils import get_train_batch, get_val_batch

from config import LR, DEVICE

from transformers import AutoTokenizer, GPTJForCausalLM


model = GPTJForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B",
    revision="float16",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    cache_dir = "/home/sarabjot/storage/backup2/pathfactory_models"
)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

model.to(DEVICE)

prompt = "The Belgian national football team "
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

generated_ids = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=200)

generated_text = tokenizer.decode(generated_ids[0])
print(generated_text)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

dataset = load_dataset("tatsu-lab/alpaca", streaming=True)['train']

with torch.cuda.amp.autocast():
    for row in tqdm(dataset):
        optimizer.zero_grad()
        if len(row["text"]) <= 1:
            continue
        batch = tokenizer(row["text"], truncation=True, max_length=128, return_tensors='pt')
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model.forward(**batch,)
        loss = F.cross_entropy(out.logits[:, :-1, :].flatten(0, -2), batch['input_ids'][:, 1:].flatten(),
                               reduction='mean')
        print(loss)
        loss.backward()
        optimizer.step()
