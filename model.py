from tqdm import tqdm

from datasets import from_generator

import torch
import torch.functional as F

from utils import get_train_batch, get_val_batch

from config import LR, DEVICE

from transformers import AutoTokenizer, AutoModelForCausalLM


model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B",
    revision="float16",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    cache_dir = "/home/sarabjot/storage/backup2/pathfactory_models"
)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

dataset = [{"text": "hello how are you?"}]
dataset = from_generator(get_train_batch)

with torch.cuda.amp.autocast():
    for row in tqdm(dataset):
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
        optimizer.zero_grad()
