import torch

from config import DEVICE, CONTEXT_SIZE, BATCH_SIZE


train_data = [{"instruction": "sdkjchkdjchsdkjchsdkjch", "input": "fdlkvhnldfknvldfkvn", "output": "fljkvndflkvndflvkn"}]
val_data = []

def get_train_batch():
    # generate a small batch of data of inputs x and targets y
    data = train_data
    while True:
        ix = torch.randint(len(data) - CONTEXT_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i:i + CONTEXT_SIZE] for i in ix])
        y = torch.stack([data[i + 1:i + CONTEXT_SIZE + 1] for i in ix])
        x, y = x.to(DEVICE), y.to(DEVICE)
        yield x, y

def get_val_batch():
    # generate a small batch of data of inputs x and targets y
    data = val_data
    while True:
        ix = torch.randint(len(data) - CONTEXT_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i:i + CONTEXT_SIZE] for i in ix])
        y = torch.stack([data[i + 1:i + CONTEXT_SIZE + 1] for i in ix])
        x, y = x.to(DEVICE), y.to(DEVICE)
        yield x, y
