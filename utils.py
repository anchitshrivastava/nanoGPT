import numpy as np

import torch

from config import CONTEXT_SIZE, BATCH_SIZE, DEVICE


def return_vocab() -> list:
    vocab = list(set(data))
    vocab.sort()
    return vocab


def encode_data(data: str = None) -> list:
    vocab = return_vocab()
    encodings = {ch: i for i, ch in enumerate(vocab)}
    return [encodings[i] for i in data]


def decode_data(data: list = None, vocab: list = None) -> str:
    vocab = return_vocab()
    decodings = {i: ch for i, ch in enumerate(vocab)}
    return "".join([decodings[i] for i in data])


def encode_train_test():
    # splitting in train and test splits

    total_len = len(data)
    train_split = 0.90

    train = data[:int(train_split * total_len)]
    test = data[int(train_split * total_len):]

    print(f"""Len of Train = {len(train)} \n Len of Test = {len(test)}""")

    encoded_train_data = encode_data(data=train)
    encoded_test_data = encode_data(data=test)

    encoded_train_data = torch.tensor(encoded_train_data, dtype=torch.long)
    encoded_test_data = torch.tensor(encoded_test_data, dtype=torch.long)

    return encoded_train_data, encoded_test_data


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - CONTEXT_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i + CONTEXT_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + CONTEXT_SIZE + 1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


with open("./dataset/shakespeare.txt", "r") as file:
    data = file.read()
with open("./dataset/shakespeare.txt", "r") as file:
    data_lines = file.readlines()
    print("Average length of line", np.mean([len(i) for i in data_lines]))
    print("Max length of line", np.max([len(i) for i in data_lines]))

print(data[:10])

train_data, val_data = encode_train_test()

if __name__ == '__main__':
    a = get_batch(split="train")
    X, y = a
    print(X, y)
