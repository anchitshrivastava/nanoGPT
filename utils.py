import torch

from config import CONTEXT_SIZE, BATCH_SIZE, DEVICE


def return_vocab(data: str = None) -> list:
    vocab = list(set(data))
    vocab.sort()
    return vocab


def encode_data(data: str = None) -> list:
    vocab = return_vocab(data=data)
    encodings = {ch: i for i, ch in enumerate(vocab)}
    return [encodings[i] for i in data]


def decode_data(data: list = None, vocab: list = None) -> str:
    vocab = return_vocab(data=vocab)
    decodings = {i: ch for i, ch in enumerate(vocab)}
    return "".join([decodings[i] for i in data])


def data_generator(data: torch.Tensor = None):
    while True:
        start_indexes = torch.randint(low=0, high=len(data) - CONTEXT_SIZE - 1, size=(BATCH_SIZE,))

        X = torch.stack([data[start_index: start_index + CONTEXT_SIZE] for start_index in start_indexes])
        y = torch.stack([data[start_index + 1: start_index + CONTEXT_SIZE + 1] for start_index in start_indexes])

        yield X.to(DEVICE), y.to(DEVICE)


if __name__ == '__main__':
    a = data_generator(data=torch.tensor(list(range(100)), dtype=torch.long))
    for i in range(100):
        print(next(a))
