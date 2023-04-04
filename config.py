import torch

LR = 1e-3
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
CONTEXT_SIZE = 10
BATCH_SIZE=32