import torch

CONTEXT_SIZE = 8
N_EMBEDS = 32
HEAD_SIZE = 32
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 1e-3
