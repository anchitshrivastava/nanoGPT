import torch

CONTEXT_SIZE = 32
N_EMBEDS = 8
HEAD_SIZE = 16
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
LR = 1e-3
