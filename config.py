import torch

CONTEXT_SIZE = 64
N_EMBEDS = 384
NUM_HEADS = 6
HEAD_SIZE = N_EMBEDS // NUM_HEADS
NUM_LAYERS = 6
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 1e-3
DROPOUT = 0.2
