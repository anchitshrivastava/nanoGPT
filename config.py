import torch

CONTEXT_SIZE = 256
N_EMBEDS = 384
NUM_HEADS = 6
HEAD_SIZE = N_EMBEDS // NUM_HEADS
NUM_LAYERS = 6
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 3e-4
DROPOUT = 0.2
