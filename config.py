import torch

LR = 3e-4
EPOCH = 300
# DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
BATCH_SIZE=8

END_KEY = "### End"
INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY_NL = f"### Response:\n"
INPUT_KEY = "### Input:"
START_TOKEN = "<|startoftext|>"