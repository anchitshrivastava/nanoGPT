import torch

LR = 3e-4
EPOCH = 3
# DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
BATCH_SIZE=1

END_KEY = "### End"
INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY_NL = f"### Response:\n"
START_TOKEN = "<|startoftext|>"