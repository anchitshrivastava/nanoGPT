import torch

LR = 1e-3
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
CONTEXT_SIZE = 10
BATCH_SIZE=32

END_KEY = "### End"
INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY_NL = f"### Response:\n"