import torch

LR = 3e-4
EPOCH = 300
# DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
BATCH_SIZE=4
MAX_TOKENIZE_LEN=1024

END_KEY = "### End"
INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY_NL = f"### Response:\n"
INPUT_KEY = "### Input:"
START_TOKEN = "<|startoftext|>"

PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""