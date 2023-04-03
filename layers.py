import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BATCH_SIZE, CONTEXT_SIZE, DEVICE, N_EMBEDS, HEAD_SIZE


class SelfAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer('tril_matrix', torch.tril(torch.ones(CONTEXT_SIZE, CONTEXT_SIZE)))  # B, T, C
        self.key = nn.Linear(N_EMBEDS, HEAD_SIZE, bias=False)
        self.query = nn.Linear(N_EMBEDS, HEAD_SIZE, bias=False)
        self.value = nn.Linear(N_EMBEDS, HEAD_SIZE, bias=False)

    def forward(self, x):
        _, T, C = x.shape
        key = self.key(x)
        query = self.query(x)

        initialized_weights = query @ key.transpose(-2, -1) * HEAD_SIZE**-0.5
        initialized_weights = initialized_weights.masked_fill(self.tril_matrix[:T, :T] == 0, float("-inf"))
        # Basically we have to take the avg of context vector
        initialized_weights = F.softmax(initialized_weights, dim=-1)

        value = self.value(x)
        changed_x = initialized_weights @ value

        return changed_x