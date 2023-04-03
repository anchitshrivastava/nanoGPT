import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONTEXT_SIZE, N_EMBEDS, HEAD_SIZE, NUM_HEADS, DROPOUT


class SelfAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer('tril_matrix', torch.tril(torch.ones(CONTEXT_SIZE, CONTEXT_SIZE)))  # B, T, C
        self.key = nn.Linear(N_EMBEDS, HEAD_SIZE, bias=False)
        self.query = nn.Linear(N_EMBEDS, HEAD_SIZE, bias=False)
        self.value = nn.Linear(N_EMBEDS, HEAD_SIZE, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        _, T, C = x.shape
        key = self.key(x)
        query = self.query(x)

        initialized_weights = query @ key.transpose(-2, -1) * HEAD_SIZE ** -0.5
        initialized_weights = initialized_weights.masked_fill(self.tril_matrix[:T, :T] == 0, float("-inf"))
        # Basically we have to take the avg of context vector
        initialized_weights = F.softmax(initialized_weights, dim=-1)
        initialized_weights = self.dropout(initialized_weights)

        value = self.value(x)
        changed_x = initialized_weights @ value

        return changed_x


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead() for _ in range(NUM_HEADS)])
        self.projection = nn.Linear(HEAD_SIZE * NUM_HEADS, N_EMBEDS)

    def forward(self, x):
        concat_x = torch.concat([h(x) for h in self.heads], dim=-1)
        return self.projection(concat_x)


class FeedFoward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBEDS, N_EMBEDS * 4),
            nn.ReLU(),
            nn.Linear(N_EMBEDS * 4, N_EMBEDS),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention()
        self.feed_forward = FeedFoward()
        self.layer_norm = nn.LayerNorm(N_EMBEDS)

    def forward(self, x):
        x = self.layer_norm(x)
        x = x + self.multi_head_attention(x)
        x = self.layer_norm(x)
        x = x + self.feed_forward(x)
        return x
