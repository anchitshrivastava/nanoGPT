import torch
import torch.nn as nn
from torch.nn import functional as F

from layers import SelfAttentionHead

from utils import get_batch

from config import N_EMBEDS, CONTEXT_SIZE, DEVICE, HEAD_SIZE, LR

"""
The layers required for this project are:
1. Embedding layer 
2. Multi-Head Attention layer
3. Feed forward layer
4. Softmax layer 
5. Linear layer
6. Normalization layer
"""


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, N_EMBEDS)
        self.positional_embedding_table = nn.Embedding(CONTEXT_SIZE, N_EMBEDS)
        self.lm_head = nn.Linear(HEAD_SIZE, vocab_size)
        self.attention_head = SelfAttentionHead()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos_embeds = self.positional_embedding_table(torch.arange(T, device=DEVICE))  # T, C
        embeds = self.embedding_table(idx)  # B, T, C
        x = embeds + pos_embeds
        x = self.attention_head(x)
        logits = self.lm_head(x)  # B, T, vocab_size
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)  # requires B, C, T
        return logits, loss

    def generate(self, idx, max_new_tokens: int = 100):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -CONTEXT_SIZE:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.no_grad()
    def estimate_loss(self, eval_iters):
        out = {}

        self.eval()

        losses_train = torch.zeros(eval_iters)
        for index in range(eval_iters):
            X_train, y_train = get_batch("train")
            logits_train, loss_train = self(idx=X_train, targets=y_train)
            losses_train[index] = loss_train.item()
        out['train'] = losses_train.mean()

        losses_test = torch.zeros(eval_iters)
        for index in range(eval_iters):
            X_test, y_test = get_batch("test")
            logits_test, loss_test = self(X_test, y_test)
            losses_test[index] = loss_test.item()
        out['test'] = losses_test.mean()

        self.train()

        return out

    def train_model(self, epochs, eval_epochs):

        optimizer = torch.optim.AdamW(self.parameters(), lr=LR)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=1e-5, patience=1000)

        for epoch in range(epochs):
            x, y = get_batch("train")
            logits, loss = self(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if epoch % 500 == 0:
                losses = self.estimate_loss(eval_iters=eval_epochs)
                # scheduler.step(losses['test'])
                print(f"Train loss is {losses['train']}")
                print(f"Validation loss is {losses['test']}")
                print("epoch is ", epoch)
