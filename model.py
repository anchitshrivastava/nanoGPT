import torch
import torch.nn as nn
from torch.nn import functional as F

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
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.embedding_table(idx)  # B, T, C
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
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.no_grad()
    def estimate_loss(self, train_generator, test_generator, eval_iters):
        out = {}

        self.eval()

        losses = torch.zeros(eval_iters)
        for index in range(eval_iters):
            X, y = next(train_generator)
            logits, loss = self(X, y)
            losses[index] = loss.item()
        out['train'] = losses.mean()

        losses = torch.zeros(eval_iters)
        for index in range(eval_iters):
            X, y = next(test_generator)
            logits, loss = self(X, y)
            losses[index] = loss.item()
        out['test'] = losses.mean()

        self.train()

        return out

    def train_model(self, epochs, eval_epochs, train_generator, test_generator):

        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=1e-5, patience=1000)

        for epoch in range(epochs):
            x, y = next(train_generator)
            logits, loss = self(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                losses = self.estimate_loss(train_generator=train_generator, test_generator=test_generator,
                                            eval_iters=eval_epochs)
                # scheduler.step(losses['test'])
                print(f"Train loss is {losses['train']}")
                print(f"Validation loss is {losses['test']}")
                print("epoch is ", epoch)
