import torch
import torch.nn.functional as F

from utils.model import return_model, return_data_loader

from config import LR, DEVICE

class Trainer:
    def __init__(self):
        self.model = return_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR)
    
    def training_step(self, row):
        # here row is the batch that is provided I am just too lazy to change the names
        self.optimizer.zero_grad()
        batch_in = {
            "input_ids": torch.tensor(row['input_ids']).to(DEVICE),
            "attention_mask": torch.tensor(row['attention_mask']).to(DEVICE)
        }
        out_ = self.model.forward(**batch_in,)
        out_ = out_.logits
        b, t, c = out_.shape
        out_ = out_.view(b*t, c)
        y_ = torch.tensor(row['labels']).flatten()
        loss = F.cross_entropy(out_, y_)
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def train_model():
        pass