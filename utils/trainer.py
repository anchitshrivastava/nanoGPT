import torch

from utils.model import return_model

from config import LR

class Trainer:
    def __init__(self):
        self.model = return_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR)
    
    def forward_pass():
        pass
    
    def backward_pass():
        pass
    
    def train_model():
        pass