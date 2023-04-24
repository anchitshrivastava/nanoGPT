from functools import partial

from torch.utils.data import DataLoader

from datasets import load_dataset

from config import RESPONSE_KEY_NL, MAX_TOKENIZE_LEN, BATCH_SIZE

from utils.model import return_tokenizer

from utils.data_helpers import adding_starting_ending_token, tokenize_batch, creating_masking_labels_inputs

def return_alpaca_dataset_batch():
    dataset = load_dataset("tatsu-lab/alpaca", split='train', streaming=True)
    
    dataset = dataset.shuffle(buffer_size=BATCH_SIZE, seed=42)
    
    return dataset

def preprocess_dataset(dataset):
    
    tokenizer = return_tokenizer()
    
    response_key_stripped = RESPONSE_KEY_NL.strip()
    dataset = dataset.filter(lambda rec: not rec["text"].strip().endswith(response_key_stripped))
    
    add_start_end_token = partial(adding_starting_ending_token, tokenizer=tokenizer)
    dataset = dataset.map(add_start_end_token)
    
    tokenizing_function = partial(tokenize_batch, max_length=MAX_TOKENIZE_LEN, tokenizer=tokenizer)
    
    dataset = dataset.map(
    tokenizing_function,
    batched=True,
    remove_columns=["instruction", "input", "output", "text"],
    )
    
    dataset = dataset.map(creating_masking_labels_inputs)
    
    return dataset


def return_data_loader():
    dataset = return_alpaca_dataset_batch()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2, 
                            collate_fn=lambda x: preprocess_dataset(x))
    return dataloader
