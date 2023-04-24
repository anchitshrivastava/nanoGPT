import os
import torch

from config import DEVICE, END_KEY, RESPONSE_KEY_NL, INSTRUCTION_KEY,  \
    START_TOKEN, INPUT_KEY, MODEL_NAME, model_config

from transformers import AutoTokenizer, AutoModelForCausalLM

def return_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.bos_token = START_TOKEN
    tokenizer.pad_token = END_KEY
    tokenizer.eos_token = END_KEY
    tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL, INPUT_KEY]})
    
    return tokenizer
    
def return_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True,
        cache_dir = "/home/sarabjot/storage/backup2/pathfactory_models",
        config = model_config
    )
    model.to(DEVICE)

    tokenizer = return_tokenizer()
    model.resize_token_embeddings(len(tokenizer))
    
    if os.path.exists("/home/sarabjot/PathFactory/GPT-j/saved_hf_model/saved_model.pth"):
        model.load_state_dict(torch.load("/home/sarabjot/PathFactory/GPT-j/saved_hf_model/saved_model.pth"))
    
    return model