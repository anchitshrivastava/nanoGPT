import numpy as np

import sys

sys.path.append("/home/sarabjot/PathFactory/GPT-j/")

from config import PROMPT_FORMAT, DEVICE

def generate_response(instruction: str, *, model, tokenizer, 
                      do_sample: bool = True, max_new_tokens: int = 256, top_p: float = 0.92, top_k: int = 0, **kwargs) -> str:
    input_ids = tokenizer(PROMPT_FORMAT.format(instruction=instruction), return_tensors="pt").input_ids.to("cuda:1")

    # each of these is encoded to a single token
    response_key_token_id = tokenizer.encode(RESPONSE_KEY_NL)[0]
    end_key_token_id = tokenizer.encode(END_KEY)[0]

    gen_tokens = model.generate(input_ids, pad_token_id=tokenizer.pad_token_id, eos_token_id=end_key_token_id,
                                do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k, **kwargs)[0].cpu()

    # find where the response begins
    response_positions = np.where(gen_tokens == response_key_token_id)[0]

    if len(response_positions) >= 0:
        response_pos = response_positions[0]
        
        # find where the response ends
        end_pos = None
        end_positions = np.where(gen_tokens == end_key_token_id)[0]
        if len(end_positions) > 0:
            end_pos = end_positions[0]

        return tokenizer.decode(gen_tokens[response_pos + 1 : end_pos]).strip()

    return None

if __name__ == "__main__":
    import torch
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    from config import *
    
    model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-neo-2.7B",
    low_cpu_mem_usage=True,
    cache_dir = "/home/sarabjot/storage/backup2/pathfactory_models",
    )

    model.to("cuda:1")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    tokenizer.bos_token = START_TOKEN
    tokenizer.pad_token = END_KEY
    tokenizer.eos_token = END_KEY
    tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL, INPUT_KEY]})

    model.resize_token_embeddings(len(tokenizer))
    
    model.load_state_dict(torch.load("/home/sarabjot/PathFactory/GPT-j/saved_hf_model/saved_model.pth"))

    
    generate_response(instruction="What is the difference between AI and Data Science", model=model, tokenizer=tokenizer)