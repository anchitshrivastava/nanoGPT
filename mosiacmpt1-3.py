import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from config import PROMPT_FORMAT, DEVICE

model = AutoModelForCausalLM.from_pretrained('mosaicml/mpt-1b-redpajama-200b-dolly', 
                                             trust_remote_code=True,
                                             cache_dir = "/home/sarabjot/storage/backup2/pathfactory_models")

tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-1b-redpajama-200b-dolly")


def generate_response(instruction: str, *, model, tokenizer, 
                      do_sample: bool = True, max_new_tokens: int = 256, top_p: float = 0.92, top_k: int = 0, **kwargs) -> str:
    input_ids = tokenizer(PROMPT_FORMAT.format(instruction=instruction), return_tensors="pt").input_ids.to(DEVICE)

    # each of these is encoded to a single token
    response_key_token_id = tokenizer.encode("### Response:")[0]
    end_key_token_id = tokenizer.encode("### End")[0]

    gen_tokens = model.generate(input_ids, pad_token_id=tokenizer.pad_token_id, eos_token_id=end_key_token_id,
                                do_sample=do_sample, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k, **kwargs)[0].cpu()
    print(tokenizer.decode(gen_tokens).strip())
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

print(generate_response("Write a tweet announcing Dolly, a large language model from Databricks.", model=model, tokenizer=tokenizer))
print("hey")