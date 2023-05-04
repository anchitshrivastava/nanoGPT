import os
import argparse
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import click

import numpy as np
from datasets import Dataset, load_dataset

import torch
import torch.nn.functional as F

import deepspeed

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer
)

from config import (
    PROMPT_WITH_INPUT_FORMAT,
    PROMPT_NO_INPUT_FORMAT,
    END_KEY,
    INSTRUCTION_KEY,
    RESPONSE_KEY_NL,
)

DEVICE = "cuda"
latest_checkpoint = "EleutherAI/gpt-neo-125m"
model_dir = "/home/sarabjot/pathfactory/nanoGPT/GPT-MPT/trainer_mpt_saved_290423" # "EleutherAI/gpt-neo-125m"
# model_dir = "/home/sarabjot/pathfactory/nanoGPT/GPT-MPT/trainer_saved_030523" # "EleutherAI/gpt-neo-1.3B"
DATASET = "databricks/databricks-dolly-15k"
EPOCH=3
        
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        # The prompt ends with the response key plus a newline.  We encode this and then try to find it in the
        # sequence of tokens.  This should just be a single token.
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)

        labels = batch["labels"].clone()

        for i in range(len(examples)):

            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch


def preprocess_batch(batch: Dict[str, List], tokenizer: AutoTokenizer, max_length: int) -> dict:
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


def load_training_dataset(path_or_dataset: str = DATASET) -> Dataset:
    dataset = load_dataset(path_or_dataset)["train"]

    def _add_text(rec):
        instruction = rec["instruction"]
        response = rec["response"]
        context = rec.get("context")

        if not instruction:
            raise ValueError(f"Expected an instruction in: {rec}")

        if not response:
            raise ValueError(f"Expected a response in: {rec}")

        # For some instructions there is an input that goes along with the instruction, providing context for the
        # instruction.  For example, the input might be a passage from Wikipedia and the instruction says to extract
        # some piece of information from it.  The response is that information to extract.  In other cases there is
        # no input.  For example, the instruction might be open QA such as asking what year some historic figure was
        # born.
        if context:
            rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
        else:
            rec["text"] = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
        return rec

    dataset = dataset.map(_add_text)

    return dataset


def load_tokenizer(pretrained_model_name_or_path) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})

    return tokenizer


def load_model(
    pretrained_model_name_or_path: str, *, gradient_checkpointing: bool = False
) -> AutoModelForCausalLM:
    
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True,)
    config.hidden_size = config.hidden_size

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
    cache_dir = "/home/sarabjot/storage/backup2/pathfactory_models",
    # gradient_checkpointing=True,
    # trust_remote_code=True,
    config = config
    )
    return model


def get_model_tokenizer(
    pretrained_model_name_or_path: str, *, gradient_checkpointing: bool = False
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    model = load_model(pretrained_model_name_or_path, )
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    """Loads the training dataset and tokenizes it so it is ready for training.

    Args:
        tokenizer (AutoTokenizer): Tokenizer tied to the model.
        max_length (int): Maximum number of tokens to emit from tokenizer.

    Returns:
        Dataset: HuggingFace dataset
    """

    dataset = load_training_dataset()
    
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "context", "response", "text", "category"],
    )

    # Make sure we don't have any truncated records, as this would mean the end keyword is missing.
    dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)

    dataset = dataset.shuffle()

    return dataset


if os.path.exists(model_dir):
    checkpoints = os.listdir(model_dir)
    if checkpoints:
        checkpoints = [os.path.join(model_dir, i) for i in checkpoints]
        checkpoints = max(checkpoints, key=os.path.getmtime)
        latest_checkpoint = checkpoints
    

model, tokenizer = get_model_tokenizer(
    pretrained_model_name_or_path=latest_checkpoint
)

# Use the same max length that the model supports.  Fall back to 1024 if the setting can't be found.
# The configuraton for the length can be stored under different names depending on the model.  Here we attempt
# a few possible names we've encountered.
conf = model.config
max_length = None
for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
    max_length = getattr(model.config, length_setting, None)
    if max_length:
        break
if not max_length:
    max_length = 1024

processed_dataset = preprocess_dataset(tokenizer=tokenizer, max_length=max_length)

split_dataset = processed_dataset.train_test_split(test_size=0.10)

data_collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
)


# deepspeed.init_distributed()

# model, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
#                                                      model=model)
# model.config = AutoConfig.from_pretrained(latest_checkpoint, trust_remote_code=True,)

print("*"*100)
print(model.config.hidden_size)
print("*"*100)

def train(lr, local_rank, deepspeed):
    training_args = TrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        # fp16=False,
        learning_rate=lr,
        num_train_epochs=EPOCH,
        # logging_dir=f"{model_dir}/runs",
        # logging_strategy="steps",
        # logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=10_000,
        save_strategy="steps",
        save_steps=10_000,
        load_best_model_at_end=False,
        # report_to="tensorboard",
        # disable_tqdm=True,
        remove_unused_columns=False,
        local_rank=local_rank,
        warmup_steps=500,
        deepspeed=deepspeed
        )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
    )

    trainer.train()
    # trainer.save_model(output_dir=model_dir)


@click.command()
@click.option("--lr", type=float, default=1e-5, help="Learning rate to use for training.")
@click.option("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
@click.option(
    "--local_rank",
    type=str,
    default=True,
    help="Provided by deepspeed to identify which instance this process is when performing multi-GPU training.",
)
def main(**kwargs):
    train(**kwargs)
    
if __name__ == "__main__":
    main()