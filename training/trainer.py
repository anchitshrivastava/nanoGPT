from functools import partial
import numpy as np
import torch
from datasets import Dataset, from_generator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import model_config
from utils import get_train_batch

DEFAULT_TRAINING_DATASET = "tatsu-lab/alpaca"
DEFAULT_INPUT_MODEL = "EleutherAI/gpt-j-6B"
RESPONSE_KEY = "### Response:\n"
DEFAULT_SEED = 42
MAX_LENGTH = 1024

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_INPUT_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model =  AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B",
    revision="float16",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    cache_dir = "/home/sarabjot/storage/backup2/pathfactory_models"
)
    
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        
        batch = super().torch_call(examples)

        response_token_ids = self.tokenizer.encode(RESPONSE_KEY)

        labels = batch["labels"].clone()

        for i in range(len(examples)):

            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                if np.array_equal(response_token_ids, batch["labels"][i, idx : idx + len(response_token_ids)]):
                    response_token_ids_start_idx = idx
                    break

            if response_token_ids_start_idx is None:
                raise RuntimeError("Could not find response key token IDs")

            response_token_ids_end_idx = response_token_ids_start_idx + len(response_token_ids)

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch


def preprocess_batch(batch, tokenizer, max_length=MAX_LENGTH):
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


def load_training_dataset(training_data_id= DEFAULT_TRAINING_DATASET):
    dataset = from_generator(get_train_batch)

    # Remove empty responses
    dataset = dataset.filter(lambda rec: not rec["text"].strip().endswith("### Response:"))

    def _func(rec):
        rec["text"] += "\n\n### End"
        return rec

    dataset = dataset.map(_func)

    return dataset


def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int = MAX_LENGTH, seed=DEFAULT_SEED) -> Dataset:
    """Loads the training dataset and tokenizes it so it is ready for training.

    Args:
        tokenizer (AutoTokenizer): Tokenizer tied to the model.
        max_length (int, optional): Maximum number of tokens to emit from tokenizer. Defaults to MAX_INPUT_LENGTH.

    Returns:
        Dataset: HuggingFace dataset
    """

    dataset = load_training_dataset()
    
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "input", "output", "text"],
    )
    dataset = dataset.shuffle(seed=seed)

    return dataset


def train(
    test_size=1000,
):
    set_seed(DEFAULT_SEED)

    processed_dataset = preprocess_dataset(tokenizer=tokenizer, seed=DEFAULT_SEED)

    split_dataset = processed_dataset.train_test_split(test_size=test_size, seed=DEFAULT_SEED)

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
    )

    training_args = TrainingArguments(
        learning_rate=1e-5,
        num_train_epochs=3,
        deepspeed=model_config,
        eval_steps=10,
        remove_unused_columns=False
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




if __name__ == "__main__":
    train()
