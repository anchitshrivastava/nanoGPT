import numpy as np
import torch

from model import BigramLanguageModel

from utils import return_vocab, encode_data, decode_data, data_generator

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    # reading the dataset

    with open("./dataset/shakespeare.txt", "r") as file:
        data = file.read()
    with open("./dataset/shakespeare.txt", "r") as file:
        data_lines = file.readlines()
        print("Average length of line", np.mean([len(i) for i in data_lines]))
        print("Max length of line", np.max([len(i) for i in data_lines]))

    print(data[:10])

    vocab = return_vocab(data=data)
    # splitting in train and test splits

    total_len = len(data)
    train_split = 0.90
    test_split = 0.10

    train = data[:int(train_split * total_len)]
    test = data[-1 * int(test_split * total_len) - 1:]

    print(f"""Len of Train = {len(train)} \n Len of Test = {len(test)}""")

    encoded_train_data = encode_data(data=train)
    encoded_test_data = encode_data(data=test)

    encoded_train_data = torch.tensor(encoded_train_data, dtype=torch.long)
    encoded_test_data = torch.tensor(encoded_test_data, dtype=torch.long)

    train_generator = data_generator(data=encoded_train_data)
    test_generator = data_generator(data=encoded_test_data)

    base_line_model = BigramLanguageModel(vocab_size=len(vocab))

    base_line_model = base_line_model.to(DEVICE)

    ################# TESTING ########################
    x, y = next(train_generator)
    out, loss = base_line_model(x, y)

    print(loss)

    # trying to generate randomly

    print(decode_data(base_line_model.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=DEVICE),
                                               max_new_tokens=1000)[0].tolist(), vocab=vocab))
    #################################################

    base_line_model.train_model(epochs=5000, eval_epochs=200, train_generator=train_generator,
                                test_generator=test_generator)

    print(decode_data(base_line_model.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=DEVICE),
                                               max_new_tokens=1000)[0].tolist(), vocab=vocab))
