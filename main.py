import torch

from model import BigramLanguageModel

from utils import return_vocab, decode_data, get_batch, encode_train_test

from config import DEVICE

if __name__ == '__main__':
    # reading the dataset
    vocab = return_vocab()

    encoded_train_data, encoded_test_data = encode_train_test()

    base_line_model = BigramLanguageModel(vocab_size=len(vocab))

    base_line_model = base_line_model.to(DEVICE)

    ################# TESTING ########################
    x, y = get_batch("train")
    out, loss = base_line_model(x, y)

    print(loss)

    # trying to generate randomly

    print(decode_data(base_line_model.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=DEVICE),
                                               max_new_tokens=1000)[0].tolist(), vocab=vocab))
    #################################################

    base_line_model.train_model(epochs=5000, eval_epochs=200)

    print(decode_data(base_line_model.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=DEVICE),
                                               max_new_tokens=1000)[0].tolist(), vocab=vocab))
