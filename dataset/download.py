import os
import requests

if __name__ == '__main__':

    filename = "shakespeare.txt"

    download_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    if not os.path.exists(f"./{filename}"):
        with open(filename, "w") as file:
            file.write(requests.get(download_url).text)
