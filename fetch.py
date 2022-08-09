from os.path import exists
from shutil import copyfileobj
from urllib.request import urlopen

endpoint = "http://yann.lecun.com/exdb/mnist/"
paths = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def main():
    for path in paths:
        if not exists(path):
            print(f"fetch {path}...")
            with urlopen(endpoint + path) as istrm:
                with open(path, "wb+") as ostrm:
                    copyfileobj(istrm, ostrm)


if __name__ == "__main__":
    main()
