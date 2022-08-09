"Decode file format specified at http://yann.lecun.com/exdb/mnist/"
import gzip
from glob import glob

import idx2numpy as npidx
import numpy as np


def main():
    import fetch

    fetch.main()

    arrays = dict()
    for path in glob("*.gz"):
        with gzip.open(path) as istrm:
            data = npidx.convert_from_string(istrm.read())
        pfx = "train" if "train" in path else "test"
        sfx = "images" if data.ndim > 2 else "labels"
        arrays[f"{pfx}_{sfx}"] = data

    np.savez("mnist.npz", **arrays)


if __name__ == "__main__":
    main()
