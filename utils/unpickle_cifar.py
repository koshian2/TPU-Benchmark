import pickle, os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

with open("cifar-100-python/train", "rb") as fp:
    train = pickle.load(fp, encoding="latin-1")
with open("cifar-100-python/test", "rb") as fp:
    test = pickle.load(fp, encoding="latin-1")


def parse_pickle(rawdata, rootdir):
    for i in range(100):
        dir = rootdir + "/" + f"{i:02d}"
        if not os.path.exists(dir):
            os.mkdir(dir)    
    m = len(rawdata["filenames"])
    for i in range(m):
        if i % 100 == 0:
            print(i)
        filename = rawdata["filenames"][i]
        label = rawdata["fine_labels"][i]
        data = rawdata["data"][i]
        data = data.reshape(3, 32, 32)
        data = np.swapaxes(data, 0, 2)
        data = np.swapaxes(data, 0, 1)
        with Image.fromarray(data) as img:
            img.save(f"{rootdir}/{label:02d}/{filename}")

parse_pickle(train, "cifar100-raw/train")
parse_pickle(test, "cifar100-raw/test")