import tarfile

with tarfile.open("cifar100-raw.tar.gz", "x:gz") as tar:
    tar.add("cifar100-raw")