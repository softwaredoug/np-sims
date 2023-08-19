import os.path
import pathlib
import numpy as np

DATA_PATH = 'data/'
pathlib.Path(DATA_PATH).mkdir(parents=True, exist_ok=True)


def read_np(key):
    path = os.path.join(DATA_PATH, key + '.npy')
    return np.load(path)


def write_np(key, data):
    path = os.path.join(DATA_PATH, key + '.npy')
    np.save(path, data)


def read_lines(key):
    path = os.path.join(DATA_PATH, key + '.txt')
    with open(path, 'r') as f:
        return f.readlines()


def write_lines(key, lines):
    path = os.path.join(DATA_PATH, key + '.txt')
    with open(path, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')


def lines_of(key):
    path = os.path.join(DATA_PATH, key + '.txt')
    with open(path, "r") as f:
        for line in f:
            yield line
