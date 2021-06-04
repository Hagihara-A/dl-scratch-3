import math
import numpy as np
from .datasets import Dataset


class DataLoader():
    def __init__(self, dataset: Dataset, batch_size: int, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)

        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index: np.ndarray = np.random.permutation(self.data_size)
        else:
            self.index: np.ndarray = np.arange(self.data_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration()
        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i*batch_size: (i+1)*batch_size]
        batch = [self.dataset[i] for i in batch_index]
        x = np.array([e[0] for e in batch])
        t = np.array([e[1] for e in batch])
        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()
