
from typing import List, Tuple
from Variable import Variable
from abc import ABC, abstractmethod
import numpy as np


class Function(ABC):
    def __call__(self, *inputs: Variable):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(y) for y in ys]

        for output in outputs:
            output.creator = self
        self.inputs = inputs
        self.outputs = outputs
        return outputs

    @abstractmethod
    def forward(self, *xs: np.ndarray) -> Tuple[np.ndarray, ...]:
        pass

    @abstractmethod
    def backward(self, gy: np.ndarray):
        pass


class Square(Function):
    def forward(self, x: np.ndarray):
        return x ** 2

    def backward(self, gy: np.ndarray):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: np.ndarray):
        return np.exp(x)

    def backward(self, gy: np.ndarray):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray):
        return x0+x1,

    def backward(self, gy: np.ndarray):
        return (gy, gy)


def square(x: Variable):
    return Square()(x)


def exp(x: Variable):
    return Exp()(x)


x = Variable(np.array([0.5]))
a = square(x)
b = exp(a)
y = square(b)

y.backward()
print(x.grad)
