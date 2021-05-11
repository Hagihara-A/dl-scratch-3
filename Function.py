from Variable import Variable
from abc import ABC, abstractmethod
import numpy as np


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    else:
        return x


class Function(ABC):
    def __call__(self, *inputs: Variable):
        xs = (x.data for x in inputs)
        ys = self.forward(*xs)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.creator = self
        self.inputs = inputs
        self.outputs = outputs
        return outputs[0] if len(outputs) == 1 else outputs

    @abstractmethod
    def forward(self, *xs: np.ndarray) -> tuple:
        pass

    @abstractmethod
    def backward(self, *gy: np.ndarray) -> tuple:
        pass


class Square(Function):
    def forward(self, *xs: np.ndarray):
        return xs[0] ** 2,

    def backward(self, gy: np.ndarray):
        x = self.inputs
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, *xs: np.ndarray):
        return np.exp(xs[0]),

    def backward(self, gy: np.ndarray):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


class Add(Function):
    def forward(self, *xs: np.ndarray):
        x0, x1 = xs
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
