from Variable import Variable
from abc import ABC, abstractmethod
import numpy as np
import weakref


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
        self.generation = max([x.generation for x in inputs])

        for output in outputs:
            output.creator = self
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs[0] if len(outputs) == 1 else outputs

    @abstractmethod
    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray, ...]:
        pass

    @abstractmethod
    def backward(self, *gy: np.ndarray) -> tuple[np.ndarray, ...]:
        pass


class Square(Function):

    def __call__(self, *inputs: Variable) -> Variable:
        return super().__call__(*inputs)

    def forward(self, *xs: np.ndarray):
        x, = xs
        return x ** 2,

    def backward(self, *gy: np.ndarray):
        x = self.inputs[0].data
        gx = 2 * x * gy[0]
        return gx,


class Exp(Function):

    def __call__(self, *inputs: Variable) -> Variable:
        return super().__call__(*inputs)

    def forward(self, *xs: np.ndarray):
        x, = xs
        return np.exp(x),

    def backward(self, *gys: np.ndarray):
        gy, = gys
        x, = self.inputs
        gx = np.exp(x.data) * gy
        return gx,


class Add(Function):

    def __call__(self, *inputs: Variable) -> Variable:
        return super().__call__(*inputs)

    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray]:
        x0, x1 = xs
        return x0+x1,

    def backward(self, *gys: np.ndarray):
        gy, = gys
        return (gy, gy)


def square(x: Variable):
    return Square()(x)


def exp(x: Variable):
    return Exp()(x)


def add(x0: Variable, x1: Variable):
    return Add()(x0, x1)
