from __future__ import annotations
from Config import Config
import weakref
from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np


class Variable:
    __array_priority = 200

    def __init__(self, data: np.ndarray, name: Optional[str] = None) -> None:
        self.data = data
        self.grad: Optional[np.ndarray] = None
        self.__creator: Optional[Function] = None
        self.generation = 0
        self.name = name

    @property
    def creator(self):
        return self.__creator

    @creator.setter
    def creator(self, creator: Function):
        self.generation = creator.generation + 1
        self.__creator = creator

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs: list[Function] = []
        seen_set: set[Function] = set()

        def add_func(f: Function):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def clear_grad(self):
        self.grad = None

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    def __len__(self):
        return len(self.data)

    def __repr__(self) -> str:
        desc = 'variable({})'
        if self.data is None:
            return desc.format(None)
        else:
            return desc.format(str(self.data).replace("\n", "\n" + " "*9))

    def __mul__(self, other: Operatable):
        return mul(self, other)

    def __add__(self, other: Operatable):
        return add(self, other)

    def __rmul__(self, other: Operatable):
        return mul(self, other)

    def __radd__(self, other: Operatable):
        return add(self, other)

    def __neg__(self):
        return neg(self)

    def __sub__(self, other: Operatable):
        return sub(self, other)

    def __rsub__(self, other: Operatable):
        return sub(other, self)

    def __truediv__(self, other: Operatable):
        return div(self, other)

    def __rtruediv__(self, other: Operatable):
        return div(other, self)

    def __pow__(self, other: int):
        return pow(self, other)


Operatable = Union[int, float, np.ndarray, Variable]


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    else:
        return x


def as_variable(x):
    if isinstance(x, Variable):
        return x
    else:
        return Variable(as_array(x))


class Function(ABC):
    def __call__(self, *inputs_raw: Operatable):
        inputs = [as_variable(x) for x in inputs_raw]
        xs = (x.data for x in inputs)
        ys = self.forward(*xs)
        outputs = [as_variable(y) for y in ys]
        if Config.enable_backprop:
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
    def backward(self, *gys: np.ndarray) -> tuple[np.ndarray, ...]:
        pass


class Square(Function):
    def forward(self, *xs: np.ndarray):
        x, = xs
        return x ** 2,

    def backward(self, *gy: np.ndarray):
        x = self.inputs[0].data
        gx = 2 * x * gy[0]
        return gx,


def square(x: Variable):
    return Square()(x)


class Exp(Function):
    def forward(self, *xs: np.ndarray):
        x, = xs
        return np.exp(x),

    def backward(self, *gys: np.ndarray):
        gy, = gys
        x, = self.inputs
        gx = np.exp(x.data) * gy
        return gx,


def exp(x: Variable):
    return Exp()(x)


class Add(Function):
    def forward(self, *xs: np.ndarray):
        x0, x1 = xs
        return x0+x1,

    def backward(self, *gys: np.ndarray):
        gy, = gys
        return (gy, gy)


def add(x0: Variable, x1: Operatable):
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, *xs: np.ndarray):
        x0, x1 = xs
        return x0*x1,

    def backward(self, *gys: np.ndarray):
        gy, = gys
        x0, x1 = self.inputs
        return x1.data*gy, x0.data*gy


def mul(x0: Variable, x1: Operatable):
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, *xs: np.ndarray):
        x, = xs
        return -x,

    def backward(self, *gys: np.ndarray):
        gy, = gys
        return -gy


def neg(x: Variable):
    return Neg()(x)


class Sub(Function):
    def forward(self, *xs: np.ndarray):
        x0, x1 = xs
        return x0 - x1,

    def backward(self, *gys: np.ndarray):
        gy, = gys
        return gy, -gy


def sub(x0: Operatable, x1: Operatable):
    return Sub()(x0, x1)


class Div(Function):
    def forward(self, *xs: np.ndarray):
        x0, x1 = xs
        return x0/x1,

    def backward(self, *gys: np.ndarray):
        gy, = gys
        x0, x1 = [x.data for x in self.inputs]
        return gy/x1, -gy*x0/(x1**2)


def div(x0: Operatable, x1: Operatable):
    return Div()(x0, x1)


class Pow(Function):
    def __init__(self, c: float):
        self.c = c

    def forward(self, *xs: np.ndarray):
        x, = xs
        return x**self.c,

    def backward(self, *gys: np.ndarray):
        gy, = gys
        x = self.inputs[0].data
        return gy * self.c * x ** (self.c - 1),


def pow(x: Variable, c: int):
    return Pow(c)(x)
