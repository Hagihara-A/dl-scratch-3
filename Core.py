from __future__ import annotations
from Config import Config
import weakref
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class Variable:
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

    def __mul__(self, other):
        return mul(self, other)

    def __add__(self, other: Variable):
        return add(self, other)


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


class Mul(Function):
    def __call__(self, *inputs: Variable) -> Variable:
        return super().__call__(*inputs)

    def forward(self, *xs: np.ndarray):
        x0, x1 = xs
        return x0*x1,

    def backward(self, *gys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gy, = gys
        x0, x1 = self.inputs
        return x1.data*gy, x0.data*gy


def square(x: Variable):
    return Square()(x)


def exp(x: Variable):
    return Exp()(x)


def add(x0: Variable, x1: Variable):
    return Add()(x0, x1)


def mul(x0: Variable, x1: Variable):
    return Mul()(x0, x1)
