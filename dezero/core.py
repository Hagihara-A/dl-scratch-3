from __future__ import annotations
import dezero
import heapq as hq
import weakref
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

from .config import Config, using_config


class Variable:
    __array_priority__ = 200

    def __init__(self, data: np.ndarray | None, name: str = "") -> None:
        if (data is not None) and (not isinstance(data, np.ndarray)):
            raise TypeError(f"data must be ndarray, not {type(data)}")

        self.data = data
        self.grad: Optional[Variable] = None
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

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))
        funcs: list[tuple[int, Function]] = []
        seen_set: set[Function] = set()

        def add_func(f: Function):
            if f not in seen_set:
                hq.heappush(funcs, (-f.generation, f))
                seen_set.add(f)
        add_func(self.creator)
        while funcs:
            _, f = hq.heappop(funcs)
            gys = [output().grad for output in f.outputs]

            with using_config("enable_backprop", create_graph):
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

    def transpose(self, axes=None):
        return dezero.functions.transpose(self, axes)

    @property
    def T(self):
        return self.transpose()

    def reshape(self, shape: tuple[int, ...]):
        return dezero.functions.reshape(self, shape)


Operatable = Union[int, float, np.ndarray, Variable, None]


def as_variable(x: Operatable):
    if isinstance(x, Variable):
        return x
    else:
        if np.isscalar(x):
            return Variable(np.array(x))
        elif isinstance(x, np.ndarray):
            return Variable(x)
        elif x is None:
            return Variable(x)
        else:
            raise TypeError(
                f"given type is {type(x)},\
                 expected Union[int, float, np.ndarray, Variable, None]")


class Function(ABC):
    def __call__(self, *inputs_raw: Operatable):
        inputs = [as_variable(x) for x in inputs_raw]
        xs = [x.data for x in inputs]
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
    def backward(self, *gys: Variable) -> tuple[Variable, ...]:
        pass

    def __lt__(self, other: Function):
        return self.generation < other.generation


class Add(Function):
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

    def forward(self, *xs: np.ndarray):
        x0, x1 = xs
        return x0+x1,

    def backward(self, *gys: Variable):
        gy, = gys
        x0, x1 = self.inputs
        gx0, gx1 = gy, gy
        if (x0.shape != x1.shape):
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return (gx0, gx1)


def add(x0: Variable, x1: Operatable):
    return Add()(x0, x1)


class Mul(Function):
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

    def forward(self, *xs: np.ndarray):
        x0, x1 = xs
        return x0*x1,

    def backward(self, *gys: Variable):
        gy, = gys
        x0, x1 = self.inputs
        gx0, gx1 = x1*gy, x0 * gy
        if (x0.shape != x1.shape):
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0: Variable, x1: Operatable):
    return Mul()(x0, x1)


class Neg(Function):
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

    def forward(self, *xs: np.ndarray):
        x, = xs
        return -x,

    def backward(self, *gys: Variable):
        gy, = gys
        return -gy,


def neg(x: Variable):
    return Neg()(x)


class Sub(Function):
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

    def forward(self, *xs: np.ndarray):
        x0, x1 = xs
        return x0 - x1,

    def backward(self, *gys: Variable):
        gy, = gys
        x0, x1 = self.inputs
        gx0, gx1 = gy, -gy
        if (x0.shape != x1.shape):
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def sub(x0: Operatable, x1: Operatable):
    return Sub()(x0, x1)


class Div(Function):
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

    def forward(self, *xs: np.ndarray):
        x0, x1 = xs
        return x0/x1,

    def backward(self, *gys: Variable):
        gy, = gys
        x0, x1 = self.inputs
        gx0, gx1 = gy/x1, -gy*x0/(x1**2)
        if x0.shape != x1.shape:
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def div(x0: Operatable, x1: Operatable):
    return Div()(x0, x1)


class Pow(Function):
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

    def __init__(self, c: float):
        self.c = c

    def forward(self, *xs: np.ndarray):
        x, = xs
        return x**self.c,

    def backward(self, *gys: Variable):
        gy, = gys
        x, = self.inputs
        return gy * self.c * x ** (self.c - 1),


def pow(x: Variable, c: int):
    return Pow(c)(x)


class Parameter(Variable):
    pass
