from __future__ import annotations

import numpy as np

from .core import Function, Variable, as_variable


class Square(Function):
    def forward(self, *xs: np.ndarray):
        x, = xs
        return x ** 2,

    def backward(self, *gys: Variable):
        x, = self.inputs
        gy, = gys
        gx = 2 * x * gy
        return gx,


def square(x: Variable):
    return Square()(x)


class Exp(Function):
    def forward(self, *xs: np.ndarray):
        x, = xs
        return np.exp(x),

    def backward(self, *gys: Variable):
        gy, = gys
        x, = self.inputs
        gx = np.exp(x.data) * gy
        return gx,


def exp(x: Variable):
    return Exp()(x)


class Sin(Function):
    def forward(self, *xs: np.ndarray):
        x, = xs
        return np.sin(x),

    def backward(self, *gys: Variable):
        x, = self.inputs
        gy, = gys
        return gy * np.cos(x.data),


def sin(x: Variable):
    return Sin()(x)


class Cos(Function):
    def forward(self, *xs: np.ndarray):
        x, = xs
        return np.cos(x),

    def backward(self, *gys: Variable):
        x, = self.inputs
        gy, = gys
        return gy * -sin(x),


def cos(x: Variable):
    return Cos()(x)


class Tanh(Function):
    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray, ...]:
        x, = xs
        y = np.tanh(x)
        return y,

    def backward(self, *gys: Variable) -> tuple[Variable, ...]:
        gy, = gys
        y, = self.outputs
        return gy * (1-y()**2),


def tanh(x: Variable):
    return Tanh()(x)


class Reshape(Function):
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def forward(self, *xs: np.ndarray):
        x, = xs
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y,

    def backward(self, *gys: Variable) -> tuple:
        gy, = gys
        return reshape(gy, self.x_shape),


def reshape(x: Variable, shape: tuple[int, ...]):
    if x.shape == shape:
        return as_variable(x)
    else:
        return Reshape(shape)(x)


class Transpose(Function):
    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray, ...]:
        x, = xs
        return x.transpose(),

    def backward(self, *gys: Variable) -> tuple[Variable, ...]:
        gy, = gys
        return transpose(gy),


def transpose(x: Variable):
    return Transpose()(x)
