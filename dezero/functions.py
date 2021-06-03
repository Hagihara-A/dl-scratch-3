from __future__ import annotations

from typing import Optional, Union

import numpy as np

from dezero import utils

from .core import Function, Operatable, Variable, as_variable


class Square(Function):
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

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
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

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
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

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
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

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
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

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
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

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
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray, ...]:
        x, = xs
        return x.transpose(),

    def backward(self, *gys: Variable) -> tuple[Variable, ...]:
        gy, = gys
        return transpose(gy),


def transpose(x: Variable):
    return Transpose()(x)


class Sum(Function):
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

    def __init__(self, axis: Optional[int | tuple[int]], keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray, ...]:
        x, = xs
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y,

    def backward(self, *gys: Variable) -> tuple[Variable, ...]:
        gy, = gys
        gy = utils.reshape_sum_backward(
            gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx,


def sum(x: Variable, axis: Optional[int | tuple[int]] = None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray, ...]:
        x, = xs
        self.x_shape = x.shape
        return np.broadcast_to(x, self.shape),

    def backward(self, *gys: Variable) -> tuple[Variable, ...]:
        gy, = gys
        gx = sum_to(gy, self.x_shape)
        return gx,


def broadcast_to(x: Variable, shape: tuple[int, ...]):
    if (x.shape == shape):
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray, ...]:
        x, = xs
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y,

    def backward(self, *gys: Variable) -> tuple[Variable, ...]:
        gy, = gys
        gx = broadcast_to(gy, self.x_shape)
        return gx,


def sum_to(x: Variable, shape: tuple[int, ...]):
    if x.shape == shape:
        return as_variable(x)
    else:
        return SumTo(shape)(x)


class MatMul(Function):
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray, ...]:
        x, W = xs
        y = x.dot(W)
        return y,

    def backward(self, *gys: Variable) -> tuple[Variable, ...]:
        gy, = gys
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x: Variable, W: Variable):
    return MatMul()(x, W)


class MeanSquaredError(Function):
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray, ...]:
        x0, x1 = xs
        diff = x0-x1
        return (diff**2).sum() / len(diff),

    def backward(self, *gys: Variable) -> tuple[Variable, ...]:
        gy, = gys
        x0, x1 = self.inputs
        diff = x0 - x1
        gy = broadcast_to(gy, diff.shape)
        gx0 = gy*diff*(2./len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0: Operatable, x1: Operatable):
    return MeanSquaredError()(x0, x1)


class Linear(Function):
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray, ...]:
        x, W, b = xs
        y = x.dot(W)
        if b is not None:
            y += b
        return y,

    def backward(self, *gys: Variable) -> tuple[Variable, ...]:
        x, W, b = self.inputs
        gy, = gys
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x: Operatable, W: Operatable,
           b: Operatable = None):
    return Linear()(x, W, b)


class Sigmoid(Function):
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray, ...]:
        x, = xs
        y = np.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y,

    def backward(self, *gys: Variable) -> tuple[Variable, ...]:
        gy, = gys
        y, = self.outputs
        gx = gy * y() * (1 - y())
        return gx,


def sigmoid(x):
    return Sigmoid()(x)


Slice = Union[list[int], np.ndarray]


class GetItem(Function):
    def __init__(self, slices: Slice):
        self.slices = slices

    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray, ...]:
        x, = xs
        return x[self.slices],

    def backward(self, *gys: Variable) -> tuple[Variable, ...]:
        gy, = gys
        x, = self.inputs
        return GetItemGrad(self.slices, x.shape)(gy),


class GetItemGrad(Function):
    def __init__(self, slices: Slice, in_shape: tuple[int]):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, *gys: np.ndarray) -> tuple[np.ndarray, ...]:
        gy, = gys
        gx = np.zeros(self.in_shape, dtype=gy.dtype)
        np.add.at(gx, self.slices, gy)
        return gx,

    def backward(self, *ggxs: Variable) -> tuple[Variable, ...]:
        ggx, = ggxs
        return get_item(ggx, self.slices),


def get_item(x: Variable, indices: list[int] | np.ndarray):
    return GetItem(indices)(x)


class Softmax(Function):
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, *xs: np.ndarray):
        x, = xs
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y,

    def backward(self, *gys: Variable):
        gy, = gys
        y = self.outputs[0]()
        gx = gy * y * (1-y)
        return gx,


def softmax(x: Operatable, axis=1):
    return Softmax(axis)(x)
