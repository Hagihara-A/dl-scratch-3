from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import _ShapeLike
from dezero import utils
import dezero

from .core import Function, Operatable, Variable, as_variable
from .functions_conv import *
Array = Union[np.ndarray, Variable]


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
        x, self.axes = xs
        return x.transpose(self.axes),

    def backward(self, *gys: Variable) -> tuple[Variable, ...]:
        gy, = gys
        return transpose(gy, self.axes),


def transpose(x: Variable, axes: _ShapeLike = None):
    ax = None if axes is None else np.array(axes)
    return Transpose()(x, ax)


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


def _softmax(x: np.ndarray, axis=1) -> np.ndarray:
    x -= x.max(axis=axis, keepdims=True)
    x_exp = np.exp(x)
    y = x_exp / x_exp.sum(axis=1, keepdims=True) + 1e-7
    return y


class Softmax(Function):
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, *xs: np.ndarray):
        x, = xs
        return _softmax(x),

    def backward(self, *gys: Variable):
        gy, = gys
        y = self.outputs[0]()
        gx = gy * y * (1-y)
        return gx,


def softmax(x: Operatable, axis=1):
    return Softmax(axis)(x)


class Log(Function):
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray, ...]:
        x, = xs
        return np.log(x),

    def backward(self, *gys: Variable) -> tuple[Variable, ...]:
        x, = self.inputs
        gy, = gys
        return gy/x,


def log(x: Operatable):
    return Log()(x)


class Clip(Function):
    def __init__(self, x_min: float, x_max: Optional[float]):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        y = np.clip(x, self.x_min, self.x_max)
        return y,

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx,


def clip(x, x_min: float, x_max: Optional[float]):
    return Clip(x_min, x_max)(x)


def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]
    p = softmax(x)
    p = clip(p, 1e-15, 1.0)  # To avoid log(0)
    log_p = log(p)
    tlog_p = get_item(log_p, (np.arange(N), t.data))
    y = -1 * sum(tlog_p) / N
    return y


class SoftmaxCrossEntropy(Function):
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray, ...]:
        x, t = xs
        y = _softmax(x)

        L = -np.log(y[np.arange((len(x))), t]).sum() / len(x)
        return L,

    def backward(self, *gys: Variable) -> tuple[Variable, ...]:
        gy, = gys
        x, t = self.inputs
        y = softmax(x)
        t_onehot = np.eye(y.shape[1], dtype=t.dtype)[t.data]
        return (y - t_onehot) / len(y) * gy,


def softmax_cross_entropy(x: Operatable, t: Operatable):
    return SoftmaxCrossEntropy()(x, t)


def accuracy(y: np.ndarray, t: np.ndarray) -> float:
    pred: np.ndarray = y.argmax(axis=1).reshape(t.shape)
    result: np.ndarray = (pred == t)
    return result.mean()


class ReLU(Function):
    def __call__(self, *inputs_raw: Operatable) -> Variable:
        return super().__call__(*inputs_raw)

    def forward(self, *xs: np.ndarray) -> tuple[np.ndarray, ...]:
        x, = xs
        y = np.maximum(x, 0.0)
        return y,

    def backward(self, *gys: Variable) -> tuple[Variable, ...]:
        gy, = gys
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx,


def relu(x: Variable):
    return ReLU()(x)


def dropout(x: Variable | np.ndarray, dropout_ratio=0.5) -> tuple[Variable]:
    xv = as_variable(x)

    if dezero.Config.train:
        mask = np.random.rand(*x.shape) > dropout_ratio
        scale = np.array(1-dropout_ratio).astype(xv.dtype)
        y = x * mask / scale
        return y,
    else:
        return xv,
