from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from Function import Function


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.grad: Optional[np.ndarray] = None
        self.__creator: Optional[Function] = None

    @property
    def creator(self):
        return self.__creator

    @creator.setter
    def creator(self, creator: Function):
        self.__creator = creator

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            for x, gx in zip(f.inputs, gxs):
                x.grad = gx
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)
