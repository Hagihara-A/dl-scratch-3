from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from Function import Function


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
