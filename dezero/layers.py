from abc import ABC, abstractmethod
from typing import Optional
import weakref
from dezero.core import Parameter
import numpy as np
import dezero.functions as F


class Layer(ABC):
    def __init__(self) -> None:
        self._params: set[str] = set()

    def __setattr__(self, name: str, value) -> None:
        if isinstance(value, Parameter):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs: np.ndarray):
        outputs = self.forward(*inputs)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(x) for x in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self, *xs: np.ndarray):
        pass

    def params(self):
        for name in self._params:
            yield self.__dict__[name]

    def clear_grads(self):
        for param in self.params():
            param.clear_grad()


class Linear(Layer):
    def __init__(self, out_size: int, nobias: bool = False, dtype=np.float32,
                 in_size: Optional[int] = None) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        self.W = Parameter(None, name="W")
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self):
        IN, OUT = self.in_size, self.out_size
        W_data = np.random.randn(IN, OUT).astype(self.dtype) * np.sqrt(1/IN)
        self.W.data = W_data

    def forward(self, *xs: np.ndarray):
        x, = xs
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()
        y = F.linear(x, self.W, self.b)
        return y,
