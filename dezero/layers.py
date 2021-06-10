from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generator, Optional
import weakref
from dezero.core import Parameter, Variable
import numpy as np
import dezero.functions as F
from dezero.utils import pair
import os


class Layer(ABC):
    def __init__(self) -> None:
        self._params: set[str] = set()

    def __setattr__(self, name: str, value: Parameter | Layer) -> None:
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs: np.ndarray):
        outputs = self.forward(*inputs)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(x) for x in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self, *xs: np.ndarray) -> tuple[Variable, ...]:
        pass

    def params(self) -> Generator[Parameter, None, None]:
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def clear_grads(self):
        for param in self.params():
            param.clear_grad()

    def _flatten_params(self, params_dict: dict[str, Parameter], parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name
            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path: str):
        params_dict: dict[str, Parameter] = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items()}
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise e

    def load_weights(self, path: str):
        npz = np.load(path)
        params_dict: dict[str, Parameter] = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


class Linear(Layer):
    def __call__(self, *inputs: np.ndarray) -> Variable:
        return super().__call__(*inputs)

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


class Conv2d(Layer):
    def __init__(self, out_channels: int, kernel_size: int, stride=1, pad=0,
                 nobias=False, dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name="W")
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name="b")

    def _init_W(self):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1/(C*KH*KW))
        W_data = np.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W_data = W_data

    def forward(self, *xs: np.ndarray) -> tuple[Variable, ...]:
        x, = xs
        if self.W_data is None:
            self.in_channels = x.shape[1]
            self._init_W()

        y = F.conv2d_simple(x, self.W, self.b, self.stride, self.pad)
        return y,

