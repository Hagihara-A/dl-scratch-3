from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable
import numpy as np
from dezero.core import Parameter
from dezero.models import Model


class Optimizer(ABC):
    def __init__(self) -> None:
        self.target: Model | None = None
        self.hooks: list[Callable[[Parameter], Any]] = []

    def setup(self, target: Model):
        self.target = target
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.hooks:
            f(params)

        for param in params:
            self.update_one(param)

    @abstractmethod
    def update_one(self, param: Parameter):
        pass

    def add_hook(self, hook):
        self.hooks.append(hook)


class SGD(Optimizer):
    def __init__(self, lr=0.01) -> None:
        super().__init__()
        self.lr = lr

    def update_one(self, param: Parameter):
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.999) -> None:
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs: dict[int, np.ndarray] = {}

    def update_one(self, param: Parameter):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v
