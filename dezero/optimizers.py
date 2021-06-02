from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable
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
