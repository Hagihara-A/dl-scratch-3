from dezero.core import Variable
import numpy as np
from .layers import Layer, Linear, RNN
from .functions import sigmoid
from .utils import plot_dot_graph


class Model(Layer):
    def plot(self, *inputs, to_file="modelpng"):
        y = self.forward(*inputs)
        return plot_dot_graph(y, verbose=True, to_file=to_file)


class MLP(Model):
    def __init__(self, fc_output_sizes: tuple[int, ...], activation=sigmoid)\
            -> None:
        super().__init__()
        self.activation = activation
        self.layers: list[Linear] = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = Linear(out_size)
            setattr(self, f'l{str(i)}', layer)
            self.layers.append(layer)

    def forward(self, *xs: np.ndarray):
        x, = xs
        for ly in self.layers[:-1]:
            x = self.activation(ly(x))
        return self.layers[-1](x),


class SimpleRNN(Model):
    def __init__(self, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.rnn = RNN(hidden_size)
        self.fc = Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, *xs: np.ndarray) -> tuple[Variable, ...]:
        x, = xs
        h = self.rnn(x)
        y = self.fc(h)
        return y,
