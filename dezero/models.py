from .layers import Layer
from .utils import plot_dot_graph


class Model(Layer):
    def plot(self, *inputs, to_file="modelpng"):
        y = self.forward(*inputs)
        return plot_dot_graph(y, verbose=True, to_file=to_file)
