import numpy as np
from dezero.core import Variable
import dezero.functions as F
from dezero.utils import plot_dot_graph


def network():
    x = Variable(np.array(1.0), name="x")
    y = F.exp(x)
    y.name = "y"
    y.backward(create_graph=True)
    gx = x.grad
    gx.name = "gx"
    gx.backward(create_graph=True)
    return gx


if __name__ == "__main__":
    plot_dot_graph(network())
