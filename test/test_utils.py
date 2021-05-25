import numpy as np
from dezero.core import Variable, square
from dezero.utils import get_dot_graph
from unittest import TestCase


class TestGetDotGraph(TestCase):
    def test__dot_var(self):
        a = Variable(np.array(1.0))
        b = Variable(np.array(1.0))
        c = a + b
        d = square(c)

        actual = get_dot_graph(d)
        self.assertIn(f"{id(d.creator)} -> {id(d)}", actual)
        self.assertIn(f"{id(c)} -> {id(d.creator)}", actual)
