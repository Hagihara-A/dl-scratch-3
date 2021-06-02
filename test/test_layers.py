from unittest import TestCase

import numpy as np
from dezero.layers import Linear


class LinearTest(TestCase):
    def test_forward(self):
        layer = Linear(10)
        x = np.random.rand(100, 20)
        y = layer(x)
        self.assertEqual(y.shape, (100, 10))

    def test_can_backward(self):
        layer = Linear(10)
        x = np.random.rand(100, 20)
        y = layer(x)
        y.backward()
