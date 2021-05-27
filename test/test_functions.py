import numpy as np
from dezero.core import Variable
from dezero.functions import tanh
from unittest import TestCase


class TanhTest(TestCase):
    def test_forward(self):
        x = Variable(np.array(0.5))
        y = tanh(x)
        self.assertAlmostEqual(y.data, np.array(0.4621171572))

    def test_backward(self):
        x = Variable(np.array(0.5))
        y = tanh(x)
        y.backward()
        self.assertAlmostEqual(x.grad.data, np.array(0.7864477330213905))
