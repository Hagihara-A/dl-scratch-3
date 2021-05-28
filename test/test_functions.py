import numpy as np
from dezero.core import Variable
from dezero.functions import reshape, tanh, transpose
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


class ReshapeTest(TestCase):
    def test_forward(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = reshape(x, (6,))
        self.assertEqual(y.shape, (6,))

    def test_backward(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = reshape(x, (6,))
        y.backward(retain_grad=True)
        self.assertEqual(y.grad.shape, (6,))
        self.assertEqual(x.grad.shape, x.shape)
        self.assertTrue((x.grad.data == np.ones((2, 3))).all())


class TransposeTest(TestCase):
    def test_forward(self):
        x = Variable(np.arange(6).reshape(2, 3))
        y = transpose(x)
        self.assertEqual(y.shape, (3, 2))

    def test_backward(self):
        x = Variable(np.arange(6).reshape(2, 3))
        y = transpose(x)
        y.backward()
        self.assertEqual(x.grad.shape, x.shape)
