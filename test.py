import unittest
from typing import Callable
import numpy as np
from Function import add, square
from Variable import Variable


def numercal_diff(f: Callable[[Variable], Variable], x: Variable, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class SquareTest(unittest.TestCase):
    def test_foreard(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_backprop_with_numercal_diff(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numercal_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


class BranchedGraphDiffTest(unittest.TestCase):
    def test_two_branch_diff(self):
        x = Variable(np.array(2.0))
        a = square(x)
        y = add(square(a), square(a))
        y.backward()
        self.assertEqual(x.grad, np.array(64.0))
