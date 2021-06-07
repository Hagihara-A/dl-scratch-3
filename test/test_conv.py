from numpy.testing import assert_equal
import numpy as np
from dezero.core import Variable
import dezero.functions_conv as C
from unittest import TestCase


class Conv2dSimpleTest(TestCase):
    def test_forward(self):
        x = Variable(np.arange(2 * 3 * 3).reshape(1, 2, 3, 3))
        kernel = Variable(np.array([[
            [[1, 2],
             [2, 1]],
            [[0, 1],
             [0, 1]]
        ]]))
        y = C.conv2d_simple(x, kernel, pad=1)
        a = np.array([[[[0, 1, 4, 4], [3, 12, 18, 12], [
                     12, 30, 36, 21], [12, 20, 23, 8]]]])
        b = np.array([[[[9, 10, 11, 0], [21, 23, 25, 0],
                     [27, 29, 31, 0], [15, 16, 17, 0]]]])

        assert_equal(y.shape, (1, 1, 4, 4))
        assert_equal(y.data, a+b)

    def test_backward(self):
        x = Variable(np.arange(2 * 3 * 3).reshape(1, 2, 3, 3))
        kernel = Variable(np.array([[
            [[1, 2],
             [2, 1]],
            [[0, 1],
             [0, 1]]
        ]]))
        y = C.conv2d_simple(x, kernel, pad=1)
        y.backward()
        assert_equal(x.grad.shape, x.shape)
        assert_equal(x.grad.data, [[[[6, 6, 6],
                                     [6, 6, 6],
                                     [6, 6, 6]],

                                    [[2, 2, 2],
                                     [2, 2, 2],
                                     [2, 2, 2]]]])
