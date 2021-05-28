import numpy as np
from dezero.core import Variable
from dezero.functions import broadcast_to, matmul, reshape, sum_to, tanh, transpose, sum
from unittest import TestCase
from numpy.testing import assert_equal


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


class SumTest(TestCase):
    def test_forward(self):
        x = Variable(np.arange(6).reshape(2, 3))
        y = sum(x)
        self.assertEqual(y.data, np.array(15))

    def test_backward(self):
        x = Variable(np.arange(6).reshape(2, 3))
        y = sum(x)
        y.backward()
        assert_equal(x.grad.data, np.ones_like(x.data))


class BroadcastToTest(TestCase):
    def test_forward(self):
        x = Variable(np.array([1, 2, 3]))
        y = broadcast_to(x, (2, 3))
        self.assertEqual(y.data.shape, (2, 3))
        assert_equal(np.array([[1, 2, 3], [1, 2, 3]]), y.data)

    def test_backward(self):
        x = Variable(np.array([1, 2, 3]))
        y = broadcast_to(x, (2, 3))
        y.backward()
        self.assertEqual(x.grad.shape, (3,))
        assert_equal(x.grad.data, np.array([2, 2, 2]))


class SumToTest(TestCase):
    def test_forward(self):
        x = Variable(np.arange(12).reshape(3, 4))
        y = sum_to(x, (1, 4))
        assert_equal(y.data, np.array([[12, 15, 18, 21]]))

    def test_backward(self):
        x = Variable(np.arange(12).reshape(3, 4))
        y = sum_to(x, (1, 4))
        y.backward()
        assert_equal(x.grad.data, np.ones((3, 4)))


class MatMulTest(TestCase):
    def test_forward(self):
        x = Variable(np.arange(4).reshape(2, 2))
        W = Variable(np.arange(6).reshape(2, 3))
        y = matmul(x, W)
        assert_equal(y.data, np.array([[3, 4, 5], [9, 14, 19]]))

    def test_backward(self):
        x = Variable(np.arange(4).reshape(2, 2))
        W = Variable(np.arange(6).reshape(2, 3))
        y = matmul(x, W)
        y.backward()
