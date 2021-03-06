import dezero
from numpy.testing import assert_almost_equal, assert_equal
from unittest import TestCase
import numpy as np
from dezero.core import Variable
from dezero.functions import broadcast_to, get_item, linear, log, matmul,\
    mean_squared_error, reshape, sigmoid, softmax, sum_to, tanh, transpose, sum
import dezero.functions as F


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


class MSETest(TestCase):
    def test_forward(self):
        x0 = Variable(np.arange(10))
        x1 = np.arange(2, 12)
        L = mean_squared_error(x0, x1)
        self.assertEqual(L.data, np.array(4))

    def test_backward(self):
        x0 = Variable(np.arange(10))
        x1 = np.arange(2, 12)
        L = mean_squared_error(x0, x1)
        L.backward()
        assert_equal(x0.grad.data, 2/10*(x0.data - x1))

    def test_forward_2D(self):
        x0 = Variable(np.arange(10).reshape(2, 5))
        x1 = np.arange(2, 12).reshape(2, 5)
        L = mean_squared_error(x0, x1)
        self.assertEqual(L.data, 20)

    def test_backward_2D(self):
        x0 = Variable(np.arange(10).reshape(2, 5))
        x1 = np.arange(2, 12).reshape(2, 5)
        L = mean_squared_error(x0, x1)
        L.backward()
        assert_equal(x0.grad.data, 2/2*(x0.data - x1))


class LinearTest(TestCase):
    def test_forward_W_bias(self):
        x = Variable(np.array([[1, 2, 3], [6, 7, 8]]))
        W = Variable(np.arange(1, 7).reshape(3, 2))
        b = Variable(np.array(5))
        y = linear(x, W, b)
        assert_equal(np.array([[27, 33], [72, 93]]), y.data)

    def test_forward_WO_bias(self):
        x = Variable(np.array([[1, 2, 3], [6, 7, 8]]))
        W = Variable(np.arange(1, 7).reshape(3, 2))
        y = linear(x, W)
        assert_equal(np.array([[22, 28], [67, 88]]), y.data)

    def test_backward_W_bias(self):
        x = Variable(np.array([[1, 2, 3], [6, 7, 8]]))
        W = Variable(np.arange(1, 7).reshape(3, 2))
        b = Variable(np.array(5))
        y = linear(x, W, b)
        y.backward()
        assert_equal(x.grad.data, np.ones((2, 2)) @ W.data.T)
        assert_equal(W.grad.data, x.data.T @ np.ones((2, 2)))
        assert_equal(b.grad.data, 4)

    def test_backward_WO_bias(self):
        x = Variable(np.array([[1, 2, 3], [6, 7, 8]]))
        W = Variable(np.arange(1, 7).reshape(3, 2))
        y = linear(x, W)
        y.backward()
        assert_equal(x.grad.data, np.ones((2, 2)) @ W.data.T)
        assert_equal(W.grad.data, x.data.T @ np.ones((2, 2)))
        self.assertIsNone(y.creator.inputs[2].grad)


class SigmoidTest(TestCase):
    def test_forward(self):
        x = Variable(np.array([2.0, 2.0]))
        y = sigmoid(x)
        assert_almost_equal(y.data, np.array([0.88079707, 0.88079707]))

    def test_backward(self):
        out = 0.88079707
        x = Variable(np.array([2.0, 2.0]))
        y = sigmoid(x)
        y.backward()
        assert_almost_equal(x.grad.data, [(1-out)*out]*2)


class GetItemTest(TestCase):
    def test_forward(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        ind = [1, 1, 0]
        y = get_item(x, ind)
        assert_equal(y.data, [[4, 5, 6], [4, 5, 6], [1, 2, 3]])

    def test_backward(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        ind = [1, 1, 0]
        y = get_item(x, ind)
        y.backward()
        assert_equal(x.grad.data, [[1, 1, 1], [2, 2, 2]])


class SoftmaxTest(TestCase):
    def test_forward(self):
        x = Variable(np.array([[1, 6, 4, 1, 2]]*2))
        y = softmax(x)
        assert_almost_equal(y.data,
                            [[0.00577311, 0.85680492,
                             0.11595594, 0.00577311, 0.01569293]]*2)

    def test_backward(self):
        x = Variable(np.array([[1, 6, 4, 1, 2]]*2))
        y = softmax(x)
        y.backward()
        y_expected = np.array([[0.00577311, 0.85680492,
                               0.11595594, 0.00577311, 0.01569293]]*2)

        gx_expected = y_expected * (1 - y_expected)
        assert_almost_equal(x.grad.data, gx_expected)


class LogTest(TestCase):
    def test_forward(self):
        x = Variable(np.array([0.5, 1, 6]))
        y = log(x)
        assert_almost_equal(y.data, [-0.69314718055, 0., 1.791759469])

    def test_backward(self):
        x = Variable(np.array([0.5, 1, 6]))
        y = log(x)
        y.backward()
        gx_expected = np.array(1/x.data)
        assert_almost_equal(x.grad.data, gx_expected)


class SoftMaxCrossEntropyTest(TestCase):
    def test_forward(self):
        x = Variable(
            np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]]))
        t = np.array([1, 0, 1, 0])
        y = F.softmax_cross_entropy(x, t)
        y_expected = np.exp(x.data) / \
            np.exp(x.data).sum(axis=1, keepdims=True)
        y_expected = -np.log(y_expected)
        y_expected = y_expected[np.arange(4), t].sum() / 4
        assert_almost_equal(y.data, y_expected, decimal=5)

    def test_backward(self):
        x = Variable(
            np.array([[0.2, -0.4], [0.3, 0.5], [1., -3.2], [2.1, 0.3]]))
        t = np.array([1, 0, 1, 0])
        t_onehot = np.eye(2)[t]
        L = F.softmax_cross_entropy(x, t)
        L.backward()
        m = len(x.data)

        y = np.exp(x.data) / np.exp(x.data).sum(axis=1, keepdims=True)
        gx_expected = 1/m * (y - t_onehot)
        assert_almost_equal(x.grad.data, gx_expected)

    def test_dont_overflow(self):
        x = Variable(np.array([[1.0, 100.0, 10000.0]]))
        t = np.array([0])
        L = F.softmax_cross_entropy(x, t)
        self.assertFalse(np.isnan(L.data) or np.isinf(L.data))

    def test_dont_divide_by_zero(self):
        x = np.array([[1., 0., 300000., 400000.]])
        t = np.array([1])
        L = F.softmax_cross_entropy(x, t)
        self.assertFalse(np.isnan(L.data) or np.isinf(L.data))


class Test__softmax(TestCase):
    def test_small_x(self):
        x = np.array([[1, 2, 3, 4]])
        y = F._softmax(x)
        y_expected = np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)
        assert_almost_equal(y, y_expected)

    def test_big_input_dont_overflow(self):
        x = np.array([[100000, 200000, 300000, 400000]])
        y = F._softmax(x)
        self.assertFalse(np.isnan(y).any() or np.isinf(y).any())


class ReLUTest(TestCase):
    def test_forward(self):
        x = np.array([[1, 0, -1, 5], [-1, -2, 5, 0]])
        y = F.relu(x)
        assert_equal(y.data, [[1, 0, 0, 5], [0, 0, 5, 0]])

    def test_backward(self):
        x = Variable(np.array([[1, 0, -1, 5], [-1, -2, 5, 0]]))
        y = F.relu(x)
        y.backward()
        assert_equal(x.grad.data, [[1, 0, 0, 1], [0, 0, 1, 0]])


class DropoutTest(TestCase):
    def test_forward_on_train(self):
        np.random.seed(2)
        x = np.array([[0., 1.], [2., 3.]])
        mask = np.array([[False, False], [True, False]])
        y, = F.dropout(x)
        assert_equal(y.data, x*mask/0.5)

    def test_forward_on_infer(self):
        np.random.seed(2)
        x = np.array([[0., 1.], [2., 3.]])
        with dezero.config.test_mode():
            y, = F.dropout(x)
            assert_equal(y.data, x)

    def test_backward_on_train(self):
        np.random.seed(2)
        x = Variable(np.array([[0., 1.], [2., 3.]]))
        y, = F.dropout(x)
        y.backward()
        assert_equal(x.grad.data, [[0., 0.], [2., 0.]])
