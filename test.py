from Config import no_grad
from typing import Callable
import numpy as np
from Function import add, exp, mul, square
from Variable import Variable
from unittest import TestCase


def numercal_diff(f: Callable[[Variable], Variable], x: Variable, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class SquareTest(TestCase):
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


class TestAdd(TestCase):
    def test_forward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = add(x0, x1)
        self.assertEqual(y.data, np.array(5.0))

    def test_backward_sets_grad_on_inputs(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = add(x0, x1)
        y.backward()
        self.assertEqual(x0.grad, np.array(1.0))
        self.assertEqual(x1.grad, np.array(1.0))


class BranchedGraphDiffTest(TestCase):
    def test_two_branch_diff(self):
        x = Variable(np.array(2.0))
        a = square(x)
        y = add(square(a), square(a))
        y.backward()
        self.assertEqual(x.grad, np.array(64.0))


class DisableBackpropTest(TestCase):
    def test(self):
        with no_grad():
            x = Variable(np.array(10))
            y = exp(x)
            self.assertIsNone(y.creator)


class DontRetainGradTest(TestCase):
    def test_retain_grad_only_the_firts_inputs(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        t = add(x0, x1)
        y = add(x0, t)
        y.backward()
        self.assertIsNone(y.grad)
        self.assertIsNone(t.grad)
        self.assertEqual(x0.grad, 2.0)
        self.assertEqual(x1.grad, 1.0)


class VariableUtilityTest(TestCase):
    def test_can_specify_name(self):
        v = Variable(np.array(1.0), name="variable_name")
        self.assertEqual(v.name, "variable_name")

    def test_name_is_none_is_not_given(self):
        v = Variable(np.array(2.0))
        self.assertIsNone(v.name)

    def test_exist_properties(self):
        arr = np.array([[1, 2], [2, 3], [3, 4]])
        v = Variable(arr)
        self.assertEqual(v.shape, arr.shape)
        self.assertEqual(v.ndim, arr.ndim)
        self.assertEqual(v.size, arr.size)
        self.assertEqual(v.dtype, arr.dtype)
        self.assertEqual(len(v), len(arr))


class MulTest(TestCase):
    def test_forward(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        y = mul(a, b)
        self.assertEqual(y.data, np.array(6.0))

    def test_backward(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        y = mul(a, b)
        y.backward()

        self.assertEqual(a.grad, np.array(2.0))
        self.assertEqual(b.grad, np.array(3.0))


class VariableOverloadTest(TestCase):
    def test_variable_mul_variable(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(4.0))
        c = a * b
        self.assertEqual(c.data, np.array(6.0))
