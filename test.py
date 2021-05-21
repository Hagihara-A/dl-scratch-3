from typing import Callable
from unittest import TestCase
import numpy as np
from Config import no_grad
from Core import as_variable, div, pow, Variable, add, exp, mul, square


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


class PowTest(TestCase):
    def test_backward(self):
        a = Variable(np.array(3.0))
        b = pow(a, 4)
        b.backward()
        self.assertEqual(a.grad, 108.0)


class DivTest(TestCase):
    def test_backward(self):
        a = as_variable(6.0)
        b = as_variable(4.0)
        c = div(a, b)
        c.backward()
        self.assertEqual(a.grad, np.array(1/4))
        self.assertEqual(b.grad, np.array(-6/4**2))


class VariableOverloadTest(TestCase):
    def test_variable_mul_variable(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(4.0))
        c = a * b
        self.assertEqual(c.data, np.array(12.0))

    def test_variable_add_variable(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(4.0))
        c = a + b
        self.assertEqual(c.data, np.array(7.0))

    def test_variable_add_ndarray(self):
        a = Variable(np.array(3.0))
        b = np.array(4.0)
        c = a + b
        self.assertIsInstance(c, Variable)
        self.assertEqual(c.data, np.array(7.0))

    def test_variable_add_float(self):
        a = Variable(np.array(3.0))
        b = 4.0
        c = a + b
        self.assertIsInstance(c, Variable)
        self.assertEqual(c.data, np.array(7.0))

    def test_variable_mul_float(self):
        a = Variable(np.array(3.0))
        b = 4.0
        c = a * b
        self.assertIsInstance(c, Variable)
        self.assertEqual(c.data, np.array(12.0))

    def test_float_add_variable(self):
        a = 3.0
        b = Variable(np.array(4.0))
        c = a + b
        self.assertIsInstance(c, Variable)
        self.assertEqual(c.data, np.array(7.0))

    def test_float_mul_variable(self):
        a = 3.0
        b = Variable(np.array(4.0))
        c = a * b
        self.assertIsInstance(c, Variable)
        self.assertEqual(c.data, np.array(12.0))

    def test_ndarray_add_variable(self):
        a = np.array(3.0)
        b = Variable(np.array(4.0))
        c = a + b
        self.assertIsInstance(c, Variable)
        self.assertEqual(c.data, np.array(7.0))

    def test_ndarray_mul_variable(self):
        a = np.array(3.0)
        b = Variable(np.array(4.0))
        c = a * b
        self.assertIsInstance(c, Variable)
        self.assertEqual(c.data, np.array(12.0))

    def test_neg_variable(self):
        a = Variable(np.array(3.0))
        b = -a
        self.assertIsInstance(b, Variable)
        self.assertEqual(b.data, np.array(-3.0))

    def test_Variable_sub_Variable(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(4.0))
        c = a - b
        self.assertEqual(c.data, np.array(-1.0))

    def test_Variable_sub_float(self):
        a = Variable(np.array(3.0))
        b = 4.0
        c = a - b
        self.assertEqual(c.data, np.array(-1.0))

    def test_float_sub_Variable(self):
        a = 3.0
        b = Variable(np.array(4.0))
        c = a - b
        self.assertEqual(c.data, np.array(-1.0))

    def test_ndarray_sub_Variable(self):
        a = np.array(3.0)
        b = Variable(np.array(4.0))
        c = a - b
        self.assertEqual(c.data, np.array(-1.0))

    def test_Variable_sub_ndarray(self):
        a = Variable(np.array(3.0))
        b = np.array(4.0)
        c = a - b
        self.assertEqual(c.data, np.array(-1.0))

    def test_Variable_div_Variable(self):
        a = Variable(np.array(6.0))
        b = Variable(np.array(4.0))
        c = a / b
        self.assertEqual(c.data, np.array(1.5))

    def test_Variable_div_float(self):
        a = Variable(np.array(6.0))
        b = 4.0
        c = a / b
        self.assertEqual(c.data, np.array(1.5))

    def test_float_div_Variable(self):
        a = 6.0
        b = Variable(np.array(4.0))
        c = a / b
        self.assertEqual(c.data, np.array(1.5))

    def test_ndarray_div_Variable(self):
        a = np.array(6.0)
        b = Variable(np.array(4.0))
        c = a / b
        self.assertEqual(c.data, np.array(1.5))

    def test_Variable_div_ndarray(self):
        a = Variable(np.array(6.0))
        b = np.array(4.0)
        c = a / b
        self.assertEqual(c.data, np.array(1.5))

    def test_Variable_pow_3(self):
        a = Variable(np.array(3.0))
        b = a ** 3
        self.assertEqual(b.data, np.array(27.0))
