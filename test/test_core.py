from typing import Callable
from unittest import TestCase
from numpy.testing import assert_equal
import numpy as np
from dezero.config import no_grad
from dezero.core import Variable, add, as_variable, div, mul, pow, sub
from dezero.functions import cos, exp, sin, square


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
        self.assertEqual(x.grad.data, expected)

    def test_backprop_with_numercal_diff(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numercal_diff(square, x)
        flg = np.allclose(x.grad.data, num_grad.data)
        self.assertTrue(flg)


class AddTest(TestCase):
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
        self.assertEqual(x0.grad.data, np.array(1.0))
        self.assertEqual(x1.grad.data, np.array(1.0))

    def test_backward_when_broadcasted(self):
        x0 = Variable(np.array([10, 11, 12]))
        x1 = Variable(np.array([5]))
        y = x0 + x1
        y.backward()
        assert_equal(x0.grad.data, np.ones((3,)))
        assert_equal(x1.grad.data, np.array([3]))


class BranchedGraphDiffTest(TestCase):
    def test_two_branch_diff(self):
        x = Variable(np.array(2.0))
        a = square(x)
        y = add(square(a), square(a))
        y.backward()
        self.assertEqual(x.grad.data, np.array(64.0))


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
        self.assertEqual(x0.grad.data, 2.0)
        self.assertEqual(x1.grad.data, 1.0)


class VariableUtilityTest(TestCase):
    def test_can_specify_name(self):
        v = Variable(np.array(1.0), name="variable_name")
        self.assertEqual(v.name, "variable_name")

    def test_name_is_none_is_not_given(self):
        v = Variable(np.array(2.0))
        self.assertEqual(v.name, "")

    def test_exist_properties(self):
        arr = np.array([[1, 2], [2, 3], [3, 4]])
        v = Variable(arr)
        self.assertEqual(v.shape, arr.shape)
        self.assertEqual(v.ndim, arr.ndim)
        self.assertEqual(v.size, arr.size)
        self.assertEqual(v.dtype, arr.dtype)
        self.assertEqual(len(v), len(arr))

    def test_reshape(self):
        x = Variable(np.arange(12).reshape(3, 4))
        y = x.reshape((6, 2))
        self.assertEqual(y.shape, (6, 2))

    def test_transpose(self):
        x = Variable(np.arange(12).reshape(3, 4))
        y = x.T
        self.assertEqual(y.shape, (4, 3))


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

        self.assertEqual(a.grad.data, np.array(2.0))
        self.assertEqual(b.grad.data, np.array(3.0))

    def test_backward_with_broadcast(self):
        x0 = Variable(np.array([10, 11, 12]))
        x1 = Variable(np.array([5]))
        y = mul(x0, x1)
        y.backward()
        assert_equal(x0.grad.data, np.array([5, 5, 5]))
        assert_equal(x1.grad.data, np.array([10+11+12]))


class PowTest(TestCase):
    def test_forward(self):
        a = Variable(np.array(3.0))
        b = pow(a, 4)
        self.assertEqual(b.data, 81.0)

    def test_backward(self):
        a = Variable(np.array(3.0))
        b = pow(a, 4)
        b.backward()
        self.assertEqual(a.grad.data, 108.0)

    def test_backward_with_broadcast(self):
        x0 = Variable(np.array([10, 11, 12]))
        y = pow(x0, 3)
        y.backward()
        assert_equal(x0.grad.data, np.array([3*10**2, 3*11**2, 3*12**2]))


class SubTest(TestCase):
    def test_forward(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        y = sub(a, b)
        self.assertEqual(y.data, np.array(1.0))

    def test_backward(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        y = sub(a, b)
        y.backward()
        self.assertEqual(a.grad.data, np.array(1.0))
        self.assertEqual(b.grad.data, np.array(-1.0))

    def test_forward_with_broadcast(self):
        x0 = Variable(np.arange(100).reshape((100, 1)))
        x1 = Variable(np.ones((100, 1)))
        y = sub(x0, x1)
        assert_equal(y.data, np.arange(-1, 99).reshape(100, 1))

    def test_backward_with_broadcast(self):
        x0 = Variable(np.array([10, 11, 12]))
        x1 = Variable(np.array([5]))
        y = sub(x0, x1)
        y.backward()
        assert_equal(x0.grad.data, np.array([1, 1, 1]))
        assert_equal(x1.grad.data, np.array([-3]))


class DivTest(TestCase):
    def test_forward(self):
        a = as_variable(6.0)
        b = as_variable(4.0)
        c = div(a, b)
        self.assertEqual(c.data, np.array(1.5))

    def test_backward(self):
        a = as_variable(6.0)
        b = as_variable(4.0)
        c = div(a, b)
        c.backward()
        self.assertEqual(a.grad.data, np.array(1/4))
        self.assertEqual(b.grad.data, np.array(-6/4**2))

    def test_backward_with_broadcast(self):
        x0 = Variable(np.array([10, 11, 12]))
        x1 = Variable(np.array([2]))
        y = div(x0, x1)
        y.backward()
        assert_equal(x0.grad.data, np.array([0.5, 0.5, 0.5]))
        assert_equal(x1.grad.data, np.array([-10/4-11/4-12/4]))


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

    def test_ndarray_sub_w_broadcast(self):
        x0 = np.arange(10).reshape(2, 5)
        x1 = Variable(np.array(2))
        y = x0 + x1
        assert_equal(y.data, np.arange(2, 12).reshape(2, 5))


class ComplexGraphDiffTest(TestCase):
    def test_matyas(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = 0.26*(x**2+y**2)-0.48*x*y
        z.backward()
        self.assertAlmostEqual(x.grad.data, 0.04, delta=1e-6)
        self.assertAlmostEqual(y.grad.data, 0.04, delta=1e-6)

    def test_goldstein(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = (1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2)) * \
            (30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))
        z.backward()
        self.assertEqual(x.grad.data, -5376)
        self.assertEqual(y.grad.data, 8064)


class SinTest(TestCase):
    def test_forward(self):
        x = Variable(np.array(np.pi / 4))
        y = sin(x)
        self.assertAlmostEqual(y.data, 1/np.sqrt(2))

    def test_backward(self):
        x = Variable(np.array(np.pi / 4))
        y = sin(x)
        y.backward()
        self.assertAlmostEqual(x.grad.data, 1 / np.sqrt(2))


class CosTest(TestCase):
    def test_forward(self):
        x = Variable(np.array(np.pi / 6))
        y = cos(x)
        self.assertAlmostEqual(y.data, np.array(np.sqrt(3)/2))

    def test_backward(self):
        x = Variable(np.array(np.pi / 6))
        y = cos(x)
        y.backward()
        self.assertAlmostEqual(x.grad.data, np.array(-1/2))


class TwoOrderDiffTest(TestCase):
    def test_two_order_diff(self):
        x = Variable(np.array(2.0))
        y = x**4 - 2*x**2
        y.backward(create_graph=True)
        gx = x.grad
        x.clear_grad()
        self.assertEqual(gx.data, np.array(24.0))
        gx.backward()
        self.assertEqual(x.grad.data, np.array(44.0))
