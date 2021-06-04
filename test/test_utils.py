from unittest import TestCase
import numpy as np
from dezero.core import Variable
from dezero.functions import square
from dezero.utils import _dot_var, get_dot_graph, to_onehot
from numpy.testing import assert_equal


class TestGetDotGraph(TestCase):
    def test__dot_var(self):
        a = Variable(np.array(1.0))
        b = Variable(np.array(1.0))
        c = a + b
        d = square(c)

        actual = get_dot_graph(d)
        self.assertIn(f"{id(d.creator)} -> {id(d)}", actual)
        self.assertIn(f"{id(c)} -> {id(d.creator)}", actual)


class Test_dot_var(TestCase):
    def test_no_name_variable_when_verbose_is_false(self):
        a = Variable(np.array(1.0))
        txt = _dot_var(a)
        self.assertEqual(
            f'{id(a)} [label="", color=orange, style=filled]\n', txt)

    def test_no_name_when_verbose_is_true(self):
        a = Variable(np.array(1.0))
        txt = _dot_var(a, verbose=True)
        self.assertEqual(
            f'{id(a)} [label="{a.shape} {a.dtype}", color=orange, style=filled]\n', txt)


class ToOnehotTest(TestCase):
    def test_1d(self):
        t = np.array([2, 1, 3, 0])
        t_onehot = to_onehot(t, labels=4)
        expected = [[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
        assert_equal(t_onehot, expected)
