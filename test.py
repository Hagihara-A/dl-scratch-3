from Function import square
import numpy as np
import unittest
from Variable import Variable


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array([2.0]))
        y = square(x)
        expected = np.array([4.0])
        self.assertEqual(y.data, expected)
