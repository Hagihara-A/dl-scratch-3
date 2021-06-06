from dezero.core import Parameter
from unittest import TestCase
from dezero.models import MLP
import numpy as np


class MLPTest(TestCase):
    def test_flatten_palams(self):
        model = MLP((10, 10, 10))
        x = np.array([[1.0]])
        model(x)
        d = {}
        model._flatten_params(d)
        self.assertIn('l0/W', d)
        self.assertIn('l0/b', d)
        self.assertIn('l1/W', d)
        self.assertIn('l1/b', d)
        self.assertIn('l2/W', d)
        self.assertIn('l2/b', d)

        for v in d.values():
            self.assertIsInstance(v, Parameter)
