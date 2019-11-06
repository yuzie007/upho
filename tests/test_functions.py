#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import unittest
import numpy as np
from upho.analysis.functions import lorentzian_unnormalized


class TestFunctions(unittest.TestCase):
    def test_lorentzian_unnnormalized(self):
        # norms = np.linspace(0.0, 10.0, 101)
        norms = np.linspace(1.0, 1.0, 1)
        widths = np.linspace(0.01, 5.0, 500)
        # peak_positions = np.linspace(-2.0, 2.0, 401)
        peak_positions = np.linspace(0.0, 0.0, 1)
        xs = np.linspace(-500.0, 500.0, 100001)
        dx = xs[1] - xs[0]
        prec = 1e-2
        for peak_position in peak_positions:
            for width in widths:
                for norm in norms:
                    ys = lorentzian_unnormalized(xs, peak_position, width, norm)
                    norm_integration = np.sum(ys) * dx
                    ratio = norm_integration / norm
                    print('{:12.3f}'.format(peak_position   ), end='')
                    print('{:12.3f}'.format(width           ), end='')
                    print('{:12.6f}'.format(norm            ), end='')
                    print('{:12.6f}'.format(norm_integration), end='')
                    print('{:12.6f}'.format(ratio           ), end='')
                    print()
                    if not np.isnan(ratio):
                        self.assertTrue(np.abs(ratio - 1.0) < prec)
                

if __name__ == "__main__":
    unittest.main()
