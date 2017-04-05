#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

__author__ = "Yuji Ikeda"

import numpy as np
from unfolding.analysis.smearing import Smearing

xs = np.linspace(-10.0, 10.0, 21)
sigma = 1.0
mu = np.arange(1)

print(xs)
print(mu)

smearing = Smearing()
smearing.set_sigma(sigma)
smearing.set_xs(xs)
values = smearing.run(mu)

print(values)

weights = np.array([2.0])
values = smearing.run(mu, weights)

print(values)

smearing.build_xs(-2.5, 10.0, 0.1)
print(smearing.get_xs())
