#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import numpy as np


def lorentzian_old(x, position, width):
    return 1.0 / (np.pi * width * (1.0 + ((x - position) / width) ** 2))


def lorentzian(x, position, width):
    """This is faster than lorentzian_old"""
    return width / (np.pi * (width ** 2 + (x - position) ** 2))


def lorentzian_unnormalized(x, position, width, norm):
    return norm * lorentzian(x, position, width)


def gaussian(x, position, width):
    sigma = width / np.sqrt(2.0 * np.log(2.0))
    tmp = np.exp(- (x - position) ** 2 / (2.0 * sigma ** 2))
    return 1.0 / np.sqrt(2.0 * np.pi) / sigma * tmp
