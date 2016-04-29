#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

from functools import reduce


def gcd2(a, b):
    while b:
        a, b = b, a % b
    return a


def lcm2(a, b):
    return (a * b) // gcd2(a, b)


def gcd(*numbers):
    return reduce(gcd2, numbers)


def lcm(*numbers):
    return reduce(lcm2, numbers)
