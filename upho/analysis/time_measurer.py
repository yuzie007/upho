#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import time


class TimeMeasurer(object):
    """Measure method execution time.

    This class is made based on the suggestion in the following reference.
    http://preshing.com/20110924/timing-your-code-using-pythons-with-statement/
    """
    def __init__(self, time_string):
        self._time_string = time_string

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        time_string = self._time_string

        self._finish = time.time()
        interval = self._finish - self._start

        print('{:36s} (sec.):  {:12.4f}'.format(time_string, interval))
