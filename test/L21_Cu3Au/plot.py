#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import numpy as np
import matplotlib.pyplot as plt


def main():
    distances, frequencies, sfs = np.loadtxt('sf.dat', unpack=True)
    n2 = len(np.unique(frequencies))
    distances   = distances.reshape((-1, n2))
    frequencies = frequencies.reshape((-1, n2))
    sfs         = sfs.reshape((-1, n2))

    fig, ax = plt.subplots(
        1, 1, figsize=(5.0, 3.0), frameon=False, tight_layout=True)
    ax.pcolormesh(distances, frequencies, sfs, cmap='RdBu', rasterized=True)
    fig.savefig('sf.pdf', transparent=True)

if __name__ == "__main__":
    main()
