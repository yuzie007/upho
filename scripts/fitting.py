#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"


from ph_unfolder.phonon.fitting import SFFitter

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("-f", "--filenames",
    #                     nargs="+",
    #                     type=str,
    #                     required=True,
    #                     help="filenames for fitting")
    # parser.add_argument("--mode_band",
    #                     action="store_true",
    #                     help="Fitting for each band")
    args = parser.parse_args()

    SFFitter()


if __name__ == "__main__":
    main()
