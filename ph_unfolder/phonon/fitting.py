#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import os
import argparse
import numpy as np
from functions import lorentzian, gaussian
from scipy.optimize import curve_fit


class Fitting(object):
    def __init__(self, function_name=None):
        if function_name is not None:
            self.set_function(functions[function_name])

    def set_function(self, function):
        self._function = function

    def get_function(self):
        return self._function

    def run(self, xs, ys_list):
        """Fitting for each band.
        """
        fit_params_list = []
        for ys in ys_list:
            position_ini = xs[np.argmax(ys)]
            # "curve_fit" does not work well for extremely small initial guess.
            # To avoid this problem, "position_ini" is rounded.
            # See also "http://stackoverflow.com/questions/15624070"
            prec = 1.0e-12
            if abs(position_ini) < prec:
                position_ini = 0.0
            width_ini = 0.1
            p0 = [position_ini, width_ini]
            fit_params, pcov = curve_fit(
                self._fitting_function, xs, ys, p0=p0)
            fit_params_list.append(fit_params)
        fit_params_list = np.array(fit_params_list)
        return fit_params_list


def test():
    dx = 0.00001
    width = 0.001

    xmax = 1000
    xs = np.linspace(-xmax, xmax, int(np.rint(2 * xmax / dx)) + 1)
    print(len(xs))
    vs = lorentzian(xs, 0.0, width)

    print(np.sum(vs) * dx)


def print_fitted_values(fitting_function, xs, fit_params, f):
    for i, e in enumerate(fit_params[::2]):
        f.write("# eigvals_effective     ")
        f.write("{:6d}".format(i))
        f.write("{:12.6f}".format(e))
        f.write("\n")
    for x in xs:
        f.write("{:12.6f}{:12.6f}".format(x, fitting_function(x, *fit_params)))
        f.write("\n")


def lorentzians_3(x, position0, width0, position1, width1, position2, width2):
    return (
        lorentzian(x, position0, width0) +
        lorentzian(x, position1, width1) +
        lorentzian(x, position2, width2))


def run_band(filenames):
    for filename in filenames:
        print(filename)
        data = np.loadtxt(filename).T
        frequencies = data[0]
        probabilities = data[3:]
        values = []
        peaks = []
        for probability in probabilities:
            position = frequencies[np.argsort(probability)[-1]]
            p0 = [position, 0.1]
            fit_params, fit_covariances = curve_fit(
                lorentzian,
                frequencies,
                probability,
                p0=p0)
            values.append(lorentzian(frequencies, *fit_params))
            peaks.append(fit_params[0])

        filename_write = filename.replace(".sdat", ".sfbdat")
        with open(filename_write, "w") as f:
            print_fitted_values_band(peaks, frequencies, values, f)


def print_fitted_values_band(peaks, xs, values, f):
    for i, e in enumerate(peaks):
        f.write("# eigvals_effective     ")
        f.write("{:6d}".format(i))
        f.write("{:12.6f}".format(e))
        f.write("\n")
    for i, x in enumerate(xs):
        f.write("{:12.6f}".format(x))
        for v in values:
            f.write("{:12.6f}".format(v[i]))
        f.write("\n")


def run_total(filenames):
    for filename in filenames:
        filename_write = filename.replace(".sdat", ".sfdat")
        with open(filename_write, "w") as f:
            frequencies, probabilities = np.loadtxt(
                filename, usecols=(0, 1), unpack=True)
            (position0, position1, position2) = (
                frequencies[np.argsort(probabilities)[-3:]])
            p0 = [position0, 1.0, position1, 1.0, position2, 1.0]
            fit_params, fit_covariances = curve_fit(
                lorentzians_3,
                frequencies,
                probabilities,
                p0=p0)
            print_fitted_values(lorentzians_3, frequencies, fit_params, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filenames",
                        nargs="+",
                        type=str,
                        required=True,
                        help="filenames for fitting")
    parser.add_argument("--mode_band",
                        action="store_true",
                        help="Fitting for each band")
    args = parser.parse_args()

    if args.mode_band:
        run_band(filenames=args.filenames)
    else:
        run_total(filenames=args.filenames)


if __name__ == "__main__":
    main()
