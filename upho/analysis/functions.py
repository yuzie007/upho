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


def gaussian_unnormalized(x, position, width, norm):
    return norm * gaussian(x, position, width)


class FittingFunctionFactory(object):
    def __init__(self, name, is_normalized):
        """

        Parameters
        ----------
        name : str
        is_normalized : bool
        """
        self._name = name
        self._is_normalized = is_normalized

    def create(self):
        name = self._name
        is_normalized = self._is_normalized
        if name == 'lorentzian':
            if is_normalized:
                return lorentzian
            else:
                return lorentzian_unnormalized
        elif name == 'gaussian':
            if is_normalized:
                return gaussian
            else:
                return gaussian_unnormalized
        else:
            raise ValueError('Unknown name', name)
