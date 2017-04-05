#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

__author__ = "Yuji Ikeda"

import numpy as np


def read_band_yaml(yaml_file="band.yaml"):
    import yaml
    data = yaml.load(open(yaml_file, "r"))
    nqpoint = data['nqpoint']
    npath = data['npath']
    distance = np.zeros(nqpoint)
    frequency = []
    weight    = []
    for iqpoint in range(nqpoint):
        distance[iqpoint] = data['phonon'][iqpoint]['distance']
        nband = len(data['phonon'][iqpoint]['band'])
        f = np.zeros(nband)
        w = np.zeros(nband)
        for i in range(nband):
            f[i] = data['phonon'][iqpoint]['band'][i]['frequency']
            w[i] = data['phonon'][iqpoint]['band'][i]['weight']
        frequency.append(f)
        weight   .append(w)

    nsep = nqpoint // npath

    return distance, frequency, weight, nsep


def read_band_hdf5(hdf5_file="band.hdf5"):
    import h5py
    band_data = {}
    with h5py.File(hdf5_file, "r") as f:
        for key in f.keys():
            band_data[key] = np.array(f[key])
    return band_data


def read_input(filename_input):
    import json
    with open(filename_input, "r") as f:
        dict_input = json.load(f)
    return dict_input
