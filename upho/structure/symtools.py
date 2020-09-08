#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
import spglib


def get_rotations_cart(atoms):
    cell = atoms.get_cell()
    dataset = spglib.get_symmetry_dataset(atoms)
    rotations = dataset["rotations"]
    translations = dataset["translations"]

    rotations_cart = [
        np.dot(np.dot(cell.T, r), np.linalg.inv(cell.T)) for r in rotations
    ]
    rotations_cart = np.array(rotations_cart)

    return rotations_cart
