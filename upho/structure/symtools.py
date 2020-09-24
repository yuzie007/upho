import numpy as np
import spglib


def get_rotations_cart(atoms):
    cell = atoms.get_cell()
    dataset = spglib.get_symmetry_dataset(atoms)
    rotations = dataset["rotations"]

    rotations_cart = [
        np.dot(np.dot(cell.T, r), np.linalg.inv(cell.T)) for r in rotations
    ]
    rotations_cart = np.array(rotations_cart)

    return rotations_cart
