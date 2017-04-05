#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

__author__ = "Yuji Ikeda"

import sys
import itertools
import numpy as np
from phonopy.structure.symmetry import Symmetry


class StructureAnalyzer(object):
    def __init__(self, atoms):
        self._atoms = atoms
        self._filename = None
        self._dictionary = {}

    def update_attributes(self):
        """Update attributes of the class Poscar from atoms attribute.

        This method should be called after tne update of atoms attribute.
        """
        self.generate_dictionary()
        self.deform_cell = self.deform_cell_right  # alias

    def generate_dictionary(self):

        cell = self._atoms.get_cell()
        number_of_atoms = self._atoms.get_number_of_atoms()
        chemical_symbols = self._atoms.get_chemical_symbols()

        dictionary = convert_cell_to_lc(cell)

        volume = dictionary["volume"]

        dictionary.update({
            "filename": self._filename,
            "number_of_atoms": number_of_atoms,
            "chemical_symbols": chemical_symbols,
            "volume_per_atom": volume / number_of_atoms,
        })

        self._dictionary = dictionary

        self.create_symmetry_dataset()

        return self

    def create_symmetry_dataset(self):
        symmetry_dataset = self.get_symmetry_dataset()
        self._dictionary.update({
            "spg_number": symmetry_dataset["number"],
            "spg_international": symmetry_dataset["international"],
        })

    def generate_distance_matrix(self):

        cell = self._atoms.get_cell()
        scaled_positions = self._atoms.get_scaled_positions()
        number_of_atoms = self._atoms.get_number_of_atoms()

        expansion = range(-1, 2)
        distance_matrix = np.zeros((number_of_atoms, number_of_atoms))
        distance_matrix *= np.nan  # initialization
        scaled_distances = np.zeros((number_of_atoms, number_of_atoms, 3))
        scaled_distances *= np.nan  # initialization
        for i1, p1 in enumerate(scaled_positions):
            for i2, p2 in enumerate(scaled_positions):
                distance = 100000  # np.inf
                for addition in itertools.product(expansion, repeat=3):
                    scaled_distance_new = p2 - p1
                    scaled_distance_new -= np.rint(scaled_distance_new)
                    scaled_distance_new += addition
                    distance_new = np.linalg.norm(
                        np.dot(cell.T, scaled_distance_new))
                    if distance > distance_new:
                        distance = distance_new
                        scaled_distance = scaled_distance_new
                distance_matrix[i1, i2] = distance
                scaled_distances[i1, i2] = scaled_distance
        self._distance_matrix = distance_matrix
        self._scaled_distances = scaled_distances
        return self

    def write_properties(self, precision=16):
        width = precision + 6
        width_int = 5

        key_order = [
            "filename",
            "number_of_atoms",
            "volume",
            "volume_per_atom",
            "a",
            "b",
            "c",
            "b/a",
            "c/a",
            "a/b",
            "c/b",
            "a/c",
            "b/c",
            "alpha",
            "beta",
            "gamma",
            "b_x_c",
            "c_x_a",
            "a_x_b",
            "spg_number",
            "spg_international",
        ]

        print("-" * 80)
        print(self._filename)
        print("-" * 80)
        for k in key_order:
            if k not in self._dictionary:
                continue
            value = self._dictionary[k]
            sys.stdout.write("{:s}".format(k))
            sys.stdout.write(": ")
            if isinstance(value, float):
                sys.stdout.write(
                    "{:{width}.{precision}f}".format(
                        value,
                        width=width,
                        precision=precision,))
            elif isinstance(value, (int, long)):
                sys.stdout.write(
                    "{:{width}d}".format(
                        value,
                        width=width_int,))
            else:
                sys.stdout.write("{:s}".format(value))
            sys.stdout.write("\n")

    def write_specified_properties(self, keys, precision=16):
        width = precision + 6
        width_int = 5
        for k in keys:
            value = self._dictionary[k]
            sys.stdout.write(" ")
            sys.stdout.write("{:s}".format(k))
            sys.stdout.write(" ")
            if isinstance(value, float):
                sys.stdout.write(
                    "{:{width}.{precision}f}".format(
                        value,
                        width=width,
                        precision=precision,))
            elif isinstance(value, (int, long)):
                sys.stdout.write(
                    "{:{width}d}".format(
                        value,
                        width=width_int,))
                sys.stdout.write(" " * (precision + 1))
            else:
                sys.stdout.write("{:s}".format(value))
        sys.stdout.write("\n")

    def get_index_from_position(self, position, symprec=1e-6):

        for i, p in enumerate(self._atoms.get_scaled_positions()):
            diff = position - p
            diff -= np.rint(diff)
            if all([abs(x) < symprec for x in diff]):
                return i
        print("WARNING: {}".format(__name__))
        print("Index for the specified position cannot be found.")
        return None

    def write_distance_matrix(self):
        number_of_atoms = self._atoms.get_number_of_atoms()
        for i1 in range(number_of_atoms):
            for i2 in range(number_of_atoms):
                distance = self._distance_matrix[i1, i2]
                sys.stdout.write("{:22.16f}".format(distance))
            sys.stdout.write("\n")

    def write_sorted_distance_matrix(self):
        """Write distances between an atom and another one.
        """
        number_of_atoms = self._atoms.get_number_of_atoms()
        chemical_symbols = self._atoms.get_chemical_symbols()
        positions = self._atoms.get_scaled_positions()
#        sys.stdout.write("# {:4d}\n".format(number_of_atoms))
        for i1, c1 in enumerate(chemical_symbols):
            distances_index = np.argsort(self._distance_matrix[i1])
            for i2 in distances_index:
                c2 = chemical_symbols[i2]
                d = self._distance_matrix[i1, i2]
                # dp = positions[i1] - positions[i2]
                dp = self._scaled_distances[i1, i2]
                sys.stdout.write("{:6d}".format(i1))
                sys.stdout.write("{:>6s}".format(c1))
                sys.stdout.write("{:6d}".format(i2))
                sys.stdout.write("{:>6s}".format(c2))
                sys.stdout.write("{:12.6f}".format(d))
                sys.stdout.write(" ")
                sys.stdout.write(("{:12.6f}" * 3).format(*dp))
                sys.stdout.write("\n")
            sys.stdout.write("\n")

    def set_atoms(self, atoms):
        self._atoms = atoms
        return self

    def set_scaled_positions(self, scaled_positions):
        self._atoms.set_scaled_positions(scaled_positions)
        return self

    def set_positions(self, positions):
        cell = self._atoms.get_cell()
        scaled_positions = np.dot(positions, np.linalg.inv(cell))
        return self.set_scaled_positions(scaled_positions)

    def displace_scaled_positions(self, scaled_displacements):
        scaled_positions = self._atoms.get_scaled_positions()
        scaled_positions += scaled_displacements
        self._atoms.set_scaled_positions(scaled_positions)
        return self

    def displace_positions(self, displacements):
        cell = self._atoms.get_cell()
        scaled_displacements = np.dot(displacements, np.linalg.inv(cell))
        return self.displace_scaled_positions(scaled_displacements)

    def shift_to_origin(self, index):
        scaled_displacements = -self._atoms.get_scaled_positions()[index]
        return self.displace_scaled_positions(scaled_displacements)

    def remove_atoms_indices(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        indices = sorted(set(indices), reverse=True)
        scaled_positions = self._atoms.get_scaled_positions()
        chemical_symbols = self._atoms.get_chemical_symbols()
        for i in indices:
            scaled_positions = np.delete(scaled_positions, i, 0)
            del chemical_symbols[i]
        self.set_scaled_positions(scaled_positions)
        self.set_chemical_symbols(chemical_symbols)
        self._atoms._symbols_to_numbers()
        self._atoms._symbols_to_masses()
        return self

    def remove_atoms_outside(self, region):
        """

        region: 3 x 2 arrays given by direct coordinates.
            [[a-, a+],
             [b-, b+],
             [c-, c+]]
        """
        self.wrap_into_cell()
        region = np.array(region)
        scaled_positions = self._atoms.get_scaled_positions()
        indices_removed = []
        for ix in range(3):
            for i, sp in enumerate(scaled_positions):
                if (sp[ix] < region[ix, 0] or region[ix, 1] < sp[ix]):
                    indices_removed.append(i)
        return self.remove_atoms_indices(indices_removed)

    def add_vacuum_layer(self, vacuum_layer):
        """

        vacuum_layer: 3 x 2 arrays given by direct coordinates.
            [[a-, a+],
             [b-, b+],
             [c-, c+]]
        """
        self.wrap_into_cell()
        vacuum_layer = np.array(vacuum_layer)
        cell = self._atoms.get_cell()
        scaled_positions = self._atoms.get_scaled_positions()
        natoms = self._atoms.get_number_of_atoms()
        for ix in range(3):
            cell[ix] *= (1.0 + sum(vacuum_layer[ix, :]))
            for i in range(natoms):
                scaled_positions[i, ix] += vacuum_layer[ix, 0]
                scaled_positions[i, ix] /= (1.0 + sum(vacuum_layer[ix, :]))
        self.set_cell(cell)
        self.set_scaled_positions(scaled_positions)
        return self

    def wrap_into_cell(self):
        scaled_positions = self.get_atoms().get_scaled_positions()
        scaled_positions -= np.floor(scaled_positions)
        self.set_scaled_positions(scaled_positions)
        return self

    def set_cell(self, cell):
        self._atoms.set_cell(cell)
        return self

    def set_chemical_symbols(self, symbols):
        self._atoms.set_chemical_symbols(symbols)
        return self

    def deform_cell_left(self, matrix):
        """Deform cell as (a, b, c) = M * (a, b, c)
        """
        matrix = _get_matrix(matrix)
        # Generate lattice vectors for the deformed cell.
        cell = self._atoms.get_cell()
        cell = np.dot(matrix, cell.T).T
        self._atoms.set_cell(cell)

        self.update_attributes()

        return self

    def deform_cell_right(self, matrix):
        """Deform cell as (a, b, c) = (a, b, c) * M
        """
        matrix = _get_matrix(matrix)
        # Generate lattice vectors for the deformed cell.
        cell = self._atoms.get_cell()
        cell = np.dot(cell.T, matrix).T
        self._atoms.set_cell(cell)

        self.update_attributes()

        return self

    def generate_supercell(self, dim, prec=1e-9):
        """Generate supercell according to "dim".

        (a_s, b_s, c_s) = (a_u, b_u, c_u) * dim
        """
        dim = _get_matrix(dim)

        # Generate lattice vectors for the suprecell.
        cell = self._atoms.get_cell()
        cell = np.dot(cell.T, dim).T
        self._atoms.set_cell(cell)

        self._generate_supercell_positions(dim, prec)

        nexpansion = np.rint(np.abs(np.linalg.det(dim)))
        chemical_symbols_new = []
        for chemical_symbol in self._atoms.get_chemical_symbols():
            chemical_symbols_new += [chemical_symbol] * nexpansion
        self._atoms.set_chemical_symbols(chemical_symbols_new)
        self._atoms._symbols_to_numbers()
        self._atoms._symbols_to_masses()

        self.update_attributes()

        return self

    def _generate_supercell_positions(self, dim, prec=1e-9):
        """Generate scaled positions in the supercell."""
        translation_vectors = find_lattice_vectors(dim, prec=prec)

        positions = self._atoms.get_scaled_positions()

        # Convert positions to into the fractional coordinates for SC.
        positions = np.dot(np.linalg.inv(dim), positions.T).T

        supercell_positions = (positions[:, None] +
                               translation_vectors[None, :])
        supercell_positions = supercell_positions.reshape(-1, 3)
        self.set_scaled_positions(supercell_positions)

    def sort_by_coordinates(self, index, sorted_by_symbols=False):
        """

        index:
            0: a, 1: b, 2: c
        """
        symbols = self._atoms.get_chemical_symbols()
        positions = self._atoms.get_scaled_positions()
        order = list(symbols)
        data = zip(symbols, positions)
        data = sorted(data, key=lambda x: x[1][index])
        self.set_chemical_symbols(zip(*data)[0])
        self.set_scaled_positions(zip(*data)[1])
        if sorted_by_symbols:
            self = self.sort_by_symbols(order=order)
        self._atoms._symbols_to_numbers()
        self._atoms._symbols_to_masses()
        return self

    def sort_by_symbols(self, order=None):
        """Combine the same chemical symbols.

        Positions are sorted by the combined chemical symbols.
        """
        symbols = self._atoms.get_chemical_symbols()
        positions = self._atoms.get_scaled_positions()
        if order is None:
            order = list(symbols)
        data = zip(symbols, positions)
        data = sorted(data, key=lambda x: order.index(x[0]))
        self.set_chemical_symbols(zip(*data)[0])
        self.set_scaled_positions(zip(*data)[1])
        self._atoms._symbols_to_numbers()
        self._atoms._symbols_to_masses()
        return self

    def get_dictionary(self):
        return self._dictionary

    def get_atoms(self):
        return self._atoms

    def get_cell(self):
        return self._atoms.get_cell()

    def get_scaled_distances(self):
        return self._scaled_distances.copy()

    def get_distance_matrix(self):
        return self._distance_matrix.copy()

    def change_volume(self, volume):
        cell_current = self._atoms.get_cell()
        volume_current = np.linalg.det(cell_current)
        scale = (volume / volume_current) ** (1.0 / 3.0)
        self._atoms.set_cell(cell_current * scale)
        return self

    def change_volume_per_atom(self, volume_per_atom):
        volume = volume_per_atom * self._atoms.get_number_of_atoms()
        return self.change_volume(volume)

    def get_symmetry_dataset(self):
        return Symmetry(self._atoms).get_dataset()

    def get_mappings_for_symops(self, prec=1e-6):
        """Get mappings for symmetry operations."""
        natoms = self._atoms.get_number_of_atoms()

        dataset = self.get_symmetry_dataset()
        rotations = dataset["rotations"]
        translations = dataset["translations"]
        nopr = len(rotations)
        mappings = -1 * np.ones((nopr, natoms), dtype=int)
        for iopr, (r, t) in enumerate(zip(rotations, translations)):
            mappings[iopr] = self.extract_mapping_for_symopr(r, t, prec)

        if -1 in mappings:
            print("ERROR: {}".format(__name__))
            print("Some atoms are not mapped by some symmetry operations.")
            raise ValueError
            sys.exit(1)

        return mappings

    def extract_transformed_scaled_positions(self, rotation, translation):
        """Extract transformed scaled positions.

        Args:
            rotation (3x3 array): Rotation matrix.
            translation (3 array): Translation vector.

        Returns:
            Transformed scaled positions by the rotation and translation.
            Note that if the rotation and the translation is not a symmetry
            operations, the returned values could be strange.
        """
        scaled_positions = self._atoms.get_scaled_positions()
        transformed_scaled_positions = transform_scaled_positions(
            scaled_positions, rotation, translation)
        return transformed_scaled_positions

    def extract_mapping_for_symopr(self, rotation, translation, prec=1e-6):
        """Extract a mapping for a pair of a symmetry operation.

        Args:
            rotation (3x3 array): Rotation matrix.
            translation (3 array): Translation vector.

        Returns:
            mapping (n integral array):
                Indices are for new numbers and contents are for old ones.
        """
        chemical_symbols = self._atoms.get_chemical_symbols()
        transformed_scaled_positions = (
            self.extract_transformed_scaled_positions(rotation, translation))
        mapping = self.extract_mapping_for_atoms(
            chemical_symbols, transformed_scaled_positions, prec)

        return mapping

    def extract_mapping_for_atoms(self, symbols_new, positions_new, prec=1e-6):
        """
        Args:
            symbols_new: Chemical symbols for the transformed structures.
            positions_new: Fractional positions for the transformed structures.

        Return:
            mapping (n integral array):
                Indices are for new numbers and contents are for old ones.
        """
        natoms = self._atoms.get_number_of_atoms()
        symbols_old = np.array(self._atoms.get_chemical_symbols())
        positions_old = self._atoms.get_scaled_positions()

        diff = positions_new[:, None, :] - positions_old[None, :, :]
        wrapped_dpos = diff - np.rint(diff)
        tmp, mapping = np.where(np.all(np.abs(wrapped_dpos) < prec, axis=2))

        # Guarantee one-to-one correspondence
        if not np.array_equal(tmp, np.arange(natoms, dtype=int)):
            raise ValueError('Mapping is failed.')

        if not np.array_equal(symbols_new, symbols_old[mapping]):
            raise ValueError('Symbols do not correspond.')

        return mapping


def _get_matrix(matrix):
    matrix = np.array(matrix)
    if matrix.size == 1 or matrix.size == 3:
        matrix = matrix * np.eye(3)
    elif matrix.size == 9:
        matrix = matrix.reshape((3, 3))
    else:
        print("ERROR {}:".format(__name__))
        print("Size of matrix must be 1, 3 or 9.")
        print("The current size is {}.".format(matrix.size))
        raise ValueError
    return matrix


def convert_cell_to_lc(cell):
    """Convert lattice vectors to lattice constants.

    Args:
        cell: [[a_x, a_y, a_z], [b_x, b_y, b_z], [c_x, c_y, c_z]]

    Returns:
        lc: a dictionary like
            {"a": ..., "b": ..., "c": ...,
            "alpha": ..., "beta": ..., "gamma": ...}
    """
    a = np.linalg.norm(cell[0])
    b = np.linalg.norm(cell[1])
    c = np.linalg.norm(cell[2])
    alpha = np.dot(cell[1], cell[2]) / (b * c)
    beta = np.dot(cell[2], cell[0]) / (c * a)
    gamma = np.dot(cell[0], cell[1]) / (a * b)
    alpha = np.arccos(alpha) * 180.0 / np.pi
    beta = np.arccos(beta) * 180.0 / np.pi
    gamma = np.arccos(gamma) * 180.0 / np.pi
    b_x_c = np.cross(cell[1], cell[2])
    c_x_a = np.cross(cell[2], cell[0])
    a_x_b = np.cross(cell[0], cell[1])
    volume = np.linalg.det(cell)
    lc = {
        "volume": volume,
        "a": a,
        "b": b,
        "c": c,
        "b/a": b / a,
        "c/a": c / a,
        "a/b": a / b,
        "c/b": c / b,
        "a/c": a / c,
        "b/c": b / c,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "b_x_c": np.linalg.norm(b_x_c),
        "c_x_a": np.linalg.norm(c_x_a),
        "a_x_b": np.linalg.norm(a_x_b),
    }

    return lc


def convert_lc_to_cell(lc, prec=1e-12):
    """Convert lattice constants to lattice vectors.

    Args:
        lc: a dictionary like
            {"a": ..., "b": ..., "c": ...,
            "alpha": ..., "beta": ..., "gamma": ...}

    Returns:
        cell: [[a_x, a_y, a_z], [b_x, b_y, b_z], [c_x, c_y, c_z]]
    """
    a = lc["a"]
    b = lc["b"]
    c = lc["c"]
    alpha = np.radians(lc["alpha"])
    beta = np.radians(lc["beta"])
    gamma = np.radians(lc["gamma"])

    cell = np.zeros((3, 3))
    cell[0, 0] = a
    cell[1, 0] = b * np.cos(gamma)
    cell[1, 1] = b * np.sin(gamma)
    cell[2, 0] = c * np.cos(beta)
    tmp = np.cos(alpha) - np.cos(gamma) * np.cos(beta)
    tmp /= np.sin(gamma)
    cell[2, 1] = c * tmp
    cell[2, 2] = c * np.sqrt(np.sin(beta) ** 2 - tmp ** 2)

    cell[abs(cell) < prec] = 0.0  # Small values are replaced by zero.

    return cell


def find_lattice_vectors(supercell_matrix, prec=1e-9):
    """Find the set of latice vectors inside the supercell.

    Args:
        supercell_matrix (3x3 array):
            (a_s, b_s, c_s) = (a_u, b_u, c_u) * supercell_matrix

    Returns:
        the set of translation vectors (fractional coordinates for SC).
        The 1st index moves the fastest.
    """
    nexpansion = np.rint(np.abs(np.linalg.det(supercell_matrix)))

    def generate_lv_range(i):
        low = 0
        high = 0
        for j in supercell_matrix[i]:
            if j > 0:
                high += j
            else:
                low += j
        return np.arange(low, high + 1)
    range_a = generate_lv_range(0)[:, None] * np.array([1, 0, 0])[None, :]
    range_b = generate_lv_range(1)[:, None] * np.array([0, 1, 0])[None, :]
    range_c = generate_lv_range(2)[:, None] * np.array([0, 0, 1])[None, :]

    # translation vectors in unit cell units
    translation_vectors = (range_c[:, None, None] +
                           range_b[None, :, None] +
                           range_a[None, None, :])

    translation_vectors = translation_vectors.reshape((-1, 3))
    translation_vectors = np.dot(
        np.linalg.inv(supercell_matrix), translation_vectors.T).T

    translation_vectors = translation_vectors[
        np.where(np.all(translation_vectors < 1 - prec, axis=1) &
                 np.all(translation_vectors >= -prec, axis=1))
    ]

    if len(translation_vectors) != nexpansion:
        print("ERROR: {}".format(__name__))
        print("len(translation_vectors) != abs(det(supercell_matrix))")
        print(len(translation_vectors), nexpansion)
        raise ValueError

    return translation_vectors


def transform_scaled_positions(scaled_positions, rotation, translation):
    """

    Args:
        scaled_positions (nx3 array): Scaled positions.
        rotation (3x3 array): Rotation matrix.
        translation (3 array): Translation vector.

    Returns:
        transformed_scaled_positions.
    """
    transformed_scaled_positions = np.dot(rotation, scaled_positions.T).T
    transformed_scaled_positions += translation
    return transformed_scaled_positions
