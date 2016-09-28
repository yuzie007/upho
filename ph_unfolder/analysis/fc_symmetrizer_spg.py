#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TODO(ikeda): The structure of the variable "force_constants_pair" should be
#     modified. We want to use numpy functions.
from __future__ import absolute_import, division, print_function

import itertools

import numpy as np
from phonopy.file_IO import write_FORCE_CONSTANTS
from phonopy.structure.symmetry import Symmetry
from ..analysis.mappings_modifier import MappingsModifier
from ..structure.symtools import get_rotations_cart
from ..analysis.fc_analyzer_base import FCAnalyzerBase
from ..structure.structure_analyzer import StructureAnalyzer


class FCSymmetrizerSPG(FCAnalyzerBase):
    def average_force_constants_spg(self, symprec=1e-5):
        atoms = self._atoms
        fc_orig = self._force_constants

        atoms_symmetry = self._atoms_ideal

        symmetry = Symmetry(atoms_symmetry)

        symbols = atoms.get_chemical_symbols()
        symboltypes = sorted(set(symbols), key=symbols.index)

        rotations_cart = get_rotations_cart(atoms_symmetry)
        mappings = StructureAnalyzer(
            atoms_symmetry).get_mappings_for_symops(prec=symprec)
        mappings_inv = MappingsModifier(mappings).invert_mappings()

        print("mappings: Finished.")
        (nsym, natoms) = mappings.shape
        print("nsym: {}".format(nsym))
        print("natoms: {}".format(natoms))

        fc_mean = np.zeros_like(fc_orig)
        fc_mean_square = np.zeros_like(fc_orig)

        fc_mean_symbols, fc_mean_square_symbols, fc_std_symbols, counters = (
            self.initialize_fc_symbols(fc_orig, symboltypes)
        )

        for (minv, r) in zip(mappings_inv, rotations_cart):
            for i1 in symmetry.get_independent_atoms():
                for i2 in range(natoms):
                    j1 = minv[i1]
                    j2 = minv[i2]
                    s1 = symbols[j1]
                    s2 = symbols[j2]

                    tmp = np.dot(np.dot(r, fc_orig[j1, j2]), r.T)
                    tmp2 = tmp ** 2
                    fc_mean[i1, i2] += tmp
                    fc_mean_square[i1, i2] += tmp2

                    fc_mean_symbols[(s1, s2)][i1, i2] += tmp
                    fc_mean_square_symbols[(s1, s2)][i1, i2] += tmp2

                    counters[(s1, s2)][i1, i2] += 1

        fc_mean        /= float(len(rotations_cart))
        fc_mean_square /= float(len(rotations_cart))

        for i1 in symmetry.get_independent_atoms():
            for i2 in range(natoms):
                for (key, c) in counters.items():
                    if c[i1, i2] != 0:
                        fc_mean_symbols       [key][i1, i2] /= c[i1, i2]
                        fc_mean_square_symbols[key][i1, i2] /= c[i1, i2]
                    else:
                        fc_mean_symbols       [key][i1, i2] = np.nan
                        fc_mean_square_symbols[key][i1, i2] = np.nan

        ########################################
        # STD
        ########################################
        fc_std = get_matrix_std(fc_mean, fc_mean_square)

        for key in counters.keys():
            fc_std_symbols[key] = get_matrix_std(
                fc_mean_symbols[key], fc_mean_square_symbols[key])

        ########################################
        # Distribution
        ########################################
        fc_mean = self.distribute_force_constants_spg(
            fc_mean, symmetry, rotations_cart, mappings)
        fc_std = self.distribute_force_constants_spg(
            fc_std, symmetry, rotations_cart, mappings)

        for key in counters.keys():
            fc_mean_symbols[key] = self.distribute_force_constants_spg(
                fc_mean_symbols[key], symmetry, rotations_cart, mappings)
            fc_std_symbols[key] = self.distribute_force_constants_spg(
                fc_std_symbols[key], symmetry, rotations_cart, mappings)

        # After the distributions, the signs of SDs can be changed.
        # However, the signs of SDs have no meaning.
        # To suppress the meaningless signs, we take the absolute values here.
        # TODO(ikeda): Consider the meaning of SDs for vectors or tensors.
        fc_std = np.abs(fc_std)
        for key in counters.keys():
            fc_std_symbols[key] = np.abs(fc_std_symbols[key])

        self._force_constants_symmetrized = fc_mean
        self._force_constants_sd = fc_std
        self._force_constants_pair = fc_mean_symbols
        self._force_constants_pair_sd = fc_std_symbols

    @staticmethod
    def initialize_fc_symbols(fc_orig, symboltypes):
        natoms = fc_orig.shape[0]

        fc_mean_symbols        = {}
        fc_mean_square_symbols = {}
        fc_std_symbols         = {}
        counters               = {}
        for s1 in symboltypes:
            for s2 in symboltypes:
                fc_mean_symbols       [(s1, s2)] = np.zeros_like(fc_orig)
                fc_mean_square_symbols[(s1, s2)] = np.zeros_like(fc_orig)
                fc_std_symbols        [(s1, s2)] = np.zeros_like(fc_orig)
                counters              [(s1, s2)] = np.zeros((natoms, natoms), dtype=int)

        return fc_mean_symbols, fc_mean_square_symbols, fc_std_symbols, counters

    def distribute_force_constants_spg(self, fc, symmetry, rotations_cart, mappings):
        fc_distributed = np.zeros_like(fc)
        natoms = fc_distributed.shape[0]
        map_atoms = symmetry.get_map_atoms()
        map_operations = symmetry.get_map_operations()
        for i in range(natoms):
            i_equiv = map_atoms[i]
            iop = map_operations[i]
            r = rotations_cart[iop]
            for j in range(natoms):
                j_equiv = mappings[iop, j]
                fc_distributed[i, j] = np.dot(np.dot(r.T, fc[i_equiv, j_equiv]), r)

        return fc_distributed

    def average_force_constants_spg_full(self, symprec=1e-5):
        """Generate symmetrized force constants.

        If the structure for extracting symmetry operations are different from
        the structure for extracting chemical symbols, we must specify symbols
        explicitly.
        """

        atoms = self._atoms
        symbols = atoms.get_chemical_symbols()
        symboltypes = sorted(set(symbols), key=symbols.index)
        nsymbols = len(symboltypes)

        atoms_symmetry = self._atoms_ideal

        # mappings: each index is for the "after" symmetry operations, and
        #     each element is for the "original" positions. 
        #     mappings[k][i] = j means the atom j moves to the positions of
        #     the atom i for the k-th symmetry operations.
        rotations_cart = get_rotations_cart(atoms_symmetry)
        mappings = StructureAnalyzer(
            atoms_symmetry).get_mappings_for_symops(prec=symprec)

        print("mappings: Finished.")
        (nsym, natoms) = mappings.shape
        print("nsym: {}".format(nsym))
        print("natoms: {}".format(natoms))

        shape = self._force_constants.shape

        force_constants_symmetrized = np.zeros(shape)
        force_constants_sd = np.zeros(shape)

        force_constants_pair = {}
        force_constants_pair_sd = {}
        pair_counters = {}
        for s1 in symboltypes:
            for s2 in symboltypes:
                force_constants_pair[(s1, s2)] = np.zeros(shape)
                force_constants_pair_sd[(s1, s2)] = np.zeros(shape)
                pair_counters[(s1, s2)] = np.zeros((natoms, natoms), dtype=int)

        for (m, r) in zip(mappings, rotations_cart):
            # i1, i2: indices after symmetry operations
            # j1, j2: indices before symmetry operations
            for i1 in range(natoms):
                for i2 in range(natoms):
                    j1 = m[i1]
                    j2 = m[i2]
                    s_i1 = symbols[i1]
                    s_i2 = symbols[i2]
                    s_j1 = symbols[j1]
                    s_j2 = symbols[j2]

                    tmp = np.dot(np.dot(r, self._force_constants[i1, i2]), r.T)
                    tmp2 = tmp ** 2
                    force_constants_symmetrized[j1, j2] += tmp
                    force_constants_sd[j1, j2] += tmp2

                    force_constants_pair[(s_i1, s_i2)][j1, j2] += tmp
                    force_constants_pair_sd[(s_i1, s_i2)][j1, j2] += tmp2
                    pair_counters[(s_i1, s_i2)][j1, j2] += 1

        self._pair_counters = pair_counters
        counter_check = np.zeros((natoms, natoms), dtype=int)
        for (key, c) in pair_counters.items():
            counter_check += c
        self._counter_check = counter_check

        force_constants_symmetrized /= float(nsym)
        force_constants_sd /= float(nsym)
        force_constants_sd = get_matrix_std(
            force_constants_symmetrized,
            force_constants_sd)

        for (s_i1, s_i2) in itertools.product(symboltypes, repeat=2):
            for (i1, i2) in itertools.product(range(natoms), repeat=2):
                cval = pair_counters[(s_i1, s_i2)][i1, i2]
                if cval != 0:
                    force_constants_pair[(s_i1, s_i2)][i1, i2] /= cval
                    force_constants_pair_sd[(s_i1, s_i2)][i1, i2] /= cval
                else:
                    force_constants_pair[(s_i1, s_i2)][i1, i2] = np.nan
                    force_constants_pair_sd[(s_i1, s_i2)][i1, i2] = np.nan
            force_constants_pair_sd[(s_i1, s_i2)] = get_matrix_std(
                force_constants_pair[(s_i1, s_i2)],
                force_constants_pair_sd[(s_i1, s_i2)])

        self._force_constants_symmetrized = force_constants_symmetrized
        self._force_constants_sd = force_constants_sd
        self._force_constants_pair = force_constants_pair
        self._force_constants_pair_sd = force_constants_pair_sd

    def get_force_constants_symmetrized(self):
        return self._force_constants_symmetrized

    def get_force_constants_pair(self):
        return self._force_constants_pair

    def get_force_constants_sd(self):
        return self._force_constants_sd

    def get_force_constants_pair_sd(self):
        return self._force_constants_pair_sd

    def get_pair_counters(self):
        return self._pair_counters

    def write_force_constants_symmetrized(
            self,
            filename_write="FORCE_CONSTANTS_SPG"):

        fc = self.get_force_constants_symmetrized()
        write_FORCE_CONSTANTS(fc, filename_write)

    def write_force_constants_sd(
            self,
            filename_write="FORCE_CONSTANTS_SD"):

        fc = self.get_force_constants_sd()
        write_FORCE_CONSTANTS(fc, filename_write)

    def write_force_constants_pair(
            self,
            filename_write="FORCE_CONSTANTS_PAIR"):

        force_constants_pair = self.get_force_constants_pair()

        for (pairtypes, force_constants_pair) in force_constants_pair.items():
            filename_write_pair = "{}_{}_{}".format(filename_write, *pairtypes)
            write_FORCE_CONSTANTS(force_constants_pair,
                                  filename_write_pair)

    def write_force_constants_pair_sd(
            self,
            filename_write="FORCE_CONSTANTS_PAIR_SD"):

        force_constants_pair_sd = self.get_force_constants_pair_sd()

        for (pairtypes, force_constants_pair_sd) in force_constants_pair_sd.items():
            filename_write_pair = "{}_{}_{}".format(filename_write, *pairtypes)
            write_FORCE_CONSTANTS(force_constants_pair_sd,
                                  filename_write_pair)

    def write_pair_counters(self, filename_write="PAIR_COUNTER"):

        counters = self.get_pair_counters()

        for (pairtypes, pair_counter) in counters.items():
            natoms = pair_counter.shape[0]
            filename_write_pair = "{}_{}_{}".format(filename_write, *pairtypes)
            with open(filename_write_pair, "w") as f:
                f.write("{:4d}\n".format(natoms))
                for (i1, i2) in itertools.product(range(natoms), repeat=2):
                    c = pair_counter[i1, i2]
                    f.write("{:4d}{:4d}{:8d}\n".format(i1, i2, c))

    def write_counter_check(self, filename_write="COUNTER_CHECK"):

        counters = self.get_pair_counters()
        counter_sum = np.sum(counters.values())

        natoms = counter_sum.shape[0]

        with open(filename_write, "w") as f:
            f.write("{:4d}\n".format(natoms))
            for (i1, i2) in itertools.product(range(natoms), repeat=2):
                c = counter_sum[i1, i2]
                f.write("{:4d}{:4d}{:8d}\n".format(i1, i2, c))


def get_matrix_std(matrix_mean, matrix_mean_square):
    matrix_tmp = matrix_mean_square - matrix_mean ** 2
    matrix_std = np.sqrt(matrix_tmp)
    return matrix_std
