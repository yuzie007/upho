#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import numpy as np
try:
    from group.mathtools import lcm
except:
    from mathtools import lcm


class Group(object):
    def __init__(self, Cayley_table):
        """

        : 2-dim array-like
        """
        self.set_Cayley_table(Cayley_table)
        self._find_order_of_group()
        self._find_identity()

        self._create_conjugacy_classes()
        self._create_orders_of_elements()
        self._create_exponent()  # After creating orders_or_elements
        self.create_center()
        self._create_commutator_subgroup()
        self._create_crst()
        self._create_character_table()

    def set_Cayley_table(self, Cayley_table):
        self._Cayley_table = np.array(Cayley_table)

    def _find_order_of_group(self):
        self._order = self._Cayley_table.shape[0]

    def _find_identity(self):
        # TODO(ikeda): How about arbitrary symbols for the Cayley table?
        order = self._order
        Cayley_table = self._Cayley_table
        for i in range(order):
            is_identity = True
            for j in range(order):
                if Cayley_table[i, j] != j:
                    is_identity = False
                    break
            if is_identity:
                identity = i
                break
        self._identity = identity

    def _create_conjugacy_classes(self):
        Cayley_table = self._Cayley_table
        order = self._order

        conjugacy_classes = np.ones(order, dtype=int) * -1  # Initialized by -1
        count = 0
        for i in range(order):
            if conjugacy_classes[i] == -1:
                # "i" belongs to a new conjygacy class.
                conjugacy_classes[i] = count
                count += 1
            else:
                # The conjugacy class including "i" is already obtained.
                continue
            current_conjugacy_class = conjugacy_classes[i]
            for j in range(order):
                jinv = self.create_inverse_element(j)
                k = Cayley_table[Cayley_table[j, i], jinv]
                if conjugacy_classes[k] == -1:
                    conjugacy_classes[k] = current_conjugacy_class
                elif conjugacy_classes[k] != current_conjugacy_class:
                    raise ValueError(
                        "The conjugacy class of k is not equal to that of i.")
        self._conjugacy_classes = conjugacy_classes
        self._create_orders_of_conjugacy_classes()

    def _create_orders_of_conjugacy_classes(self):
        conjugacy_classes = self._conjugacy_classes
        self._orders_of_conjugacy_classes = np.bincount(conjugacy_classes)

    def _create_character_table(self):
        pass
        # crst = self._crst
        # for tmp_matrix in crst:
        #     # Get left eigenvectors
        #     eigvals, eigvecs = np.linalg.eig(tmp_matrix.T)
        #     eigvecs = eigvecs.T
        #     print("eigvals:")
        #     print(eigvals)
        #     print("eigvecs:")
        #     print(eigvecs)

    def create_center(self):
        """
        Create the center of the group G.

        The center is a normal abelian subgroup of G.
        """
        order = self._order
        Cayley_table = self._Cayley_table

        center = []
        for i in range(order):
            is_central = True
            for j in range(order):
                jinv = self.create_inverse_element(j)
                iconj = Cayley_table[Cayley_table[j, i], jinv]
                if i != iconj:
                    is_central = False
                    break
            if is_central:
                center.append(i)  # "i" is conjugate to all the group elements.
        self._center = np.array(center)

    def _create_crst(self):
        order = self._order
        Cayley_table = self._Cayley_table
        conjugacy_classes = self._conjugacy_classes

        num_classes = np.unique(conjugacy_classes).size
        crst = np.zeros((num_classes, num_classes, num_classes), dtype=int)
        for iclass in range(num_classes):
            # the representative
            i = np.where(conjugacy_classes == iclass)[0][0]
            for j in range(order):
                for k in range(order):
                    if Cayley_table[j][k] == i:
                        jclass = conjugacy_classes[j]
                        kclass = conjugacy_classes[k]
                        crst[jclass, kclass, iclass] += 1
        self._crst = crst

    def create_centralizer(self, subset):
        order = self._order
        Cayley_table = self._Cayley_table

        centralizer = []
        for i in range(order):
            is_in_centralizer = True
            iinv = self.create_inverse_element(i)
            for s in subset:
                sconj = Cayley_table[Cayley_table[i, s], iinv]
                if s != sconj:
                    is_in_centralizer = False
                    break
            if is_in_centralizer:
                centralizer.append(i)
        self._centralizer = np.array(centralizer)

    def create_normalizer(self, subset):
        order = self._order
        Cayley_table = self._Cayley_table

        normalizer = []
        for i in range(order):
            is_in_normalizer = True
            iinv = self.create_inverse_element(i)
            for s in subset:
                sconj = Cayley_table[Cayley_table[i, s], iinv]
                if sconj not in subset:
                    is_in_normalizer = False
                    break
            if is_in_normalizer:
                normalizer.append(i)
        self._normalizer = np.array(normalizer)

    def _create_commutator_subgroup(self):
        """

        https://en.wikipedia.org/wiki/Commutator_subgroup
        """
        order = self._order
        commutator_subgroup = []
        for i in range(order):
            for j in range(order):
                k = self.create_commutator(i, j)
                commutator_subgroup.append(k)
        self._commutator_subgroup = np.unique(commutator_subgroup)

    def create_commutator(self, i, j):
        Cayley_table = self._Cayley_table
        i_inv = self.create_inverse_element(i)
        j_inv = self.create_inverse_element(j)
        k = Cayley_table[i_inv, Cayley_table[j_inv, Cayley_table[i, j]]]
        return k

    def create_inverse_element(self, element):
        Cayley_table = self._Cayley_table
        identity = self._identity
        inverse = np.where(Cayley_table[element] == identity)[0][0]
        return inverse

    def _create_orders_of_elements(self):
        order = self._order
        orders_of_elements = np.zeros(order, dtype=int)
        for i in range(order):
            orders_of_elements[i] = self.create_order_of_element(i)
        self._orders_of_elements = orders_of_elements

    def create_order_of_element(self, i):
        Cayley_table = self._Cayley_table
        identity = self._identity
        j = i
        count = 0
        while True:
            count += 1
            if j == identity:
                order_of_element = count
                return order_of_element
            else:
                j = Cayley_table[j, i]

    def _create_exponent(self):
        orders_of_elements = self._orders_of_elements
        exponent = lcm(*orders_of_elements)
        self._exponent = exponent

    def get_identity(self):
        return self._identity

    def get_order_of_group(self):
        return self._order

    def get_orders_of_elements(self):
        return self._orders_of_elements

    def get_exponent(self):
        return self._exponent

    def get_conjugacy_classes(self):
        return self._conjugacy_classes

    def get_orders_of_conjugacy_classes(self):
        return self._orders_of_conjugacy_classes

    def get_center(self):
        return self._center

    def get_centralizer(self):
        return self._centralizer

    def get_normalizer(self):
        return self._normalizer

    def get_commutator_subgroup(self):
        return self._commutator_subgroup

    def get_crst(self):
        return self._crst
