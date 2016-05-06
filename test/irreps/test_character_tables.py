#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Yuji Ikeda"

import unittest
from ph_unfolder.irreps.character_tables import character_tables

class TestCharacterTables(unittest.TestCase):
    def test_ir_labels_length(self):
        pointgroup_symbols = (
            '1',
            '-1',
            '2',
            'm',
            '2/m',
            '222',
            'mm2',
            'mmm',
            '4',
            '4mm',
            '4/mmm',
            '3m',
            '-3m',
            '6/mmm',
            'm-3m',
        )
        print()
        for pg in pointgroup_symbols:
            current_max = 0
            ir_labels = character_tables[pg]["ir_labels"]
            current_max = max(max(len(s) for s in ir_labels), current_max)
            print(pg, ir_labels, current_max)
        self.assertEqual(current_max, 3)


if __name__ == "__main__":
    unittest.main()
