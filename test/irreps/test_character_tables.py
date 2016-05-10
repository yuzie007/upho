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
            '1'   , '-1'   , '2'   , 'm'    , '2/m'  ,
            '222' , 'mm2'  , 'mmm' , '4'    , '-4'   ,
            '4/m' , '422'  , '4mm' , '-42m' , '4/mmm',
            '3'   , '-3'   , '32'  , '3m'   , '-3m'  ,
            '6'   , '-6'   , '6/m' , '622'  , '6mm'  ,
            '-6m2', '6/mmm', '23'  , 'm-3'  , '432'  ,
            '-43m', 'm-3m' ,
        )
        print()
        for pg in pointgroup_symbols:
            print("{:6s}:".format(pg), end=" ")
            current_max = 0
            if pg in character_tables:
                ir_labels = character_tables[pg]["ir_labels"]
                current_max = max(max(len(s) for s in ir_labels), current_max)
                print(ir_labels, current_max)
            else:
                print("Not implemeneted yet.")
        self.assertEqual(current_max, 3)


if __name__ == "__main__":
    unittest.main()
