import pytest
import numpy as np
from upho.irreps.irreps import find_rotation_type
from upho.irreps.character_tables import (
    character_tables, find_rotation_type_from_class_label)


pointgroup_symbols = (
    '1'   , '-1'   , '2'   , 'm'    , '2/m'  ,
    '222' , 'mm2'  , 'mmm' , '4'    , '-4'   ,
    '4/m' , '422'  , '4mm' , '-42m' , '4/mmm',
    '3'   , '-3'   , '32'  , '3m'   , '-3m'  ,
    '6'   , '-6'   , '6/m' , '622'  , '6mm'  ,
    '-6m2', '6/mmm', '23'  , 'm-3'  , '432'  ,
    '-43m', 'm-3m' ,
)


class TestCharacterTables:
    def test_ir_labels_length(self):
        for pg in pointgroup_symbols:
            print("{:6s}:".format(pg), end=" ")
            current_max = 0
            if pg in character_tables:
                ir_labels = character_tables[pg]["ir_labels"]
                current_max = max(max(len(s) for s in ir_labels), current_max)
                print(ir_labels, current_max)

                is_orthogonal = self.is_orthogonal(
                    character_tables[pg]["character_table"])
                assert is_orthogonal
            else:
                print("Not implemented yet.")
        assert current_max == 3

    @staticmethod
    def is_orthogonal(character_table):
        character_table = np.array(character_table)
        for i0, v0 in enumerate(character_table.T):
            for i1, v1 in enumerate(character_table.T):
                if i0 != i1 and not np.isclose(np.dot(v0, np.conj(v1)), 0.0):
                    print(i0, i1, np.dot(v0, v1))
                    return False
        return True

    @pytest.mark.parametrize('pg', pointgroup_symbols)
    def test_class_to_rotations_list(self, pg):
        k = 'class_to_rotations_list'
        if k not in character_tables[pg]:
            return
        class_to_rotations_list = character_tables[pg][k]
        for class_to_rotations in class_to_rotations_list:
            for rotation_label, rotations in class_to_rotations.items():
                tmp0 = find_rotation_type_from_class_label(rotation_label)
                for r in rotations:
                    tmp1 = find_rotation_type(r)
                    assert rotation_label and tmp0 == tmp1
