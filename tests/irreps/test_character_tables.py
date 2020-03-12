import pytest
import numpy as np
from upho.irreps.character_tables import character_tables


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
        class_to_rotations_list = character_tables[pg]['class_to_rotations_list']
        lookup = {
            'E'    : (+3, +1),
            'C2'   : (-1, +1),
            'C2x'  : (-1, +1),
            'C2y'  : (-1, +1),
            'C2\'' : (-1, +1),
            'C2\"' : (-1, +1),
            'C4^2' : (-1, +1),
            'C3'   : ( 0, +1),
            'C3^2' : ( 0, +1),
            'C4'   : ( 1, +1),
            'C4^3' : ( 1, +1),
            'C6'   : ( 2, +1),
            'C6^5' : ( 2, +1),
            'i'    : (-3, -1),
            'sgh'  : (+1, -1),
            'sgv'  : (+1, -1),
            'sgv\'': (+1, -1),
            'sgxy' : (+1, -1),
            'sgxz' : (+1, -1),
            'sgyz' : (+1, -1),
            'sgd'  : (+1, -1),
            'S6'   : ( 0, -1),  # -3
            'S6^5' : ( 0, -1),  # -3
            'S4'   : (-1, -1),
            'S4^3' : (-1, -1),
            'S3'   : (-2, -1),  # -6
            'S3^5' : (-2, -1),  # -6
        }
        for class_to_rotations in class_to_rotations_list:
            for rotation_label, rotations in class_to_rotations.items():
                tmp0 = lookup[rotation_label]
                for r in rotations:
                    tmp1 = np.trace(r), int(round(np.linalg.det(r)))
                    assert rotation_label and tmp0 == tmp1
