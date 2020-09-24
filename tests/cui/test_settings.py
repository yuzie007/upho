import io
import numpy as np
from upho.cui.settings import _parse_phonopy_conf_strings


class TestSettings:
    def test_comments(self):
        s0 = ''.join([
            'DIM = 2 2 2\n',
            'PRIMITIVE_AXES =  0 1 1  1 0 1  1 1 0\n',
        ])
        s1 = ''.join([
            'DIM = 2 2 2  # supercell\n',
            'PRIMITIVE_AXES =  0 1 1  1 0 1  1 1 0\n',
        ])
        with io.StringIO(s0) as f0:
            d0 = _parse_phonopy_conf_strings(f0.readlines())
        with io.StringIO(s1) as f1:
            d1 = _parse_phonopy_conf_strings(f1.readlines())
        for k0 in d0.keys():
            assert np.all(d0[k0] == d1[k0])

    def test_continuous_lines(self):
        s0 = ''.join([
            'DIM = 2 2 2\n',
            'PRIMITIVE_AXES =  0 1 1  1 0 1  1 1 0\n',
        ])
        s1 = ''.join([
            'DIM = +++\n',
            '2 2 2\n',
            'PRIMITIVE_AXES =  0 1 1  1 0 1  1 1 0\n',
        ])
        with io.StringIO(s0) as f0:
            d0 = _parse_phonopy_conf_strings(f0.readlines())
        with io.StringIO(s1) as f1:
            d1 = _parse_phonopy_conf_strings(f1.readlines())
        for k0 in d0.keys():
            assert np.all(d0[k0] == d1[k0])
