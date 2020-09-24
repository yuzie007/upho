import re
import numpy as np


def parse_phonopy_conf(filename):
    with open(filename, "r") as f:
        return _parse_phonopy_conf_strings(f.readlines())


def _parse_phonopy_conf_strings(lines):
    lines = _remove_comments(lines)
    lines = _parse_continuation_lines(lines)
    d = _create_dictionary(lines)
    return _parse_dictionary(d)


def _remove_comments(lines):
    return [re.sub('#.*', '', _) for _ in lines]


def _parse_continuation_lines(lines):
    lines_new = []
    tmp = ""
    for line in lines:
        end = line.find('+++')
        if end == -1:
            lines_new.append(tmp + line)
            tmp = ""
        else:
            tmp += line[:end]
    return lines_new


def _create_dictionary(lines):
    d = dict()
    for line in lines:
        if line.find('=') != -1:
            tmp = [x.strip() for x in line.split("=")]
            d[tmp[0].lower()] = tmp[1]
    return d


def _parse_dictionary(d):
    d_new = dict()
    for k, v in d.items():
        if k == 'dim':
            d_new['supercell_matrix'] = _parse_dim(v)
        elif k in ('primitive_axis', 'primitive_axes'):
            k_new = 'primitive_matrix'
            if v.lower() == 'auto':
                d_new['is_primitive_auto'] = True
                d_new[k_new] = v
            else:
                d_new['is_primitive_auto'] = False
                d_new[k_new] = _parse_primitive_axes(v)
        elif k == 'band':
            if v.lower() == 'auto':
                d_new['is_band_auto'] = True
                d_new['band_paths'] = v
            else:
                d_new['is_band_auto'] = False
                d_new['band_paths'] = _parse_band(v)
        elif k in ('band_points', ):
            d_new[k] = int(v)
        else:
            d_new[k] = v
    return d_new


def _parse_dim(v):
    matrix = [int(_) for _ in v.split()]
    length = len(matrix)
    if length == 9:
        return np.array(matrix).reshape(3, 3)
    elif length == 3:
        return np.diag(matrix)
    else:
        raise ValueError('Number of elements of DIM tag has to be 3 or 9.')


def _parse_primitive_axes(v):
    k = 'PRIMITIVE_AXES'
    matrix = [float(eval(_)) for _ in v.split()]
    length = len(matrix)
    if length == 9:
        matrix = np.array(matrix).reshape(3, 3)
        if np.linalg.det(matrix) < 1e-8:
            raise ValueError('{} has to have positive determinant.'.format(k))
        return matrix
    else:
        raise ValueError('Number of elements in {} has to be 9.'.format(k))


def _parse_band(v):
    bands = []
    for section in v.split(','):
        points = [eval(x) for x in section.split()]
        if len(points) % 3 != 0 or len(points) < 6:
            raise ValueError("BAND is incorrectly set.")
        bands.append(np.array(points).reshape(-1, 3))
    return bands
