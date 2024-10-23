# UPHO

[![GitHubActions](https://github.com/yuzie007/upho/actions/workflows/tests.yml/badge.svg)](https://github.com/yuzie007/upho/actions?query=workflow%3ATests)

Band unfolding for phonons (http://yuzie007.github.io/upho/)

## Requirements

- numpy
- h5py
- phonopy>=2.7.0

## Install

```bash
pip install git+https://github.com/yuzie007/upho.git@v0.6.6
```

## Usage and Tutorial

See the `examples` directory.

## Options (`upho_weights`)

### `--average_masses`

Atomic masses whose sites are equivalent in the underlying structure
are averaged.

### `--average_force_constants`

FC elements which are equivalent under the symmetry operations
for the underlying structure are averaged.

## Options (`upho_sf`)

### `-f FILENAME`, `--filename FILENAME`

Filename for the data of weights.

### `--format {hdf5,text}`

Output file format.

### `--function {gaussian,lorentzian}`

Function used for the smearing.

### `-s SIGMA`, `--sigma SIGMA`

Parameter for the smearing function (THz).
For Gaussian, this is the standard deviation.
For Lorentzian, this is the HWHM (gamma).

### `--fmax FMAX`

Maximum frequency (THz).

### `--fmin FMIN`

Minimum frequency (THz).

### `--fpitch FPITCH`

Frequency pitch (THz).

### `--squared`

Use squared frequencies instead of raw frequencies.

## Not yet (possible bugs)

(Projective) representations of little cogroup may be treated in a wrong way
when we consider wave vectors on the BZ boundary and translational parts of
symmetry operations are not equal to zero.

## Author(s)

Yuji Ikeda (yuji.ikeda.ac.jp@gmail.com, Universität Stuttgart, Germany)

## How to cite

When using this code, please cite the following article.

    *Mode decomposition based on crystallographic symmetry in the band-unfolding method*,
    Yuji Ikeda, Abel Carreras, Atsuto Seko, Atsushi Togo, and Isao Tanaka,
    Phys. Rev. B **95**, 024305 (2017).
    http://journals.aps.org/prb/abstract/10.1103/PhysRevB.95.024305

For high entropy alloy works, you can also consider

    *Phonon Broadening in High Entropy Alloys*,
    Fritz Körmann, Yuji Ikeda, Blazej Grabowski, and Marcel H. F. Sluiter,
    Npj Comput. Mater. **3**, 1 (2017).
    https://www.nature.com/articles/s41524-017-0037-8
