UPHO
====

Band unfolding for phonons <http://yuzie007.github.io/upho/>

Requirements
------------

* numpy
* h5py
* phonopy

Install
-------

1.  Download the latest version from <https://github.com/yuzie007/upho/releases>

2.  Run setup.py like::

        python setup.py install --home=.

3.  Put 'upho/lib/python' into ``$PYTHONPATH``.

Usage and Tutorial
------------------

Here we consider the hypothetical case when Cu_3Au with the L2_1 structure is regarded as a random configuration
of the A1 (fcc) structure.

1.  Create FORCE_SETS file for the structure (maybe including disordered chemical configuration)
    you want to investigate using ``phonopy`` in an usual way.
    Be careful that the number of the structures with atomic displacements to get FORCE_SETS can be huge (~100)
    for a disordered configuration.

2.  Create FORCE_CONSTANTS file from FORCE_SETS file using phonopy as::

        phonopy writefc.conf

    where writefc.conf is a text file like::

        FORCE_CONSTANTS = WRITE
        DIM = 2 2 2

    ``DIM`` must be the same as that what you used to get FORCE_SETS.

3.  Prepare two VASP-POSCAR-type files, "POSCAR" and "POSCAR_ideal".
    POSCAR includes the original chemical configuration, which may be disordered.::

        Cu Au
           1.00000000000000
             3.7530000000000001    0.0000000000000000    0.0000000000000000
             0.0000000000000000    3.7530000000000001    0.0000000000000000
             0.0000000000000000    0.0000000000000000    3.7530000000000001
           Cu   Au
             3     1
        Direct
          0.0000000000000000  0.5000000000000000  0.5000000000000000
          0.5000000000000000  0.0000000000000000  0.5000000000000000
          0.5000000000000000  0.5000000000000000  0.0000000000000000
          0.0000000000000000  0.0000000000000000  0.0000000000000000

    Note that although FORCE_CONSTANTS may be obtained using relaxed atomic positions,
    here the positions must be the ideal ones.

    POSCAR_ideal is the ideal configuration, from which the crystallographic symmetry is extracted.::

        X
           1.00000000000000
             3.7530000000000001    0.0000000000000000    0.0000000000000000
             0.0000000000000000    3.7530000000000001    0.0000000000000000
             0.0000000000000000    0.0000000000000000    3.7530000000000001
            X
             4
        Direct
          0.0000000000000000  0.5000000000000000  0.5000000000000000
          0.5000000000000000  0.0000000000000000  0.5000000000000000
          0.5000000000000000  0.5000000000000000  0.0000000000000000
          0.0000000000000000  0.0000000000000000  0.0000000000000000

    In this file I recommend to  use dummy symbols like 'X' to avoid confusion.

4.  Prepare ``band.conf`` file including something like::

        DIM =  2 2 2
        PRIMITIVE_AXIS =  0 1/2 1/2  1/2 0 1/2  1/2 1/2 0
        BAND =   0 0 0  0 1/2 1/2, 1 1/2 1/2  0 0 0  1/2 1/2 1/2
        BAND_POINTS = 101
        BAND_LABELS =  \Gamma X \Gamma L
        FORCE_CONSTANTS = READ

    The style is very similar to that of phonopy conf files, but be careful about the following tags.

    ``DIM`` describes the expansion from the original POSCAR to the POSCARs with atomic displacements used to get FORCE_SETS.
    Therefore, this should be the same as the phonopy option when creating the structures with atomic displacements (1).

    ``PRIMITIVE_AXIS`` is the conversion matrix from POSCAR_ideal to the the primitive cell you expect.

4.  Run::

        /path/to/upho/scripts/upho_weights band.conf

    then you hopefully get ``band.hdf5`` file.

5.  Run::

        /path/to/upho/scripts/upho_sf --fpitch 0.01 -s 0.05 --function lorentzian --nosquared --format text

    then you hopefully get ``sf.dat`` file.
    In this file, the first, second, and third columns are for distances in reciprocal space, frequencies,
    and the values of spectral functions, respectively.

Options (upho_weights)
----------------------

--average_masses
^^^^^^^^^^^^^^^^

Atomic masses whose sites are equivalent in the underlying structure
are averaged.

--average_force_constants
^^^^^^^^^^^^^^^^^^^^^^^^^

FC elements which are equivalent under the symmetry operations
for the underlying structure are averaged.

Options (upho_sf)
-----------------

-f FILENAME, --filename FILENAME
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Filename for the weights data.

--format {hdf5,text}
^^^^^^^^^^^^^^^^^^^^
Output file format.

--function {gaussian,lorentzian}
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Function used for the smearing.

-s SIGMA, --sigma SIGMA
^^^^^^^^^^^^^^^^^^^^^^^
Paramter for the smearing function (THz).
For Gaussian, this is the standard deviation.
For Lorentzian, this is the HWHM (gamma).

--fmax FMAX
^^^^^^^^^^^
Maximum frequency (THz).

--fmin FMIN
^^^^^^^^^^^
Minimum frequency (THz).

--fpitch FPITCH
^^^^^^^^^^^^^^^
Frequency pitch (THz).

--nosquared
^^^^^^^^^^^
Use raw frequencies instead of Squared frequencies.

Not yet (possible bugs)
-----------------------
(Projective) representations of little cogroup may be treated in a wrong way
when we consider wave vectors on the BZ boundary and translational parts of
symmetry operations are not equal to zero.

Author(s)
---------
Yuji IKEDA (Kyoto University, Japan)

How to cite
-----------

When using this code, please cite the following article.

    *Mode decomposition based on crystallographic symmetry in the band-unfolding method*,
    Yuji Ikeda, Abel Carreras, Atsuto Seko, Atsushi Togo, and Isao Tanaka,
    Phys. Rev. B **95**, 024305 (2017).
    http://journals.aps.org/prb/abstract/10.1103/PhysRevB.95.024305
