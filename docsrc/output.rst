############
Output files
############

``band.hdf5``
=============

.. code-block:: console

    band.hdf5
    ├── paths (Dataset {3, 101, 3})
    ├── 0 (Group)
    │   ├── 0 (Group)
    │   ├── 1 (Group)
    │   │   ├── point             (Dataset {3})
    │   │   ├── q_star            (Dataset {6, 3})
    │   │   ├── distance          (Dataset {SCALAR})
    │   │   ├── natoms_primitive  (Dataset {SCALAR})
    │   │   ├── elements          (Dataset {2})
    │   │   ├── num_arms          (Dataset {SCALAR})
    │   │   ├── pointgroup_symbol (Dataset {SCALAR})
    │   │   ├── num_irreps        (Dataset {SCALAR})
    │   │   ├── ir_labels         (Dataset {10})
    │   │   ├── frequencies       (Dataset {6, 12})
    │   │   ├── weights_t         (Dataset {6, 12})
    │   │   ├── weights_e         (Dataset {6, 1, 2, 1, 2, 12})
    │   │   ├── weights_s         (Dataset {6, 5, 12})
    │   │   ├── weights_s_e       (Dataset {6, 5, 1, 2, 1, 2, 12})
    │   │   └── weights_e2        (Dataset {6, 1, 2, 12})
    │   ├── 2 (Group)
    │   .
    │   .
    │   .
    │
    ├── 1 (Group)
    │   ├── 0 (Group)
    │   ├── 1 (Group)
    │   ├── 2 (Group)
    │   .
    │   .
    │   .
    │
    ├── 2 (Group)
    │   ├── 0 (Group)
    │   ├── 1 (Group)
    │   ├── 2 (Group)
    │   .
    │   .
    │   .


``paths``
---------

Array with the shape of ``(NPATHS, BAND_POINTS, 3)``.

Q-point sets and the shape of band paths.

``NPATHS`` corresponds to the number of 3-integer sets in ``BAND`` in ``band.conf``.

``BAND_POINTS`` corresponds to that in ``band.conf``.

Each group corresponds to one of the q-points given here.

``point``
---------

Reciprocal space point in fractional coordinates to be computed.

``q_star``
----------

Star (symmetrically equivalent set of points) of ``point`` in the ideal lattice considered in the calculations.
The actual behavior depends on the value of ``star`` in ``input.json``.

- ``sym`` (default): Symmetrically euqivalent q-points, w/o duplications, are considered.
- ``all``: Symmetrically equivalent q-points, possibly duplicated due to symmetry operations. Maybe used for testing purposes.
- ``none``: Only the given q-point is considered.

``distance``
------------

Distance of the considered reciprocal-space point from the first one along the band paths in fractional coordinates.

``natoms_primitive``
--------------------

The number of atoms in the primitive cell of the ideal lattice. (``natoms_p`` in ``eigenstates.py``.)

``elements``
------------

Chemical elements in the disordered-alloy model.
The number of the chemical elements are ``nelms`` in ``eigenstates.py``.

``num_arms``
------------

The number of the arms of the star.
Specifically this equals to ``len(q_star)``.

``pointgroup_symbol``
---------------------

Point group symbol in the Hermann–Mauguin notation associated with the little co-group of the considered reciprocal-space point.

``num_irreps``
--------------

The number of irreducible representations of the little co-group.

``ir_labels``
-------------

Labels of the irreducible representations.

``frequencies``
---------------

Array with the shape of ``(num_arms, nfreqs)``, where ``nfreqs`` is equal to ``3 * natoms_primitive``.

Phonon frequencies projected onto this q-point.

``weights_t``
-------------

Array with the shape of ``(num_arms, nfreqs)``.

Weights of the phonon frequencies determined by the band unfolding.

``weights_e``
-------------

Array with the shape of ``(num_arms, natoms_p, nelms, natoms_p, nelms, nfreqs)``.

Partial weights for chemical pairs.

``weights_s``
-------------

Array with the shape of ``(num_arms, num_irreps, nfreqs)``.

Partial weights for irreducible representations.
Introduced in
`[Y. Ikeda, A. Carreras, A. Seko, A. Togo, and I. Tanaka, Phys. Rev. B 95, 24305 (2017)
<https://doi.org/10.1103/PhysRevB.95.024305>`_].

``weights_s_e``
---------------

Array with the shape of
``(num_arms, num_irreps, natoms_p, nelms, natoms_p, nelms, nfreqs)``.

Partial weights for irreducible representations and for chemical pairs.

``weights_e2``
--------------

Array with the shape of
``(num_arms, natoms_p, nelms, nfreqs)``.

Partial weights for chemical elements. Introduced in
[`F. Körmann, Y. Ikeda, B. Grabowski, and M. H. F. Sluiter, Npj Comput. Mater. 3, 36 (2017)
<https://doi.org/10.1038/s41524-017-0037-8>`_].
