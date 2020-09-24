import h5py
from phonopy.units import VaspToTHz
from upho.phonon.eigenstates import Eigenstates


class SinglePoint:
    def __init__(self,
                 qpoint,
                 distance,
                 dynamical_matrix,
                 unitcell_ideal,
                 primitive_matrix_ideal,
                 factor=VaspToTHz,
                 star="none",
                 mode="eigenvector",
                 verbose=False):

        self._qpoint = qpoint
        self._distance = distance

        self._factor = factor

        self._eigenstates = Eigenstates(
            dynamical_matrix,
            unitcell_ideal,
            primitive_matrix_ideal,
            mode=mode,
            star=star,
            verbose=verbose)

        with h5py.File('point.hdf5', 'w') as f:
            self._hdf5_file = f
            self.run()

    def run(self):
        qpoint = self._qpoint
        distance = self._distance

        eigenstates = self._eigenstates

        eigenstates.set_distance(distance)
        eigenstates.extract_eigenstates(qpoint)

        eigenstates.write_hdf5(self._hdf5_file, group='')
