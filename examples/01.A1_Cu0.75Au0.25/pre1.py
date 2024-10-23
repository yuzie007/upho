from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import read
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.file_IO import write_FORCE_CONSTANTS

atoms_ref = read("POSCAR")

sets_of_forces = []
unitcell = PhonopyAtoms(
    symbols=atoms_ref.symbols,
    cell=atoms_ref.cell,
    scaled_positions=atoms_ref.get_scaled_positions(),
)
phonon = Phonopy(unitcell)
phonon.generate_displacements(distance=0.03)
supercells = phonon.supercells_with_displacements
for i, phonopy_atoms in enumerate(supercells):
    print(i)
    atoms = Atoms(
        symbols=phonopy_atoms.symbols,
        cell=phonopy_atoms.cell,
        pbc=True,
        scaled_positions=phonopy_atoms.scaled_positions,
    )
    atoms.calc = EMT()
    sets_of_forces.append(atoms.get_forces())
phonon.forces = sets_of_forces
phonon.produce_force_constants()
write_FORCE_CONSTANTS(phonon.force_constants)
