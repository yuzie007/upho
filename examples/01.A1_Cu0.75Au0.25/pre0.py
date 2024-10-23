import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import BFGS

# Ideal atomic positions with dummy species.
atoms = bulk("X", "fcc", a=3.753, cubic=True) * 3
atoms.write("POSCAR_ideal", direct=True)

# Replace 3/4 of atoms with Cu and 1/4 with Au randomly.
# (In real researches we should use more sophisticated method like SQS.)
rng = np.random.default_rng(42)
indices = np.arange(len(atoms))
rng.shuffle(indices)
atoms.symbols[indices[: 3 * len(atoms) // 4]] = "Cu"
atoms.symbols[indices[3 * len(atoms) // 4 :]] = "Au"

# Relax atomic positions.
atoms.calc = EMT()
with BFGS(atoms) as dyn:
    dyn.run(fmax=1e-6)

# Sort by symbols.
atoms = atoms[atoms.numbers.argsort()]

atoms.write("POSCAR", direct=True)

# Fill with the dummy symbol.
atoms.symbols = len(atoms) * ["X"]
