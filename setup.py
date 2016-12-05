#!/usr/bin/env python
from distutils.core import setup

packages = ['ph_unfolder',
            'ph_unfolder.phonon',
            'ph_unfolder.harmonic',
            'ph_unfolder.analysis',
            'ph_unfolder.structure',
            'ph_unfolder.irreps',
            'group',
]
scripts = [
    'scripts/upho_weights',
    'scripts/upho_sf',
]
setup(name='upho',
      version='0.4.0',
      author="Yuji Ikeda",
      author_email="ikeda.yuji.6m@kyoto-u.ac.jp",
      packages=packages,
      scripts=scripts)
