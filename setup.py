#!/usr/bin/env python
from distutils.core import setup

packages = ['upho',
            'upho.phonon',
            'upho.harmonic',
            'upho.analysis',
            'upho.structure',
            'upho.irreps',
            'group',
]
scripts = [
    'scripts/upho_weights',
    'scripts/upho_sf',
]
setup(name='upho',
      version='0.5.0',
      author="Yuji Ikeda",
      author_email="ikeda.yuji.6m@kyoto-u.ac.jp",
      packages=packages,
      scripts=scripts)
