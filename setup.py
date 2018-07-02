#!/usr/bin/env python
from distutils.core import setup

packages = [
    'upho',
    'upho.phonon',
    'upho.harmonic',
    'upho.analysis',
    'upho.structure',
    'upho.irreps',
    'upho.qpoints',
    'group',
]
scripts = [
    'scripts/upho_weights',
    'scripts/upho_sf',
    'scripts/upho_qpoints',
    'scripts/upho_fit',
]
setup(name='upho',
      version='0.5.4',
      author="Yuji Ikeda",
      author_email="y.ikeda@mpie.de",
      packages=packages,
      scripts=scripts)
