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
    'scripts/qpoints',
]
setup(name='upho',
      version='0.5.1',
      author="Yuji Ikeda",
      author_email="ikeda.yuji.6m@kyoto-u.ac.jp",
      packages=packages,
      scripts=scripts,
      install_requires=['numpy'])
