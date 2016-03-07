from distutils.core import setup
packages = ['ph_unfolder',
            'ph_unfolder.phonon',
            'ph_unfolder.analysis',
            'ph_unfolder.structure']
scripts = ['scripts/ph_unfolder']
setup(name='ph_unfolder',
      version='0.0.0',
      author="Yuji Ikeda",
      author_email="ikeda.yuji.6m@kyoto-u.ac.jp",
      packages=packages,
      scripts=scripts)
