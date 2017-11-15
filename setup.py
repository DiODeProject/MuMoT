#!/usr/bin/env python

from setuptools import setup

setup(name='MuMoT',
      description='Multiscale Modelling Tool',
      author='James A. R. Marshall, Andreagiovanni Reina, Thomas Bose',
      author_email='james.marshall@shef.ac.uk',
      # @todo: switch to using semantic versioning
      # (https://packaging.python.org/tutorials/distributing-packages/#semantic-versioning-preferred)
      version='0.0.0',
      url='https://github.com/DiODeProject/MuMoT',
      packages=['MuMoT'],
      package_dir={'': '.'},
      python_requires='>=3',
      license='GPL-3.0',
      # Use patched latex2sympy whilst waiting for https://github.com/jackatbancast/latex2sympy/pull/1 to be merged
      dependency_links=['https://github.com/willfurnass/latex2sympy/tarball/BUG_ensure_gen_pkg_installed#egg=latex2sympy-0.0.1-patched'],
      install_requires=[
          'graphviz',
          'ipython',
          'ipywidgets',
          'matplotlib',
          'networkx',
          'pydstool',
          'scipy<1.0.0',
          'sympy>=1.1.1',
          'latex2sympy==0.0.1-patched',
          ],
      )
