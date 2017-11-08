#!/usr/bin/env python

from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='MuMoT',
    description='Multiscale Modelling Tool',
    author='James A. R. Marshall, Andreagiovanni Reina, Thomas Bose',
    author_email='james.marshall@shef.ac.uk',
    version='0.0',
    url='https://github.com/DiODeProject/MuMoT',
    packages=['MuMoT', 'gen'],
    package_dir={'':'.'},
    package_data={'gen': ['PS.tokens','PSLexer.tokens']},
    license='GPL-3.0 (latex2sympy distributed under MIT licence)',
    install_requires=requirements)
