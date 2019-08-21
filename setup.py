import io
import os

from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

pkgname = 'mumot'

version_namespace = {}
here = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, pkgname, '_version.py'), encoding='utf8') as f:
    exec(f.read(), {}, version_namespace)


setup(
    name=pkgname,
    description='Multiscale Modelling Tool',
    version=version_namespace['__version__'],
    author='James A. R. Marshall, Andreagiovanni Reina, Thomas Bose',
    author_email='james.marshall@shef.ac.uk',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/DiODeProject/MuMoT',
    packages=find_packages(),
    package_dir={'': '.'},
    package_data={'mumot.gen': ['PS.tokens', 'PSLexer.tokens']},
    license='GPL-3.0',
    install_requires=[
        'antlr4-python3-runtime==4.5.3',
        'graphviz',
        'ipykernel<4.7',  # needed so no duplicate figures when wiggle ipywidgets
        'ipython',
        'ipywidgets',
        'matplotlib',
        'networkx',
        'notebook<5.5',  # needed if using pyzmq < 17
        'pydstool>=0.90.3',  # min version that allows scipy >= 1.0.0 to be used
        'pyzmq<17',  # needed if using tornado < 5
        'scipy',
        'sympy >= 1.1.1, < 1.3',  # see https://github.com/DiODeProject/MuMoT/issues/170
        'tornado<5'  # needed to avoid errors with older ipykernel
    ],
    extras_require={
        'test': [
            'pytest',
            'pytest-cov',
            'nbval',
            'nbdime',
            'jupyter',
        ],
        'docs': [
            'sphinx',
        ],
    },
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
