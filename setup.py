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
        'ipython',
        'ipywidgets',
        'matplotlib',
        'networkx',
        'pydstool',
        'scipy<1.0.0',
        'sympy>=1.1.1',
        ],
    extras_require={
        'test': [
            'pytest',
            'pytest-cov',
            'nbval' ,
            'nbdime',
            'jupyter',
            ],
        'docs': [
            'sphinx',
            'sphinx-rtd-theme',
            'numpydoc'
        ],
    },
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)
