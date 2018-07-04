from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='mumot',
    # @todo: switch to using semantic versioning
    # (https://packaging.python.org/tutorials/distributing-packages/#semantic-versioning-preferred)
    version='0.0.0',
    author='James A. R. Marshall, Andreagiovanni Reina, Thomas Bose',
    author_email='james.marshall@shef.ac.uk',
    description='Multiscale Modelling Tool',
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
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)
