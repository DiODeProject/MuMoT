# MuMoT
Multiscale Modelling Tool
---
Repository should contain following files/folders:
* `MuMoT/MuMoT.py` (main functionality)
* `MuMoTtest.py` and other demo files

## Dependencies
You need to install the following supporting software:

* a LaTeX compiler

### Path
Given subfolders are used for demo and for test notebooks, the `PYTHONPATH` environment variable needs to be set appropriately, *e.g.* `export PYTHONPATH=$PYTHONPATH:<your MuMoT repository directoty` (Mac). Ideally, add this somewhere where it will be set during every session, *e.g.* in your `.bashrc` file

### Creating a conda environment for MuMoT named "MumotEnv" (this name is specified in environment.yml)

#### The following has been tested on macOS Sierra Version 10.12.6 and Ubuntu 16.04 (Windows test to follow)

* upgrade to at least conda 4.4
* get the environment.yml file from the MuMoT repository, go to the folder containing the environment.yml file and type in terminal: `conda env create -f environment.yml`
* check that the environment has been created: `conda env list` (MumotEnv should appear in that list)
* activate the environment: `source activate MumotEnv` (on macOS/Linux) or `activate MumotEnv` (on Windows)
* check that the environment was installed correctly (after it has been activated): `conda list` (all packages in the environment.yml file should be listed)
* if installed on Linux system you need to delete the following line (on macOS you can skip this step): - appnope=0.1.0=py35_0 as it is macOS specific (this is the first line under dependencies in the environment.yml file). If you are using Ubuntu 16.04 or Mac osX the package graphviz is probably not working correctly (when installed via pip - this is done when creating the environment via the environment.yml file). This can be resolved by installing it again by typing: `sudo apt-get install graphviz` (Ubuntu) or `conda install graphviz` (Mac) in a terminal (you need to have root permissions, and make sure you have already run `activate MumotEnv`)
* to enable tables of contents, **especially if running a server providing access to** `MuMoTuserManual.ipynb`, enable TOC2 as follows:
    * in the command line run `jupyter nbextensions_configurator enable --user`
    * after the notebook server is running (*e.g.* next step), enable TOC2 via the *nbextensions* tab
    * within a notebook, toggle the TOC by clicking on the appropriate button in the toolbar
* start Jupyter notebook: `jupyter notebook` and run MuMoTtest.ipynb notebook
* FYI: environments can be deactivated using: `source deactivate` (on macOS/Linux) or `deactivate` (on Windows)

## Test
To test your installation run the `MuMoTuserManual.ipynb` notebook.

## Documentation
The `MuMoTuserManual.ipynb` notebook provides the most accessible introduction to working with MuMoT.

For more technical information read the documentation at [https://diodeproject.github.io/MuMoT/](https://diodeproject.github.io/MuMoT/)

## House rules
* include [Python docstrings](https://www.python.org/dev/peps/pep-0257/) at the very least for user-visible functions, but ideally also for classes, functions, etc. Include the sections `Arguments`, `Keywords` and `Returns`
* use `@todo` for reminders
* Write code using Python [naming conventions](https://www.python.org/dev/peps/pep-0008/#naming-conventions)
* Update `TestNotebooks/MuMoTtest.ipynb` to add tests for new functionality - always run this notebook before committing

## Contributors

### Core Development Team:
* James A. R. Marshall
* Andreagiovanni Reina
* Thomas Bose

### Packaging, Documentation and Deployment:
* Will Furnass

### Windows Compatibility
* Renato Pagliara Vasquez

*Contains code snippets (C) 2012 Free Software Foundation, under the MIT Licence*

## Funding
MuMoT developed with funds from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement number 647704 - [DiODe](http://diode.group.shef.ac.uk)).
