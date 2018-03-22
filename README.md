# MuMoT
Multiscale Modelling Tool
---
Repository should contain following files/folders:
* `MuMoT/MuMoT.py` (main functionality)
* `process_latex/process_latex.py` (LaTeX parser, imported from [latex2sympy](https://github.com/augustt198/latex2sympy) project and updated for Python 3)
* `gen` (includes submodules used by MuMoT, important: there must be an empty file called `__init__.py` (with 2 underscores before and after init in the filename) in that folder, so Python can recognise the modules)
* `MuMoTtest.py` and other demo files

# Dependencies:
You need to install the following supporting software:

* a LaTeX compiler

## Creating a conda environment for MuMoT named "MumotEnv" (this name is specified in environment.yml):

#### The following has been tested on macOS Sierra Version 10.12.6 and Ubuntu 16.04 (Windows test to follow):

* upgrade to at least conda 4.4
* get the environment.yml file from the MuMoT repository, go to the folder containing the environment.yml file and type in terminal: `conda env create -f environment.yml`
* check that the environment has been created: `conda env list` (MumotEnv should appear in that list)
* activate the environment: `source activate MumotEnv` (on macOS/Linux) or `activate MumotEnv` (on Windows)
* check that the environment was installed correctly (after it has been activated): `conda list` (all packages in the environment.yml file should be listed)
* if installed on Linux system you need to delete the following line (on macOS you can skip this step): - appnope=0.1.0=py35_0 as it is macOS specific (this is the first line under dependencies in the environment.yml file). If you are using Ubuntu 16.04 or Mac osX the package graphviz is probably not working correctly (when installed via pip - this is done when creating the environment via the environment.yml file). This can be resolved by installing it again by typing: `sudo apt-get install graphviz` (Ubuntu) or `conda install graphviz` (Mac) in a terminal (you need to have root permissions, and make sure you have already run `activate MumotEnv`)
* start Jupyter notebook: `jupyter notebook` and run MuMoTtest.ipynb notebook
* FYI: environments can be deactivated using: `source deactivate` (on macOS/Linux) or `deactivate` (on Windows)

# Test
To test your installation run the `MuMoTuserManual.ipynb` notebook.

# Documentation
Read the documentation at [https://diodeproject.github.io/MuMoT/](https://diodeproject.github.io/MuMoT/)

# House rules
* include [Python docstrings](https://www.python.org/dev/peps/pep-0257/) at the very least for user-visible functions, but ideally also for classes, functions, etc. Include the sections `Arguments`, `Keywords` and `Returns`
* use `@todo` for reminders
* Write code using Python [naming conventions](https://www.python.org/dev/peps/pep-0008/#naming-conventions)
* Update `MuMoTtest.ipynb` to add tests for new functionality - always run this notebook before committing

