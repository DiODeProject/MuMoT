# MuMoT
Multiscale Modelling Tool
---

Repository should contain following files/folders:

* `mumot/__init__.py` (main functionality)
* `process_latex/process_latex.py` (LaTeX parser, imported from [latex2sympy](https://github.com/augustt198/latex2sympy) project and updated for Python 3)
* `gen` (includes submodules used by MuMoT, important: there must be an empty file called `__init__.py` (with 2 underscores before and after init in the filename) in that folder, so Python can recognise the modules)
* `MuMoTtest.py` and other demo files

## Dependencies

You need to install the following supporting software:

* a LaTeX compiler

### Path

Given subfolders are used for demo and for test notebooks, the `PYTHONPATH` environment variable needs to be set appropriately, *e.g.* `export PYTHONPATH=$PYTHONPATH:<your MuMoT repository directoty` (Mac). Ideally, add this somewhere where it will be set during every session, *e.g.* in your `.bashrc` file

### Creating a conda environment for MuMoT named "MumotEnv" (this name is specified in environment.yml)

*The following has been tested on macOS Sierra Version 10.12.6 and Ubuntu 16.04; Windows test to follow*.

* upgrade to at least conda 4.4
* get the environment.yml file from the MuMoT repository, go to the folder containing the environment.yml file and type in terminal: `conda env create -f environment.yml`
* check that the environment has been created: `conda env list` (MumotEnv should appear in that list)
* activate the environment: `source activate MumotEnv` (on macOS/Linux) or `activate MumotEnv` (on Windows)
* check that the environment was installed correctly (after it has been activated): `conda list` (all packages in the environment.yml file should be listed)
* if installed on Linux system you need to delete the following line (on macOS you can skip this step): - appnope=0.1.0=py35_0 as it is macOS specific (this is the first line under dependencies in the environment.yml file). If you are using Ubuntu 16.04 or Mac osX the package graphviz is probably not working correctly (when installed via pip - this is done when creating the environment via the environment.yml file). This can be resolved by installing it again by typing: `sudo apt-get install graphviz` (Ubuntu) or `conda install graphviz` (Mac) in a terminal (you need to have root permissions, and make sure you have already run `activate MumotEnv`)
* to enable tables of contents, **especially if running a server providing access to** `MuMoTuserManual.ipynb` (see the `docs` directory), enable TOC2 as follows:
    * in the command line run `jupyter nbextensions_configurator enable --user`
    * after the notebook server is running (*e.g.* next step), enable TOC2 via the *nbextensions* tab
    * within a notebook, toggle the TOC by clicking on the appropriate button in the toolbar
* start Jupyter notebook: `jupyter notebook` and run MuMoTtest.ipynb notebook
* FYI: environments can be deactivated using: `source deactivate` (on macOS/Linux) or `deactivate` (on Windows)

jupyter nbextension enable --py widgetsnbextension --sys-prefix

## Testing

At present, MuMoT is tested by running several Jupyter notebooks:

* [docs/MuMoTuserManual.ipynb](docs/MuMoTuserManual.ipynb)
* [TestNotebooks/MuMoTtest.ipynb](TestNotebooks/MuMoTtest.ipynb)
* Further test notebooks in [TestNotebooks/MiscTests/](TestNotebooks/MiscTests)

To locally automate the running of all of the above Notebooks in an isolated Python environment containing just the necessary dependencies:

 1. Install the [tox](https://tox.readthedocs.io/en/latest/) automated testing tool
 2. Run 

    ```sh
    tox
    ```

This:
 
 1. Creates a new virtualenv (Python virtual environment) containing just 
      * MuMoT's dependencies  (see `install_requires` in <setup.py>)
      * the packages needed for testing (see `extras_require` in <setup.py>)
 1. Checks that all of the above Notebooks can be run without any unhandled Exceptions or Errors being generated 
    (using [nbval](https://github.com/computationalmodelling/nbval)).
    If an Exception/Error is encountered then a Jupyter tab is opened in the default web browser showing its location 
    (using [nbdime](https://nbdime.readthedocs.io/en/stable/)).
 1. Checks that the [docs/MuMoTuserManual.ipynb](docs/MuMoTuserManual.ipynb) and [TestNotebooks/MuMoTtest.ipynb](TestNotebooks/MuMoTtest.ipynb) Notebooks 
    generate the same output cell content as is saved in the Notebook files when re-run 
    (again, using [nbval](https://github.com/computationalmodelling/nbval)).
    If a discrepency is encountered then a Jupyter tab is opened in the default web browser showing details 
    (again, using [nbdime](https://nbdime.readthedocs.io/en/stable/)).
    
## Documentation

The [docs/MuMoTuserManual.ipynb](docs/MuMoTuserManual.ipynb) Notebook provides the most accessible introduction to working with MuMoT.

For more technical information read the documentation at [https://diodeproject.github.io/MuMoT/](https://diodeproject.github.io/MuMoT/)

## Contributing

If you want to contribute a feature or fix a bug then

* Fork this repository and create a [feature branch](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow)
* Make commits to that branch
    * Style: write code using Python [naming conventions](https://www.python.org/dev/peps/pep-0008/#naming-conventions)
    * Testing: add new test functions to [TestNotebooks/MuMoTtest.ipynb](TestNotebooks/MuMoTtest.ipynb) or test notebooks to [TestNotebooks/MiscTests/](TestNotebooks/MiscTests) to test your new features or bug fixes. 
    * Documentation: Include Python docstrings documentation in the [numpydoc](http://numpydoc.readthedocs.io/en/latest/format.html) format for all modules, functions, classes, methods and (if applicable) attributes
    * Use `@todo` for reminders
* When you are ready to merge that into the `master` branch of the 'upstream' repository:
    * Testing: run all tests using `tox` (see [Testing](#testing)).


## Contributors

### Core Development Team:

* James A. R. Marshall
* Andreagiovanni Reina
* Thomas Bose

### Packaging, Documentation and Deployment:

* [Will Furnass](http://learningpatterns.me)

### Windows Compatibility

* Renato Pagliara Vasquez

*Contains code snippets (C) 2012 Free Software Foundation, under the MIT Licence*

## Funding

MuMoT developed with funds from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement number 647704 - [DiODe](http://diode.group.shef.ac.uk)).
