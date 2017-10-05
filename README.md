# MuMoT
Multiscale Modelling Tool
---
Repository should contain following files/folders:
* `MuMoT/MuMoT.py` (main functionality)
* `process_latex/process_latex.py` (LaTeX parser, imported from [latex2sympy](https://github.com/augustt198/latex2sympy) project and updated for Python 3)
* `gen` (includes submodules used by MuMoT, important: there must be an empty file called `__init__.py` (with 2 underscores before and after init in the filename) in that folder, so Python can recognise the modules)
* `MuMoTtest.py` and other demo files

# Dependencies:
You need to install the following tools: PyDSTool, graphviz (graph visualization) and antlr4 4.5.3 (parser generator). It is possible to use pip. Open a terminal window and type:

* `pip install pydstool`
* `pip install graphviz`
* `pip install antlr4-python3-runtime=4.5.3`

# Test
To test your installation run the `MuMoTdemo.ipynb` and `MuMoTtest.ipynb` notebooks.

# Documentation
Read the documentation at [https://diodeproject.github.io/MuMoT/](https://diodeproject.github.io/MuMoT/)

# House rules
* Update the [Doxygen](http://www.stack.nl/~dimitri/doxygen/index.html) documentation when making substantive changes
  * include [Python comment blocks](http://www.stack.nl/~dimitri/doxygen/manual/docblocks.html#pythonblocks) for classes, functions, etc.
  * use `@todo` for reminders
  * [download](http://www.stack.nl/~dimitri/doxygen/download.html) Doxygen
  * run locally with `Doxyfile` configuration file from repository
  * commit the contents of `docs/`

* Update `MuMoTtest.ipynb` to add tests for new functionality - always run this notebook before committing
* Write code using Python [naming conventions](https://www.python.org/dev/peps/pep-0008/#naming-conventions)

