.. _install:

Installation
============

.. contents:: :local:

Prerequisite: LaTeX
-------------------

The visualisation of particular representations of models requires that you have a `LaTeX distribution`_ installed.

* macOS: install MacTex_
* Linux: install TexLive_
* Windows: install MiKTeX_ or TexLive_
  
You can now install MuMoT :ref:`within a conda environment <conda_inst>` or :ref:`within a Python virtualenv <venv_inst>`.

.. _conda_inst:

Installing MuMoT within a Conda environment
-------------------------------------------

#. `Clone <https://help.github.com/articles/cloning-a-repository/>`__
   `this repository <https://github.com/DiODeProject/MuMoT/>`__.
#. Install the conda package manager by 
   installing a version of Miniconda_ appropriate to your operating system.
#. Open a terminal within which conda is available; 
   you can check this with

   .. code:: sh

      conda --version

#. Create a new conda environment containing just Python >=3.10 e.g.:

   .. code:: sh

      conda update conda
      conda create -n mumot-env python=3.12

#. Check that conda environment has been created: 
   

   .. code:: sh

      conda env list

   ``mumot-env`` should appear in the output.

#. *Activate* the environment:

   .. code:: sh

      source activate mumot-env    # on macOS/Linux with older versions of conda
      conda activate mumot-env    # on macOS/Linux with newer versions of conda
      activate mumot-env           # on Windows

#. *Install* MuMoT and dependencies into this conda environment:


   .. code:: sh

      conda install graphviz
      python -m pip install path/to/clone/of/MuMoT/repository

   NB if your clone of the MuMot repository is a subdirectory of the current directory,
   make sure you run ``python3 -m pip install ./MuMoT`` instead of ``python3 -m pip install MuMoT``
   (to ensure  MuMoT is installed from your Git clone and not :ref:`from PyPI<pypi_inst>`).

.. _venv_inst:

Installing MuMoT within a VirtualEnv
------------------------------------

1. `Clone <https://help.github.com/articles/cloning-a-repository/>`__
   `this repository <https://github.com/DiODeProject/MuMoT/>`__.
2. Ensure you have the following installed:

   -  `Python >= 3.10 <https://www.python.org/downloads/>`__
   -  the pip_ package
      manager (usually comes with Python 3.x but might not for certain
      flavours of Linux)
   -  the virtualenv_ tool
      for managing Python virtual environments
   -  graphviz_

   You can check this by opening a terminal and running:

   .. code:: sh

      python3 --version
      python3 -m pip --version
      python3 -m virtualenv --version
      dot -V

3. Create a Python virtualenv in your home directory:

   .. code:: sh

      cd 
      python3 -m virtualenv mumot-env

4. *Activate* this Python virtualenv:

   .. code:: sh

      source mumot-env/bin/activate    # on macOS/Linux
      mumot-env/bin/activate           # on Windows

5. *Install* MuMoT and dependencies into this Python virtualenv,
   plus a Jupyter front end if you don't already have one:

   .. code:: sh

      python3 -m pip install path/to/clone/of/MuMoT/repository
      python3 -m pip install notebook    # or jupyterlab

   NB if your clone of the MuMot repository is a subdirectory of the current directory,
   make sure you run ``python3 -m pip install ./MuMoT`` instead of ``python3 -m pip install MuMoT``
   (to ensure  MuMoT is installed from your Git clone and not :ref:`from PyPI<pypi_inst>`).

.. _pypi_inst:

Installing MuMoT from PyPI
--------------------------

Follow the instructions as above for 'Installing MuMoT within a VirtualEnv', but at stage 5 replace

.. code:: sh

      python3 -m pip install path/to/clone/of/MuMoT/repository

with

.. code:: sh

      python3 -m pip install mumot

Interactive figures
-------------------

MuMoT uses the ipympl_ (``%matplotlib widget``) Matplotlib backend,
which works in Jupyter Notebook 7+, JupyterLab and VS Code.
It is installed automatically with MuMoT;
if it is unavailable MuMoT falls back to the ``nbagg`` backend,
which only works in the classic (version 6 or earlier) Notebook.

Tables of contents for individual Notebooks
-------------------------------------------

Hyperlinked tables of contents can be useful when viewing longer Notebooks such as
the `MuMoT User Manual <docs/MuMoTuserManual.ipynb>`__.
Jupyter Notebook 7+ and JupyterLab show one in the left-hand sidebar
(*View* → *Table of Contents*); no extension is needed.

.. _LaTeX distribution: https://www.latex-project.org/get/
.. _MacTex: http://www.tug.org/mactex/
.. _MiKTeX: http://miktex.org/
.. _TexLive: http://www.tug.org/texlive
.. _pip: https://pip.pypa.io/en/stable/installing/
.. _virtualenv: https://virtualenv.pypa.io/en/stable/
.. _graphviz: https://graphviz.org/download/
.. _ipympl: https://matplotlib.org/ipympl/
.. _Miniconda: https://conda.io/miniconda.html
