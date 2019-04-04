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

#. Create a new conda environment containing just Python 3.6:

   .. code:: sh

      conda update conda
      conda create -n mumot-env python=3.6

#. Check that conda environment has been created: 
   

   .. code:: sh

      conda env list

   ``mumot-env`` should appear in the output.

#. *Activate* the environment:

   .. code:: sh

      source activate mumot-env    # on macOS/Linux
      activate mumot-env           # on Windows

#. *Install* MuMoT and dependencies into this conda environment:

   .. code:: sh

      conda install graphviz
      python -m pip install path/to/clone/of/MuMoT/repository


.. _venv_inst:

Installing MuMoT within a VirtualEnv
------------------------------------

1. `Clone <https://help.github.com/articles/cloning-a-repository/>`__
   `this repository <https://github.com/DiODeProject/MuMoT/>`__.
2. Ensure you have the following installed:

   -  `Python >= 3.4 and <= 3.6 <https://www.python.org/downloads/>`__
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

5. *Install* MuMoT and dependencies into this Python virtualenv, then
   enable interactive Notebook widgets:

   .. code:: sh

      python -m pip install path/to/clone/of/MuMoT/repository
      jupyter nbextension enable --py widgetsnbextension --sys-prefix

Installing MuMoT from PyPI
--------------------------

Follow the instructions as above for 'Installing MuMoT within a VirtualEnv', but at stage 5 replace

.. code:: sh

      python -m pip install path/to/clone/of/MuMoT/repository

with

.. code:: sh

      python -m pip install mumot

(Optional) Enable tables of contents for individual Notebooks
-------------------------------------------------------------

Hyperlinked tables of contents can be userful when viewing longer Notebooks such as 
the `MuMoT User Manual <docs/MuMoTuserManual.ipynb>`__.

Tables of contents can be displayed if you enable the **TOC2** Jupyter Extension as follows:

#. Ensure the ``jupyter_contrib_nbextensions`` package is installed.
   This is "a collection of extensions that add functionality to the Jupyter notebook". 
   If you installed MuMoT into a *virtualenv* using **pip** then 
   you need to ensure that virtualenv is activated before running:

   .. code:: sh

      pip install jupyter_contrib_nbextensions

#. Enable ``jupyter_contrib_nbextensions``:

   .. code:: sh

      jupyter contrib nbextension install --sys-prefix

#. Enable the TOC2 ('table of contents') extension that is 
   provided by ``jupyter_contrib_nbextensions``:

   .. code:: sh

      jupyter nbextension enable toc2/main

#. Enable a graphical interface for enabling/disabling TOC2 and other
   Jupyter extensions. If using conda:

   .. code:: sh

      conda install -c conda-forge jupyter_nbextensions_configurator

   Or if using a virtualenv instead:

   .. code:: sh

      pip install jupyter_nbextensions_configurator  # AND 
      jupyter nbextensions_configurator enable --sys-prefix

The next time you start Jupyter from your conda environment or virtualenv then open a Notebook 
you should see a table of contents displayed down the left-hand-side of the Notebook.

If you subsequently want to disable the TOC2 extension 
and/or enable other Notebook extensions 
then click *Nbextensions* in the Jupyter file browser tab.

.. _LaTeX distribution: https://www.latex-project.org/get/
.. _MacTex: http://www.tug.org/mactex/
.. _MiKTeX: http://miktex.org/
.. _TexLive: http://www.tug.org/texlive
.. _pip: https://pip.pypa.io/en/stable/installing/
.. _virtualenv: https://virtualenv.pypa.io/en/stable/
.. _graphviz: https://graphviz.gitlab.io/download/
.. _Miniconda: https://conda.io/miniconda.html
