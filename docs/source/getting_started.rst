.. _getting_started:

Getting started
===============

.. todo:: Add info on test notebooks and maybe static renderings of both using the nbsphinx_ Sphinx extension?

The MuMoT user manual (a Jupyter Notebook) provides the most accessible introduction to working with MuMoT.  

MuMoT (Multiscale Modelling Tool) is a tool designed to allow sophisticated mathematical modelling and analysis, without writing equations
- the class of models that can be represented is broad, ranging from chemical reaction kinetics to demography and collective behaviour
- by using a web-based interactive interface with minimal coding, rapid development and exploration of models is facilitated
- the tool may also be particularly useful for pedagogical demonstrations

.. _mybinder_usage:

Online
------

View and interact with the user manual online: 

.. image:: https://mybinder.org/badge.svg
   :alt: Start running MuMoT user manual on mybinder.org
   :target: https://mybinder.org/v2/gh/DiODeProject/MuMoT/v1.2.1?filepath=docs%2FMuMoTuserManual.ipynb

Note that this uses the excellent and free `mybinder.org <https://mybinder.org/>`__ service,
funded by the `Gordon and Betty Moore Foundation <https://www.moore.org/>`__,
which may not always be available at times of very high demand.  

Also, note that the mybinder.org sessions may sometimes take several minutes to start:
mybinder.org will `use a cached MuMoT environment <https://binderhub.readthedocs.io/en/latest/overview.html>`__ if one is available 
but this may not always be the case
(e.g. immediately after an update to the MuMoT package).

Demo notebooks
--------------
The following demo notebooks are also available online:

* `Paper <https://mybinder.org/v2/gh/DiODeProject/MuMoT/v1.2.1?filepath=docs%2FMuMoTpaperResults.ipynb>`_: (*MuMoT authors, University of Sheffield*)
* `Epidemics <https://mybinder.org/v2/gh/DiODeProject/MuMoT/v1.2.1?filepath=DemoNotebooks%2FEpidemicsDemo_SIRI.ipynb>`_: (*Renato Pagliara, Princeton University*)
* `Agent density <https://mybinder.org/v2/gh/DiODeProject/MuMoT/v1.2.1?filepath=DemoNotebooks%2FAgent_density.ipynb>`_: (*Yara Khaluf, Ghent University*, and *MuMoT authors, University of Sheffield*)
* `COVID-19 <https://mybinder.org/v2/gh/DiODeProject/MuMoT/v1.2.1?filepath=DemoNotebooks%2FCOVID-19.ipynb>`_: (*James A. R. Marshall, University of Sheffield*)
* `Variance suppression <https://mybinder.org/v2/gh/DiODeProject/MuMoT/v1.2.1?filepath=DemoNotebooks%2FVariance_suppression.ipynb>`_: (*Andreagiovanni Reina, University of Sheffield*)
* `Michaelis-Menten <https://mybinder.org/v2/gh/DiODeProject/MuMoT/v1.2.1?filepath=DemoNotebooks%2FMichaelis-Menten_Dynamics.ipynb>`_: (*Aldo Segura, University of Sheffield*)

On your own machine
-------------------

#. Follow the :ref:`install and post-install <install>` instructions.  
   If you have already created your MuMoT conda environment or virtualenv you only need to *activate* it.
#. Start a Jupyter Notebook server, unless you have done so already:

   .. code:: sh

      jupyter notebook

#. Within the Jupyter file browser, 
   browse to a clone of this Git repository, 
   find the ``docs`` subdirectory then 
   open ``docs/MuMoTuserManual.ipynb``, 
   which is the MuMoT User Manual. To view demo notebooks navigate to ``DemoNotebooks``.


.. _nbsphinx: https://nbsphinx.readthedocs.io/en/0.3.3/
