Development
===========

.. contents:: :local:

Reporting issues
----------------

If your issue (feature request or bug) has not already been filed in the MuMoT GitHub repository 
(`list of all open issues <https://github.com/DiODeProject/MuMoT/issues>`__)
then please `file a new Issue <https://help.github.com/articles/creating-an-issue>`__ 
against the `MuMoT GitHub repository`_.

.. _cont_wflow:

Contribution workflow
---------------------

If you want to contribute a feature or fix a bug then:

#. `Fork <https://help.github.com/articles/fork-a-repo/>`__ the `MuMoT GitHub repository`_ 
   so you have your own personal MuMoT repository on GitHub.
#. Create a `feature branch <https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow>`__ 
   off the master branch within your personal MuMoT repository.
#. Make commits to that branch:

   * Style: write code using `standard Python naming conventions <https://www.python.org/dev/peps/pep-0008/#naming-conventions>`__.
   * Testing: if you add new features, fix bug(s) or change existing functionality:

     * Add (lower-level) `unit tests <https://en.wikipedia.org/wiki/Unit_testing>`__ to 
       the Python source files in the ``tests/`` directory.
     * Add (higher-level) `acceptance <https://en.wikipedia.org/wiki/Acceptance_testing>`__/`regression <https://en.wikipedia.org/wiki/Regression_testing>`__ tests 
       to ``TestNotebooks/MuMoTtest.ipynb`` (or to/as Notebooks in the ``TestNotebooks/MiscTests/`` directory).

   * Documentation: include Python docstrings documentation in the numpydoc_ format for all modules, functions, classes, methods and (if applicable) attributes.
   * Do not commit an updated User Manual Notebook containing output cells; all output cells should be stripped first using:

     .. code:: sh

        jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/MuMoTuserManual.ipynb

     Note that the test suite checks this automatically.

   * Use comments containing ``@todo`` in Python code for reminders.

#. When you are ready to merge your feature branch into the master branch of the 'upstream' repository: 

   #. Run all tests using tox (see Testing_) first.
   #. Create a `Pull Request`_ to request that 
      your feature branch be merged into the master branch of the upstream repository. 
      Automated tests will then run against your branch (using Travis) 
      and your code will be reviewed by the core development team, 
      after which your changes will be merged into the main MuMoT repository *or* 
      you will be asked to make futher changes to your branch.

.. _testing:

Setting up a local development environment
------------------------------------------

Follow the :ref:`install instructions <install>` but ensure you run:

.. code:: sh

   python -m pip install path/to/clone/of/MuMoT/repository[test,docs]

instead of just ``python -m pip install path/to/clone/of/MuMoT/repository``.
The '``[test,docs]``' bit ensures that the optional dependencies required to run tests and build the documentation are installed.

If you make local changes to the MuMoT Python package and want to use the the updated package you should

#. Re-install the package within your conda environment or virtualenv (without upgrading MuMoT's dependencies):

   .. code:: sh

      python -m pip install --upgrade --no-deps path/to/clone/of/MuMoT/repository[test,docs]

#. Restart any running IPython kernels within which you have imported the MuMoT package.

Testing
-------

.. _test_suite:

Test suite
^^^^^^^^^^

Testing of MuMoT is currently very basic; 
the test suite only checks that certain Jupyter Notebooks run without failing i.e. there are nno checks for correctness of results.
However, there is a framework in place to allow more tests to be written:

* **Unit tests**: run by pointing pytest_ at the ``tests/`` directory; also generates a code coverage data using pytest-cov_; *no tests implemented yet*.
* **Basic integration tests**: 
  Ensure that certain Jupyter Notebooks can be run without 
  raising Python exceptions/errors:

   * ``docs/MuMoTuserManual.ipynb``
   * ``TestNotebooks/MuMoTtest.ipynb``

  These tests are performed by running the Notebooks using the nbval_ plug-in for pytest_, with nbval_ being run in *lax* mode.
  Code coverage data is also captured at this stage when running ``TestNotebooks/MuMoTtest.ipynb`` and 
  appended to that captured during the unit testing.
* **Regression tests**: 
* Ensure that the ``TestNotebooks/MuMoTtest.ipynb`` integration test Notebook 
  generates sufficiently similar output cells to those saved in that file 
  when re-run in a clean environment; 
  *not yet implemented* but could be performed by running the Notebook using the nbval_ plug-in for pytest_, with nbval_ being run in normal (not *lax*) mode.
* **Notebook formatting/content**: 
  Check that the User Manual Notebook does not contain output cells (as they could confuse new users).
* **Documentation**: Check that Sphinx_ can build HTML documentation for the package 
  (more info in `Building and Serving Documentation`_ section).

..
   Further test notebooks in the ``TestNotebooks/MiscTests/`` directory.

.. _test_local:

Local testing
^^^^^^^^^^^^^

To locally run the MuMoT test suite in an isolated Python environment 
(containing just the necessary dependencies):

#. Install the tox_ testing automation tool.
#. Run: 

   .. code:: sh

      cd path/to/clone/of/MuMoT/repository
      tox

   This parses the ``tox.ini`` file then
    
    #. Creates a new virtualenv_ (Python virtual environment) containing just 

       * MuMoT's dependencies  (see ``install_requires`` in ``setup.py``)
       * the packages needed for testing and building the documentation (see ``extras_require`` in ``setup.py``)

       This environment is hidden in a ``.tox`` directory to discourage developers from manually tweaking it.
    #. Runs the :ref:`test suite described above<test_suite>`.
       If nbval_ encounters any failures/errors then 
       a Jupyter tab is opened in the default web browser showing 
       the location of the failure/error.

Note: attempts to measure code coverage using a Notebook will fail if 
you call the ``parseModel`` function in a Notebook by passing it a reference to 
an input cell that uses the ``%%model`` cell magic; you need to instead 
call ``parseModel`` by passing it a model defined as a simple string
(e.g. as is done in ``TestNotebooks/MuMoTtest.ipynb``).

.. _test_ci:

Automated testing using Travis CI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each `Pull Request`_ against the `MuMoT GitHub repository`_ and 
each push to the ``master`` branch in that repository 
trigger a `Continuous Integration <https://docs.travis-ci.com/user/for-beginners>`__ (CI) job
on the `travis-ci.org <https://docs.travis-ci.com/user/for-beginners>`__ platform 
(a service that is free for open-source projects).

Each job 
runs a set of user-defined tasks in an isolated execution  environment, 
logs output from those tasks, 
quits early if an error is encountered
and reports the exit status on completion of the job.

Benefits:

* Tests are run automatically without needing to be manually triggered and the results inspected by developers;
* If commits are typically made to :ref:`feature branches <cont_wflow>` then you will be notified that tests fail 
  *before* you merge any changes into the ``master`` branch.
* You can concentrate on other things whilst the CI service is running tests on your behalf.

The **Travis CI configuration** is in the file ``.travis.yml``.  
This does little more than :ref:`call tox <test_local>`.

The Travis CI **dashboard** for the project shows **job exit statuses** and **logs**:
`https://travis-ci.com/DiODeProject/MuMoT/ <https://travis-ci.com/DiODeProject/MuMoT/>`__.

.. _build_docs:

Building and serving documentation
----------------------------------

This MuMoT documentation is built using the Sphinx_ tool using/from:

* The ``docs/source/conf.py`` Sphinx config file;
* A number of anthropogenic pages written in reStructuredText_ format (see ``docs/source/*.rst``);
* A number of pages of API documentation that were autogenerated from module/class/method/function docstrings in the MuMoT source code.
  (These docstrings need to be written in the numpydoc_ format and are extracted/processed by the autodoc_ and autosummary_ Sphinx extensions).

The Sphinx documentation is / can be built under several different circumstances:

* Manually in a development environment;
* Automatically whenever :ref:`tox is run <test_local>`;
* Automatically whenever :ref:`a CI job is run <test_ci>`;
* Automatically following a push to the master branch of the MuMoT repository, 
  which causes the `ReadTheDocs <rtd>`_ service (free for open-source projects) to not only build HTML documentation
  but also start serving a new version of the Sphinx site.

Building the docs locally 
^^^^^^^^^^^^^^^^^^^^^^^^^

#. Ensure the optional ``docs`` dependencies of ``mumot`` have been installed within your local development environment (a conda environment or virtualenv; see also the :ref:`MumoT install guide <install>`:

   .. code::

      python -m pip install path/to/clone/of/MuMoT/repository[docs]
#. Move into the ``docs`` subdirectory within your MuMoT git repository:

   .. code::

      cd path/to/clone/of/MuMoT/repository
      cd docs

#. Use Sphinx to build HTML documentation:

   .. code::

      make html

   This writes output to the ``_build/html`` directory, which is ignored by git.

#. (Optional) view the generated documentation:

   .. code::

      firefox _build/html/index.html

ReadTheDocs
^^^^^^^^^^^

By default `ReadTheDocs <rtd>`_ builds and serves HTML documentation from/for the tip of the ``master`` branch but this is configurable.

The project's `ReadTheDocs dashboard <rtd_dash>`_ includes logs for each build (occasionally useful for trouble-shooting).

Running the User Manual Notebook on mybinder.org
------------------------------------------------

The User Manual Notebook can be run online without the need for any local installation and configuration. 

This is facilitated by mybinder.org_, a public instance of the BinderHub_ service.  
BinderHub is allows many users to start *Binder* sessions: 
within a session, BinderHub creates a per-session software environment on demand on remote hardware (using repo2docker_) then 
starts a Jupyter service within that environment.  

As an end user, all you need to start a BinderHub session is 

* The URL of an accessible Git repository that contains a software environment definition 
  (e.g. a Python ``requirements.txt`` file, conda ``environment.yml`` or a Docker ``Dockerfile``);
* The branch, tag or commit that you'd like to access within that repository;
* (Optional) a relative path within that directory to a Notebook you'd like to run.

These parameters can be supplied via a web form or as URL parameters (allowing someone to just follow a link to start a Binder session).

Configuration
^^^^^^^^^^^^^

Behind the scenes mybinder.org uses repo2docker to 
build an Ubuntu Docker image for running the MuMoT User Manual Notebook in, 
and pushes this to its Docker image registry.  The build process has three steps:

#. Install several Ubuntu packages (inc. GraphViz and a LaTeX distribution); see the ``apt.txt`` file in this repo;
#. Create a conda environment using the ``environment.yml`` file in this repo 
   (*this should idealy be a non-conda Python environment (defined using a ``requirements.txt`` file) for consistency with the testing framework*);
#. Perform some post-install steps (install the TOC2 (table of contents) Jupyter extension and generate the MatPlotLib font cache); see the ``postBuild`` file in this repo;

After an image has been created and pushed to the image registry it remains cached there until:

* a timeout is reached or;
* a user requests an image for a commit for which an image has not yet been cached 
  (e.g. if the user wants to work with the tip of master and 
  new commits have recently been pushed to that repository.

The repo2docker build process takes ~15 mins for MuMoT; 
therefore note that any pushes to the master branch will invalidate any cached image for the tip of the master branch, 
which will increase mybinder.org startup times from seconds to ~15 mins.

**Button**: A mybinder.org session for the User Manual as of the tip of the master branch can be started by 
following the link in the instructions for :ref:`getting started online <mybinder_usage>`.

Creating a new release
----------------------

.. todo:: Add info.



.. _MuMoT GitHub repository: https://github.com/DiODeProject/MuMoT
.. _Pull Request: https://help.github.com/articles/about-pull-requests/
.. _Sphinx: http://www.sphinx-doc.org/
.. _autodoc: http://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
.. _autosummary: http://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html
.. _nbdime: https://nbdime.readthedocs.io/
.. _nbval: https://github.com/computationalmodelling/nbval
.. _numpydoc: http://numpydoc.readthedocs.io/en/latest/format.html
.. _pytest-cov: https://pytest-cov.readthedocs.io/
.. _pytest: https://docs.pytest.org/en/latest/
.. _reStructuredText: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _rtd: https://readthedocs.org/
.. _rtd_dash: https://readthedocs.org/projects/mumot/
.. _tox: https://tox.readthedocs.io/
.. _virtualenv: https://virtualenv.pypa.io/
.. _mybinder.org: https://mybinder.org/
.. _BinderHub: https://binderhub.readthedocs.io/
.. _repo2docker: https://github.com/jupyter/repo2docker
