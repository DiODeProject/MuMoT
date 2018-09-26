Development
===========

Reporting issues
----------------

If your issue (feature request or bug) has not already been filed in the MuMoT GitHub repository 
(`list of all open issues <https://github.com/DiODeProject/MuMoT/issues>`__)
then please `file a new Issue <https://help.github.com/articles/creating-an-issue>`__ 
against the `MuMoT GitHub repository`_.

Contributing
------------

If you want to contribute a feature or fix a bug then:

#. `Fork <https://help.github.com/articles/fork-a-repo/>`__ the `MuMoT GitHub repository`_.
#. Create a `feature branch <https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow>`__.
#. Make commits to that branch:
   * Style: write code using `standard Python naming conventions <https://www.python.org/dev/peps/pep-0008/#naming-conventions>`__.
   * Testing: if you add new features, fix bug(s) or change existing functionality:

     * Add (lower-level) `unit tests <https://en.wikipedia.org/wiki/Unit_testing>`__ to 
       the Python source files in the ``tests/`` directory.
     * Add (higher-level) `acceptance <https://en.wikipedia.org/wiki/Acceptance_testing>`__/`regression <https://en.wikipedia.org/wiki/Regression_testing>`__ tests 
       to ``TestNotebooks/MuMoTtest.ipynb`` 
       or test notebooks to ``TestNotebooks/MiscTests/``.

   * Documentation: include Python docstrings documentation in the numpydoc_ format for all modules, functions, classes, methods and (if applicable) attributes.
   * Do not commit an updated user manual Notebook containing output cells; all output cells should be stripped first using: :: 

     .. code:: sh

        jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/MuMoTuserManual.ipynb

   * Use ``@todo`` for reminders.

#. When you are ready to merge your feature branch into the master branch of the 'upstream' repository: 

   #. Run all tests using tox (see Testing_) first.
   #. Create a `Pull Request <https://help.github.com/articles/about-pull-requests/>`__ to request that 
      your feature branch be merged into the master branch of the upstream repository. 
      Automated tests will then run against your branch and your code will be reviewed by the core development team, 
      after which your changes will be merged into the main MuMoT repository *or* 
      you will be asked to make futher changes to your branch.

.. _testing:

Testing
-------

.. todo:: Add info on what tests are run, how tests are run and current limitations of test suite

..
   * KEY DIRS/FILES (TOOLS + DATA)
   * SEVERAL THINGS
      * UNIT TESTS (NEEDED)
      * WILL TWO NOTEBOOKS RUN WITHOUT ERRORS
      * (OPTIONAL) WILL OTHER NOTEBOOKS RUN WITHOUT ERRORS
      * (FUTURE) PROPER NBVAL
      * USER MANUAL CONTAINS NO OUTPUT CELLS
      * SPHINX DOCS CAN BE BUILT
   * MECHANISMS
     * TOX
       * HOW WORKS
       * USEFUL FOR CHECKING GRAPHICAL OUTPUT (MATPLOTLIB RENDERING)
     * CI
     * RTD

   OLD TEXT:

   At present, MuMoT is tested by running several Jupyter notebooks:

   * [docs/MuMoTuserManual.ipynb](docs/MuMoTuserManual.ipynb)
   * [TestNotebooks/MuMoTtest.ipynb](TestNotebooks/MuMoTtest.ipynb)
   <!-- * Further test notebooks in [TestNotebooks/MiscTests/](TestNotebooks/MiscTests) -->

   To locally automate the running of all of the above Notebooks in an isolated Python environment containing just the necessary dependencies:

    1. Install the [tox](https://tox.readthedocs.io/en/latest/) automated testing tool
    2. Run 

       ```sh
       tox
       ```

   This:
    
    1. Creates a new virtualenv (Python virtual environment) containing just 
         * MuMoT's dependencies  (see `install_requires` in [setup.py](setup.py))
         * the packages needed for testing (see `extras_require` in [setup.py](setup.py))
    1. Checks that all of the above Notebooks can be run without any unhandled Exceptions or Errors being generated 
       (using [nbval](https://github.com/computationalmodelling/nbval)).
       If an Exception/Error is encountered then a Jupyter tab is opened in the default web browser showing its location 
       (using [nbdime](https://nbdime.readthedocs.io/en/stable/)).
    1. Checks that the user manual notebook does not contain output cells

   <!--1. Checks that the [docs/MuMoTuserManual.ipynb](docs/MuMoTuserManual.ipynb) and [TestNotebooks/MuMoTtest.ipynb](TestNotebooks/MuMoTtest.ipynb) Notebooks 
       generate the same output cell content as is saved in the Notebook files when re-run 
       (again, using [nbval](https://github.com/computationalmodelling/nbval)).
       If a discrepency is encountered then a Jupyter tab is opened in the default web browser showing details 
       (again, using [nbdime](https://nbdime.readthedocs.io/en/stable/)).-->

Building and serving documentation
----------------------------------

.. todo:: Add info on Sphinx + ReadTheDocs inc. relevant local files, auto-generated API docs

..
   DOCS BUILT USING SPHINX TOOL
      * IN DOCS DIR
      * SOURCE AND OUTPUT DIRS
      * CONF FILE
      * PAGES CREATED BY HAND (RST FORMAT)
      * PLUS AUTO-GEN API DOCS - HOW WORKS?

Creating a new release
----------------------

.. todo:: Add info.

..
   #. UPDATE VERSION NUMBER IN ONE OR TWO PLACES
   #. TAG RELEASE
   #. PUSH TO PYPI

.. _MuMoT GitHub repository: https://github.com/DiODeProject/MuMoT
.. _numpydoc: http://numpydoc.readthedocs.io/en/latest/format.html
