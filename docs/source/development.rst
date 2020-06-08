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
   See the *branch and versioning policy* below for more information about how branches, tags and versions are managed in this project.
#. Make commits to that branch:

   * Style: write code using `standard Python naming conventions <pep8>`_.
   * Testing: if you add new features, fix bug(s) or change existing functionality:

     * Add (lower-level) `unit tests <https://en.wikipedia.org/wiki/Unit_testing>`__ to
       the Python source files in the ``tests/`` directory.
     * Add (higher-level) `acceptance <https://en.wikipedia.org/wiki/Acceptance_testing>`__/`regression <https://en.wikipedia.org/wiki/Regression_testing>`__ tests
       to ``TestNotebooks/MuMoTtest.ipynb`` (or to/as Notebooks in the ``TestNotebooks/MiscTests/`` directory).

   * Documentation: include Python docstrings documentation in the numpydoc_ format for all modules, functions, classes, methods and (if applicable) attributes.
   * Do not commit an updated User Manual Notebook or test notebooks containing output cells; all output cells should be stripped first using:

     .. code:: sh

        jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace docs/MuMoTuserManual.ipynb

     Note that the test suite checks this automatically.

   * Use comments containing ``@todo`` in Python code for reminders.

#. When you are ready to merge your feature branch into the ``master`` branch of the 'upstream' repository:

   #. Run all tests using ``tox`` (see Testing_) first.
   #. Create a `Pull Request`_ to request that
      your feature branch be merged into the master branch of the upstream repository.
      Automated tests will then run against your branch (using `GitHub Actions <gh_actions_intro>`)
      and your code will be reviewed by the core development team,
      after which your changes will be merged into the main MuMoT repository *or*
      you will be asked to make further changes to your branch.

.. _testing:

Setting up a local development environment
------------------------------------------

Follow the :ref:`install instructions <install>` but ensure you run:

.. code:: sh

   python3 -m pip install path/to/clone/of/MuMoT/repository[test,docs]

instead of just ``python3 -m pip install path/to/clone/of/MuMoT/repository``.
The '``[test,docs]``' bit ensures that the optional dependencies required to run tests and build the documentation are installed.

If you make local changes to the MuMoT Python package and want to use the updated package you should:

#. Re-install the package within your conda environment or virtualenv (without upgrading MuMoT's dependencies):

   .. code:: sh

      python3 -m pip install --upgrade --upgrade-strategy only-if-needed path/to/clone/of/MuMoT/repository[test,docs]

#. Restart any running IPython kernels within which you have imported the MuMoT package.

Testing
-------

.. _test_suite:

Test suite
^^^^^^^^^^

Testing of MuMoT is currently very basic;
the test suite only checks that certain Jupyter Notebooks run without failing i.e. there are no checks for correctness of results.
However, there is a framework in place to allow more tests to be written:

* **Unit tests**: run by pointing pytest_ at the ``tests/`` directory; also generates a code coverage data using pytest-cov_; *only a few tests implemented so far*.
* **Basic integration tests**:
  Ensure that certain Jupyter Notebooks can be run without
  raising Python exceptions/errors:

   * ``TestNotebooks/MuMoTtest.ipynb``
   * ``docs/MuMoTuserManual.ipynb``

  Plus several other Notebooks (see ``tox.ini``).

  These tests are performed by running the Notebooks using the nbval_ plug-in for pytest_, with nbval_ being run in *lax* mode.
  Code coverage data is also captured at this stage when running ``TestNotebooks/MuMoTtest.ipynb`` and
  appended to that captured during the unit testing.
* **Regression tests**: *not yet implemented*.
  However could be performed by running the Notebook using the nbval_ plug-in for pytest_,
  with nbval_ being run in normal (not *lax*) mode,
  to ensure that the ``TestNotebooks/MuMoTtest.ipynb`` integration test Notebook
  generates sufficiently similar output cells to those saved in that file
  when re-run in a clean environment;
* **Notebook formatting/content**:
  Check that the User Manual Notebook does not contain output cells (as they could confuse new users).
* **Documentation**: Check that Sphinx_ can build HTML documentation for the package
  (more info in `Building and Serving Documentation`_ section).

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

   This parses the ``tox.ini`` file then:

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

Automated testing using a GitHub Actions workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each `Pull Request`_ against the `MuMoT GitHub repository`_ and
each push to the ``master`` branch in that repository
trigger a `GitHub Actions <gh_actions_intro>` continuous integration and continuous delivery (CI/CD) workflow.

Each invocation of the workflow
runs a set of user-defined tasks in an isolated execution environment,
logs output from those tasks,
quits early if an error is encountered
and reports the exit status on completion of the job.

Benefits:

* Tests are run automatically without needing to be manually triggered and the results inspected by developers;
* If pull requests are made from :ref:`feature branches <cont_wflow>` against the ``master`` branch
  then you will be notified that tests fail *before* you merge any changes into ``master``.
* You can concentrate on other things whilst the CI/CD service is running tests on your behalf.
* Packages can be automatically be pushed to the Python Package Index (PyPI) if certain conditions are met (the CD part of CI/CD).

The **GitHub Actions CI configuration** is in the file ``.github/workflows/test-and-release.yml``.
In short, this:

* :ref:`Calls tox <test_local>` to run tests for all supported Python versions;
* Uploads code coverage;
* Checks that source and binary *distributions* can be built for the package;
* If the workflow was triggered by a tagged push to master then upload those distributions to PyPI.

The GitHub Actions **dashboard** for the project shows **job exit statuses** and **logs**:
`https://github.com/DiODeProject/MuMoT/actions <gh_action_dashboard>`.
From the dashboard you can restart one or more workflow jobs via *Re-run jobs*, which might be useful if
a GH Actions workflow job `times out after 6h <gh_actions_timeouts>`,
a entire GH Actions workflow `times out after 72h <gh_actions_timeouts>`,
a job fails as it has not produced any output for several minutes
or you suspect that job failures are otherwise non-deterministic.

.. _build_docs:

Building and serving documentation
----------------------------------

This MuMoT documentation is built using the Sphinx_ tool using/from:

* The ``docs/source/conf.py`` Sphinx config file;
* A number of anthropogenic pages written in reStructuredText_ format (see ``docs/source/*.rst``);
* A number of pages of API documentation that were auto-generated from module/class/method/function docstrings in the MuMoT source code.
  (These docstrings need to be written in the numpydoc_ format and are extracted/processed by the autodoc_ and autosummary_ Sphinx extensions).

The Sphinx documentation is / can be built under several different circumstances:

* Manually in a development environment;
* Automatically whenever :ref:`tox is run <test_local>`;
* Automatically whenever :ref:`a CI job is run <test_ci>`;
* Automatically following a push to the master branch of the MuMoT repository,
  which causes the `ReadTheDocs <https://readthedocs.org/projects/mumot/>`__ service to
  rebuild and publish the documentation at `https://mumot.readthedocs.io <https://mumot.readthedocs.io/>`__.

Building the docs locally
^^^^^^^^^^^^^^^^^^^^^^^^^

#. Ensure the optional ``docs`` dependencies of ``mumot`` have been installed within your local development environment
   (a conda environment or virtualenv; see also the :ref:`MumoT install guide <install>`:

   .. code::

      python3 -m pip install path/to/clone/of/MuMoT/repository[docs]

#. Move into the ``docs`` subdirectory within your MuMoT git repository:

   .. code::

      cd path/to/clone/of/MuMoT/repository
      cd docs

#. Install Sphinx:

   .. code::

      python3 -m pip install sphinx

#. Use Sphinx to build HTML documentation:

   .. code::

      make html

   This writes output to the ``_build/html`` directory, which is ignored by git.

#. (Optional) view the generated documentation:

   .. code::

      firefox _build/html/index.html

Running the User Manual Notebook on mybinder.org
------------------------------------------------

The User Manual Notebook can be run online without the need for any local installation and configuration.

This is facilitated by mybinder.org_, a public instance of the BinderHub_ service.
BinderHub is allows many users to start *Binder* sessions:
within a session, BinderHub creates a per-session software environment on demand on remote hardware (using repo2docker_) then
starts a Jupyter service within that environment.

As an end user, all you need to start a BinderHub session is

* The URL of an accessible Git repository that contains a software environment definition
  (e.g. a Python ``requirements.txt`` file, Conda ``environment.yml`` or a Docker ``Dockerfile``);
* The branch, tag or commit that you'd like to access within that repository;
* (Optional) a relative path within that directory to a Notebook you'd like to run.

These parameters can be supplied via a web form or as URL parameters (allowing someone to just follow a link to start a Binder session).

Configuration
^^^^^^^^^^^^^

Behind the scenes mybinder.org uses repo2docker to
build an Ubuntu Docker image for running the MuMoT User Manual Notebook in,
and pushes this to its Docker image registry.  The build process has three steps:

#. Install several Ubuntu packages (inc. GraphViz and a LaTeX distribution); see the ``apt.txt`` file in this repo;
#. Create a Python virtualenv containing just the MuMoT Python package and its dependencies;
#. Perform some post-install steps (install the TOC2 (table of contents) Jupyter extension and generate the Matplotlib font cache); see the ``postBuild`` file in this repo;

After an image has been created and pushed to the image registry it remains cached there until:

* a timeout is reached or;
* a user requests an image for a commit for which an image has not yet been cached
  (e.g. if the user wants to work with the tip of master and
  new commits have recently been pushed to that repository.

The repo2docker build process takes ~15 minutes for MuMoT;
therefore note that any pushes to the master branch will invalidate any cached image for the tip of the master branch,
which will increase mybinder.org startup times from seconds to ~15 minutes.

**Button**: A mybinder.org session for the User Manual as of the latest stable release of MuMoT can be started by
following the link in the instructions for :ref:`getting started online <mybinder_usage>`.

Branch, tag and version policy, inc. how to create a new release
----------------------------------------------------------------

The project uses `semantic versioning`_ e.g. compared to version ``0.8.0``:

    - ``0.8.1`` is a *patch* version increase - backwards-compatible bugfixes *only*
    - ``0.9.0`` is *minor* version increase - new functionality added in backwards-compatible manner
    - ``1.0.0`` is a *major* version increase - introduces incompatible API changes

In this project the use of branches and git tags is as follows:

 - The ``master`` branch is the only long-lived *active* branch
 - New features are developed by creating **feature branches** from the ``master`` branch;
   these feature branches are then ultimately merged back into ``master`` via Pull Requests then deleted.
 - Changes in patch, major and minor versions are defined **solely** by
   creating an `annotated tag <https://git-scm.com/book/en/v2/Git-Basics-Tagging>`__
   for a particular commit.
   The name of this tag should be of the form ``v<major>.<minor>.<patch>``
   i.e the version preceded by a ``v``.
   **The version does not need to then be specified anywhere in the code
   (other than in links to mybinder in the Sphinx docs)**:
   whenever an installable release of MuMoT is created
   the `setuptools_scm <https://pypi.org/project/setuptools-scm/>`__ package
   will embed version information using the most recent tag on the current branch
   plus extra information derived from the output of ``git describe``
   if the most recent commit does not have an annotated tag associated with it.

To create a release:

#. Decide on the type of the next release (patch, major or minor),
   which depends on the nature of the changes.

#. (Related) determine the appropriate version number for this pending release.
#. Create a draft item for this release in ORDA_ (The University of Sheffield's Research Data Catalogue and Repository),
   so as to reserve a DOI for it.

#. *Major/minor release only*:
   ensure all GitHub Issues tagged with the pending release (*Milestone*)
   have either been addressed or
   are reassigned to a different Milestone.
   Ensure all pull requests against ``master`` relating to the pending Milestone have been merged and all CI tests pass.

#. If necessary, create a pull request against ``master`` to change the version in links to mybinder.org e.g. in

   .. code-block::

      https://mybinder.org/v2/gh/DiODeProject/MuMoT/VERSION?filepath=docs%2FMuMoTuserManual.ipynb

   ensure ``VERSION`` is ``master`` or
   a particular current or future tagged version, preceded by a ``v`` e.g. ``v0.9.0``.

   Also, check/update citation info (including the DOI and contributors) for this pending release
   in ``docs/source/about.rst``.

   Also, update the file ``CHANGELOG.md`` with changes since the last release.
   You can derive this list of changes from commits made since the last release;
   if the last release was tagged in git with ``v0.8.0``
   then you can see the first line of all commit comments since then with: ::

      $ git checkout master
      $ git log --pretty=oneline --abbrev-commit v0.8.0..HEAD

   then: ::

      $ git commit -a -m "Preparing for release of version 0.9.0"

   where 0.9.0 is the version of the new release.
   Next, create the Pull Request.

#. Merge this Pull Request into ``master`` then create an *annotated tag*: ::

      $ git checkout master
      $ git fetch --prune --all
      $ git merge --ff-only upstream/master
      $ git tag -a v0.9.0 -m "Release 0.9.0"
      $ git push upstream --tags
      $ git push

   Here we assume that you've set up your local git repository with a remote called ``upstream``
   that points at ``github.com/DiODeProject/MuMoT.git`` e.g. ::

      $ git remote -v
      origin	git@github.com:willfurnass/MuMoT.git (fetch)
      origin	git@github.com:willfurnass/MuMoT.git (push)
      upstream	git@github.com:DiODeProject/MuMoT.git (fetch)
      upstream	git@github.com:DiODeProject/MuMoT.git (push)

   NB annotated tags are are often used within git repositories to identify
   the commit corresponding to a particular release.

#. The pushing of a tagged commit to ``github.com:DiODeProject/MuMoT.git`` causes GitHub Actions to:

   #. Run through the standard tasks performed for Pull Requests (see ``.github/workflows/test-and-release.yml``) *then*
   #. Build several *distributions* for this release of MuMoT

      * One or more binary 'wheel' packages e.g. ``mumot-0.9.0-py3-none-any.whl``
      * A source package e.g. ``mumot-0.9.0.tar.gz``

   #. Upload these files to `PyPI <https://pypi.org/account/register/>`__
      using environment variables stored as encrypted credentials in this GitHub repo.

#. You can monitor the progress of building packages for MuMoT and uploading them to PyPI
   using the `GitHub Actions dashboard <gh_actions_dashboard>`__.

#. Attach an archive of the code/docs for this release to the draft item in ORDA.
   Create this archive using: ::

      git archive VERSION | gzip > mumot-VERSION.tar.gz

   For example: ::

      git archive v1.2.2 | gzip > mumot-v1.2.2.tar.gz
   
#. Publish the item in ORDA to ensure:

   * The release being referenceable/citable by DOI_.
   * The release being discoverable via the University's Library Catalogue.


.. _BinderHub: https://binderhub.readthedocs.io/
.. _DOI: https://www.doi.org/
.. _MuMoT GitHub repository: https://github.com/DiODeProject/MuMoT
.. _ORDA: https://www.sheffield.ac.uk/library/rdm/orda
.. _Pull Request: https://help.github.com/articles/about-pull-requests/
.. _Sphinx: http://www.sphinx-doc.org/
.. _annotated tag: https://git-scm.com/book/en/v2/Git-Basics-Tagging
.. _autodoc: http://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
.. _autosummary: http://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html
.. _gh_actions_dashboard: https://github.com/DiODeProject/MuMoT/actions
.. _gh_actions_intro: https://help.github.com/en/actions/getting-started-with-github-actions/about-github-actions
.. _gh_actions_timeouts: https://help.github.com/en/actions/reference/workflow-syntax-for-github-actions
.. _mybinder.org: https://mybinder.org/
.. _nbdime: https://nbdime.readthedocs.io/
.. _nbval: https://github.com/computationalmodelling/nbval
.. _numpydoc: http://numpydoc.readthedocs.io/en/latest/format.html
.. _pep8: https://www.python.org/dev/peps/pep-0008/#naming-conventions
.. _pytest-cov: https://pytest-cov.readthedocs.io/
.. _pytest: https://docs.pytest.org/en/latest/
.. _reStructuredText: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _repo2docker: https://github.com/jupyter/repo2docker
.. _semantic versioning: https://semver.org/
.. _tox: https://tox.readthedocs.io/
.. _twine: https://pypi.org/project/twine/
.. _virtualenv: https://virtualenv.pypa.io/
