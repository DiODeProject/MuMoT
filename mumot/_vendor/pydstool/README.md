# Bundled copy of PyDSTool

This directory contains a trimmed and modernised copy of
[PyDSTool](https://github.com/robclewley/pydstool) **0.91.0** (the last release
on PyPI, February 2020), by Robert Clewley and contributors, distributed under
the BSD-style licence in [`LICENSE`](LICENSE).

MuMoT uses it only for equilibrium-point continuation (PyCont's `EP-C` curves
on top of the pure-Python `Vode_ODEsystem` generator) when drawing bifurcation
diagrams, and only through [`mumot/continuation.py`](../../continuation.py).
Nothing else in MuMoT should import from here.

PyDSTool is no longer maintained and does not work with Python >= 3.10 or
NumPy 2, hence this copy.

## Changes from upstream 0.91.0

Removed (all needed C/Fortran compilers at run time and/or `distutils`, which
was removed in Python 3.12, and none are used by MuMoT):

* `PyCont/auto/` and the AUTO interface in `PyCont/ContClass.py` (asking for
  an AUTO-based curve type such as `LC-C` now raises `NotImplementedError`)
* `integrator/`, `Generator/Dopri_ODEsystem.py`, `Generator/Radau_ODEsystem.py`,
  `Generator/ADMC_ODEsystem.py`, `Generator/mixins.py`
* `Toolbox/` (parameter estimation, neural modelling, etc.)
* `conf.py`, `PyCont/Makefile`, `PyCont/biblio_for_examples.txt`
* the session-management helpers and `who()` from `__init__.py`, which
  is now a slim module that does not star-import NumPy/SciPy/Matplotlib
* compiler helpers (`distutil_destination`, `architecture`, `extra_arch_arg`,
  `get_lib_extension`) from `utils.py`

Modernised:

* absolute `PyDSTool.*` self-imports turned into relative imports, so the
  package works from any location
* NumPy 2: names removed from NumPy (`Inf`, `NaN`, `sometrue`, `alltrue`,
  `product`, `float_`, `complex_`, `int0`, `mat`, `unique1d` and the
  `float`/`int`/`complex`/`bool`/`object` aliases of builtins) are imported
  under their old names from their current NumPy equivalents, e.g.
  `from numpy import inf as Inf`, so the rest of the code is unchanged
* SciPy: `scipy.optimize.minpack`/`scipy.optimize.zeros` replaced by
  `scipy.optimize`; `scipy.polyfit` by `numpy.polyfit`; `sign`/`mod` (which
  SciPy used to re-export from NumPy) are looked up in `numpy`; special
  functions that `scipy.special` no longer provides are skipped
* Matplotlib: removed the fallback to the long-gone `matplotlib.matlab`
* `parser` module (removed in Python 3.10): `parseUtils` now gets its
  concrete syntax trees from the new [`_cst.py`](_cst.py), a small
  recursive-descent parser for Python expressions that produces the same
  trees as `parser.expr(...).tolist()` did under Python 3.8 (checked
  node-for-node against the real `parser` module on 180,000 random
  expressions)
* Python 3.13 (PEP 667): code that ran `exec(code)` and then read the
  names it defined from `locals()` now passes an explicit namespace
  (`exec(code, globals(), ns)`) and reads from that, since `locals()`
  returns a fresh snapshot on each call in Python 3.13
* `x is 'literal'` comparisons (a `SyntaxWarning` since Python 3.8) replaced
  by `==`

Numerical fixes in PyCont:

* `BifPoint.BranchPoint.process`: the direction of the new branch at a branch
  point was computed from a tangent obtained by solving a linear system that
  is singular *at* the branch point, so which branch was found depended on
  floating-point rounding (and differed between NumPy/SciPy/LAPACK versions).
  It now uses the continuation tangent and the SVD null space of the Jacobian.
* `Continuation._MoorePenrose`: falls back to the least-squares solution if
  the bordered Jacobian is exactly singular (e.g. when starting a curve at a
  branch point) instead of raising `LinAlgError`.
