# CHANGELOG

## v1.3.0

Modernisation for current Python and scientific Python versions:

 - Supports Python 3.10-3.13 (dropped 3.6-3.9) and current releases of NumPy (2.x), SciPy, SymPy,
   Matplotlib, IPython, ipywidgets and Jupyter (Notebook 7, JupyterLab).
 - Dependencies are no longer pinned, except `antlr4-python3-runtime==4.11.*`,
   which SymPy's LaTeX parser requires (see #418). Each dependency has a tested minimum version.
 - PyDSTool is no longer a dependency: a trimmed copy, updated for current Python/NumPy/SciPy,
   is bundled as `mumot._vendor.pydstool` (see its `README.md` for provenance and changes).
 - Bifurcation diagrams use the new `mumot.continuation` module, a small backend-independent
   continuation API; PyDSTool is now only used behind it.
 - Fixed branch switching at branch points, which (in PyDSTool) depended on floating-point rounding
   and could follow the wrong branch.
 - Interactive figures use the `ipympl` (`%matplotlib widget`) backend where available,
   so they work in Notebook 7+, JupyterLab and VS Code.
 - Removed the ineffective `iopub` rate limit tweak (it only changed a setting in the kernel process;
   set `--ServerApp.iopub_msg_rate_limit` when starting Jupyter instead).
 - Restored behaviour that changed with newer SymPy (`latex()` of strings, `simplify()`
   evaluating derivatives, stricter `subs()`), and noise equations now render moments as
   ⟨η⟩ as intended.
 - Removed the circular import between `mumot.utils` and the package (`mumot` can now be imported
   from a source checkout) and moved symbolic derivations from `views` to the new `mumot.equations`.
 - Fixed stream plots with noise ellipses (complex-valued angles) and widget state containing
   infinite values, plus two latent bugs (broken `chmod` calls, an undefined variable in `utils`).
 - New tests: continuation API, the bundled PyDSTool, and symbolic results checked against
   MuMoT 1.2.2 with SymPy 1.4.
 - Packaging moved to `pyproject.toml`; CI (including a minimum-dependency-versions job),
   Read the Docs and Binder configuration updated.

## v1.2.2

Enhancements: 

 - Added more demo notebooks
 - Support for newer versions of certain Python dependencies (ipykernel, notebook, pyzmq and tornado) 
 - `realtimePlot` (aka `runtimePlot`) is also available for `multiController`. 
   However, this is available only when `shareAxes` is `False` or 
   when the `realtimePlot` is the first view to be plotted. 
   Other cases cannot be supported at the moment.
 - Added possibility to use the graphical keywords (`xlab`, `ylab`, `fontsize`, `legend_loc`, `legend_fontsize`, `choose_xrange`, `choose_yrange`) on all commands.
 - Added analysis to Variance suppression notebook

## v1.1.2

Enhancements:

- Various documentation improvements
- Added another demo notebook
- Show id of Figure objects in field views
- Added possibility to have initial state >1 for `integrate()` and `bifurcation()`
  For `multiagent()` and `SSA()` the widgets are still limited to the sum of 1
- Replace `latex2sympy`'s `process_sympy` with `sympy`'s `parse_latex`
- Update docs to state Python >=3.6 required
- Refactor MuMoT into separate modules
  this sets the length of time over which streams are integrated
- `numPoints` added as keyword to 3D stream plot to set number of streams plotted
- 3D stream plot now plots streams from random subset of starting points
- 3D stream plot shading now based on velocity (calculated from line segment length)
- 1D stream added
- 3D stream added

Bug fixes:

- Guard against `iopub` rate limiting warnings
- Increase `nbval` cell exec timeout
- Suppress `matplotlib` deprecation warning in nested multicontrollers
- Sum to 1 for all views; implement warnings correctly
- Patched issue for 1D models
- Patched issue with multiController
- Fixed exceptions for stochastic analysis methods
- Fixed widgets for rates with equation
- Patched SSA bug

## v1.0.0

- First release
