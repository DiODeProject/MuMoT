# CHANGELOG

## v1.2.0

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
