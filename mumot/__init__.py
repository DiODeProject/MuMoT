"""Multiscale Modelling Tool (MuMoT)

For documentation and version information use about()

Authors:
James A. R. Marshall, Andreagiovanni Reina, Thomas Bose

Contributors:
Robert Dennison

Packaging, Documentation and Deployment:
Will Furnass

Windows Compatibility:
Renato Pagliara Vasquez
"""

import sys

from ._version import __version__

# Import the functions and classes we wish to export i.e. the public API
from .models import (
    MuMoTmodel,
    parseModel,
)
from .utils import (
    about,
)
from .views import (
    MuMoTSSAView,
    MuMoTbifurcationView,
    MuMoTfieldView,
    MuMoTintegrateView,
    MuMoTmultiView,
    MuMoTmultiagentView,
    MuMoTnoiseCorrelationsView,
    MuMoTstochasticSimulationView,
    MuMoTstreamView,
    MuMoTtimeEvolutionView,
    MuMoTvectorView,
    MuMoTview,
)
from .controllers import (
    MuMoTbifurcationController,
    MuMoTcontroller,
    MuMoTfieldController,
    MuMoTmultiController,
    MuMoTmultiagentController,
    MuMoTstochasticSimulationController,
    MuMoTtimeEvolutionController,
)
from .consts import (
    NetworkType,
    MAX_RANDOM_SEED,
)
from .exceptions import (
    MuMoTError,
    MuMoTSyntaxError,
    MuMoTValueError,
    MuMoTWarning,
)

from IPython import get_ipython

# The currently-running IPython instance (None outside IPython)
ipython = get_ipython()


def _hide_traceback(exc_tuple=None, filename=None, tb_offset=None,
                    exception_only=False, running_compiled_code=False):
    etype, value, tb = sys.exc_info()
    return ipython._showtraceback(etype, value, ipython.InteractiveTB.get_exception_only(etype, value))


if ipython is not None:
    ipython.run_line_magic('alias_magic', 'model latex')
    # Interactive figures: ipympl's widget backend works in JupyterLab,
    # Notebook 7+ and VS Code; nbagg only works in the classic Notebook
    try:
        import ipympl  # noqa: F401
        ipython.run_line_magic('matplotlib', 'widget')
    except ImportError:
        ipython.run_line_magic('matplotlib', 'nbagg')

    _show_traceback = ipython.showtraceback
    ipython.showtraceback = _hide_traceback


def setVerboseExceptions(verbose: bool = True) -> None:
    """Set the verbosity of exception handling.

    Parameters
    ----------
    verbose : bool, optional
        Whether to show a exception traceback.  Defaults to True.

    """
    if ipython is not None:
        ipython.showtraceback = _show_traceback if verbose else _hide_traceback
