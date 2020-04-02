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

# Set package version.
# NB with Python 3.8 we could use importlib.metadata (in std lib) instead.
from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
import sys

import matplotlib
from matplotlib import pyplot as plt
# Guard against iopub rate limiting warnings (https://github.com/DiODeProject/MuMoT/issues/359)
from notebook.notebookapp import NotebookApp
NotebookApp.iopub_msg_rate_limit = 10000.0
from sympy.parsing.latex import parse_latex

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

try:
    # Try to get the currently-running IPython instance
    ipython = get_ipython()
    ipython.magic('alias_magic model latex')
    ipython.magic('matplotlib nbagg')

    def _hide_traceback(exc_tuple=None, filename=None, tb_offset=None,
                        exception_only=False, running_compiled_code=False):
        etype, value, tb = sys.exc_info()
        return ipython._showtraceback(etype, value, ipython.InteractiveTB.get_exception_only(etype, value))

    _show_traceback = ipython.showtraceback
    ipython.showtraceback = _hide_traceback
except NameError:
    # There is no currently-running IPython instance
    pass


def setVerboseExceptions(verbose: bool = True) -> None:
    """Set the verbosity of exception handling.

    Parameters
    ----------
    verbose : bool, optional
        Whether to show a exception traceback.  Defaults to True.

    """
    ipython.showtraceback = _show_traceback if verbose else _hide_traceback
