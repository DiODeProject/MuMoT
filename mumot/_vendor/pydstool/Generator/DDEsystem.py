# Differential-delay system (incomplete)

from .allimports import *
from .baseclasses import ctsGen
from ..utils import *
from ..common import *

# Other imports
from numpy import (
    inf as Inf, nan as NaN, isfinite, any as sometrue, all as alltrue, array
)
import math, random
from copy import copy, deepcopy

# -----------------------------------------------------------------------------

class DDEsystem(ctsGen):
    """Delay-differential equations.

    (incomplete)"""

    # spec of initial condition _interval_ will be interesting!
    # Use a Variable trajectory over that interval?
    def __init__(self, kw):
        ctsGen.__init__(self, kw)
        raise NotImplementedError


    def validateSpec(self):
        ctsGen.validateSpec(self)


    def __del__(self):
        ctsGen.__del__(self)
