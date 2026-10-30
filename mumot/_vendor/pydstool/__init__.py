"""Vendored, trimmed copy of PyDSTool (https://github.com/robclewley/pydstool).

Only the pure-Python parts of PyDSTool that MuMoT needs for equilibrium-point
continuation (the Vode ODE generator and PyCont) are kept; compiled
integrators (Dopri, Radau), AUTO, ADMC++ and the Toolbox have been removed.
See README.md in this directory for provenance and the list of changes.
"""

__LICENSE__ = """\
Copyright (C) 2007-2012, Copyright (C) 2007-2014, Robert Clewley
All rights reserved.

Parts of this distribution that originate from different authors are
individually marked as such. Copyright and licensing of those parts remains
with the original authors.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    3. The name of Robert Clewley, or of his affiliations (Georgia State
      University, and its representatives) may not be used to endorse or
      promote products derived from this software without specific prior
      written permission.

THIS SOFTWARE IS PROVIDED BY ROBERT CLEWLEY ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL ROBERT CLEWLEY BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from .Events import *
from .Interval import *
from .Points import *
from .Variable import *
from .Trajectory import *
from .FuncSpec import *
from . import Generator as GenModule
from .Generator import Generator as Generator_
from .Generator import *
Generator = GenModule
from . import Model as ModelModule
from .Model import Model as Model_
from .Model import *
Model = ModelModule
from .ModelTools import *
from .ModelContext import *
from .Symbolic import *
from .ModelSpec import *
from .parseUtils import auxfnDBclass, protected_allnames, protected_auxnamesDB, \
         convertPowers
from .PyCont import *
from .common import Continuous, Discrete, targetLangs, args
from .utils import *
