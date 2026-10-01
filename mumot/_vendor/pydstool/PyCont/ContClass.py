""" ContClass stores continuation curves for a specified model.

    Drew LaMar, March 2006
"""


from .Continuation import (
    EquilibriumCurve, FoldCurve, HopfCurveOne, HopfCurveTwo,
    FixedPointCurve, LimitCycleCurve, UserDefinedCurve,
    FixedPointFoldCurve, FixedPointFlipCurve, FixedPointNSCurve, \
    FixedPointCuspCurve
)
from .misc import *
from .Plotting import pargs, initializeDisplay

from ..Model import Model, findTrajInitiator
from ..Generator import Generator
from ..ModelTools import embed
from .. import Point, Pointset
from ..common import pickle, Utility, args, filteredDict, isUniqueSeq
from ..utils import remain
from .. import utils
from .. import common
from ..core.context_managers import RedirectStdout
from ..errors import *
from ..matplotlib_import import *

from numpy import dot as matrixmultiply
from numpy import (
    array, float64, complex64, int32, zeros, divide, subtract, inf as Inf, nan
    as NaN, isfinite, r_, c_, sign, mod, subtract, divide, transpose, eye,
    real, imag, all, ndarray
)

import scipy.io as io

#####
_classes = ['ContClass']

_constants = ['curve_list', 'curve_args_list', 'auto_list']

__all__ = _classes + _constants
#####


curve_list = {'EP-C': EquilibriumCurve, 'LP-C': FoldCurve,
              'H-C1': HopfCurveOne, 'H-C2': HopfCurveTwo,
              'FP-C': FixedPointCurve, 'LC-C': LimitCycleCurve,
              'UD-C': UserDefinedCurve, 'FD-C': FixedPointFoldCurve,
              'FL-C': FixedPointFlipCurve, 'NS-C': FixedPointNSCurve,
              'CP-C': FixedPointCuspCurve
              }

curve_args_list = ['verbosity']

auto_list = ['LC-C']



class ContClass(Utility):
    """Stores continuation curves for a specified model."""

    curve_list = curve_list
    curve_args_list = curve_args_list
    auto_list = auto_list

    def __init__(self, model):
        if isinstance(model, Generator):
            self.model = embed(model, make_copy=False)
            self.gensys = list(self.model.registry.values())[0]
        else:
            self.model = model
            mi, swRules, globalConRules, nextModelName, reused, \
                epochStateMaps, notDone = model._findTrajInitiator(None,
                                                                   0, 0, self.model.icdict, None, None)
            self.gensys = mi.model
        self._autoMod = None
        self.curves = {}
        self.plot = pargs()

    def __getitem__(self, name):
        try:
            return self.curves[name]
        except:
            raise KeyError('No curve named ' + str(name))

    def __contains__(self, name):
        return name in self.curves

    def __copy__(self):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)

    def __deepcopy__(self, memo=None, _nil=[]):
        pickledself = pickle.dumps(self)
        return pickle.loads(pickledself)

    def delCurve(self, curvename):
        try:
            del self.curves[curvename]
        except KeyError:
            raise KeyError("Curve %s does not exist" % curvename)

    def newCurve(self, initargs):
        """Create new curve with arguments specified in the dictionary initargs."""
        curvetype = initargs['type'].upper()

        if curvetype not in self.curve_list:
            raise PyDSTool_TypeError(str(curvetype) + ' not an allowable curve type')

        # Check name
        cname = initargs['name']
        if 'force' in initargs:
            if initargs['force'] and cname in self.curves:
                del self.curves[cname]

        if cname in self.curves:
            raise ValueError('Ambiguous name field: ' + cname \
                             + ' already exists (use force=True to override)')

        # Check parameters
        if (curvetype != 'UD-C' and self.model.pars == {}) or \
           (curvetype == 'UD-C' and 'userpars' not in initargs):
            raise ValueError('No parameters defined for this system!')

        # Process initial point
        initargs = initargs.copy()   # ensures no side-effects outside
        if 'initpoint' not in initargs or initargs['initpoint'] is None:
            # Default to initial conditions for model
            if self.model.icdict == {}:
                raise ValueError('No initial point defined for this system!')
            elif 'uservars' in initargs:
                if remain(initargs['uservars'], self.model.icdict.keys()) == []:
                    # uservars just used to select a subset of system's regular state vars
                    initargs['initpoint'] = filteredDict(self.model.icdict,
                                                         initargs['uservars'])
                else:
                    raise ValueError('No initial point defined for this system!')
            else:
                initargs['initpoint'] = self.model.icdict.copy()
            #for p in initargs['freepars']:
            #    initargs['initpoint'][p] = self.model.pars[p]
        else:
            if isinstance(initargs['initpoint'], dict):
                initargs['initpoint'] = initargs['initpoint'].copy()
                #for p in initargs['freepars']:
                #    if p not in initargs['initpoint'].keys():
                #        initargs['initpoint'][p] = self.model.pars[p]
            elif isinstance(initargs['initpoint'], str):
                curvename, pointname = initargs['initpoint'].split(':')
                pointtype = pointname.strip('0123456789')
                if curvename not in self.curves:
                    raise KeyError('No curve of name ' + curvename + ' exists.')
                else:
                    point = self.curves[curvename].getSpecialPoint(pointtype, pointname)
                    if point is None:
                        raise KeyError('No point of name ' + pointname + ' exists.')
                    else:
                        initargs['initpoint'] = point

            # Separate from if-else above since 'str' clause returns type Point
            if isinstance(initargs['initpoint'], Point):
                # Check to see if point contains a cycle.  If it does, assume
                #   we are starting at a cycle and save it in initcycle
                for v in initargs['initpoint'].labels.values():
                    if 'cycle' in v:
                        initargs['initcycle'] = v   # Dictionary w/ cycle, name, and tangent information

                # Save initial point information
                initPoint = {}
                if 'curvename' in locals() and curvename in self.curves:
                    initPoint = self.curves[curvename].parsdict.copy()

                initPoint.update(initargs['initpoint'].copy().todict())
                initargs['initpoint'] = initPoint
                # initargs['initpoint'] = initargs['initpoint'].copy().todict()
                #for p in initargs['freepars']:
                #    if p not in initargs['initpoint'].keys():
                #        initargs['initpoint'][p] = self.model.pars[p]

        # Process cycle
        if 'initcycle' in initargs:
            if isinstance(initargs['initcycle'], ndarray):
                c0 = {}
                c0['data'] = args(V = {'udotps': None, 'rldot': None})
                c0['cycle'] = Pointset({'coordnames': self.gensys.funcspec.vars,
                                        'coordarray': initargs['initcycle'][1:,:].copy(),
                                        'indepvarname': 't',
                                        'indepvararray': initargs['initcycle'][0,:].copy()
                                        })
                initargs['initcycle'] = c0
            elif isinstance(initargs['initcycle'], Pointset):
                c0 = {}
                c0['data'] = args(V = {'udotps': None, 'rldot': None})
                c0['cycle'] = initargs['initcycle']
                initargs['initcycle'] = c0

        # The AUTO-based curve types need a compiled AUTO library, which is
        # not shipped with this vendored copy of PyDSTool
        automod = None
        if curvetype in auto_list:
            raise NotImplementedError(
                "Curve type %s requires AUTO, which is not available in the "
                "copy of PyDSTool bundled with MuMoT" % curvetype)

        self.curves[cname] = self.curve_list[curvetype](self.model, self.gensys, automod, self.plot, initargs)

    # Export curve data to Matlab file format
    def exportMatlab(self, filename=None):

        if not filename:
            filename = self.model.name + '.mat'

        savedict = {}

        print(list(self.__dict__['curves'].keys()))
        # Save data for each curve with different prefix
        for name in self.__dict__['curves'].keys():
            if self[name].curvetype not in ['EP-C', 'LC-C']:
                print("Can't save curve type", self[name].curvetype, "yet. (", name, ")")
                continue
            # Can save equilibrium curves
            else:
                curve = self[name].sol
                N = len(curve)
                # Currently ignoring the extra labels (B, etx)
                if self[name].curvetype == 'EP-C':
                    label = 'EP'
                elif self[name].curvetype == 'LC-C':
                    label = 'LC'


                # Add variables
                for key in curve.coordnames:
                    savedict[name+'_'+key] = array(curve[key])

                # Add stabilities
                if 'stab' in curve[0].labels[label].keys():
                    savedict[name+'_stab'] = [curve[x].labels[label]['stab'] for x in range(N)]
                    temp = zeros(N)
                    for x in range(N):
                        if savedict[name+'_stab'][x] == 'S':
                            temp[x] = -1
                        elif savedict[name+'_stab'][x] == 'U':
                            temp[x] = 1
                    savedict[name+'_stab'] = array(temp)

                # Add domain
                if 'domain' in curve[0].labels[label].keys():
                    savedict[name+'_domain'] = [curve[x].labels[label]['domain'] for x in range(N)]
                    temp = zeros(N)
                    for x in range(N):
                        if savedict[name+'_domain'][x] == 'inside':
                            temp[x] = -1
                        elif savedict[name+'_domain'][x] == 'outside':
                            temp[x] = 1
                    savedict[name+'_domain'] = array(temp)

                # Add data
                if 'data' in curve[0].labels[label].keys():
                    # Add eigenvalues
                    if 'evals' in curve[0].labels[label]['data'].keys():
                        dim = len(curve[0].labels[label]['data']['evals'])
                        evltmp = [[] for x in range(dim)]
                        for x in range(dim):
                            evltmp[x] = array([curve[y].labels[label]['data']['evals'][x] for y in range(N)])
                        savedict[name+'_evals'] = array(evltmp)

                    # Add ds
                    if 'ds' in curve[0].labels[label]['data'].keys():
                        savedict[name+'_ds'] = array([curve[y].labels[label]['data']['ds'] for y in range(N)])

                    # Add eigenvectors
                    if 'evecs' in curve[0].labels[label]['data'].keys():
                        #dim = len(curve.coordnames) - 1
                        dim = len(curve[0].labels[label]['data']['evecs'][0])
                        evectmp = []
                        for x in range(dim):
                            for y in range(dim):
                                evectmp.append(array([curve[z].labels[label]['data']['evecs'][x][y] for z in range(N)]))
                        savedict[name+'_evecs'] = array(evectmp)

                    # Add V
                    if 'V' in curve[0].labels[label]['data'].keys():
                        for key in curve[0].labels[label]['data']['V'].keys():
                            savedict[name+'_V_'+key] = array([curve[x].labels[label]['data']['V'][key] for x in range(N)])

        # Save the dictionary in matlab format
        io.savemat(filename, savedict)


    def exportGeomview(self, coords=None, filename="geom.dat"):
        if coords is not None and len(coords) == 3:
            GeomviewOutput = "(progn (geometry " + self.model.name + " { LIST {: axes_" + self.model.name + "}"
            for cname, curve in self.curves.items():
                GeomviewOutput += " {: " + cname + "}"
            GeomviewOutput += "}))\n\n"

            # Get axes limits
            alim = [[Inf,-Inf],[Inf,-Inf],[Inf,-Inf]]
            for cname, curve in self.curves.items():
                for n in range(len(coords)):
                    alim[n][0] = min(alim[n][0], min(curve.sol[coords[n]].toarray()))
                    alim[n][1] = max(alim[n][1], max(curve.sol[coords[n]].toarray()))

            GeomviewOutput += "(progn (hdefine geometry axes_" + self.model.name + " { appearance { linewidth 2 } SKEL 4 3 " + \
                "0 0 0 1 0 0 0 1 0 0 0 1 " + \
                "2 0 1 1 0 0 1 2 0 2 0 1 0 1 2 0 3 0 0 1 1})\n\n"

            for cname, curve in self.curves.items():
                GeomviewOutput += "(hdefine geometry " + cname + " { LIST {: curve_" + cname + "} {: specpts_" + cname + "}})\n\n"

                GeomviewOutput += "(hdefine geometry curve_" + cname + " { appearance { linewidth 2 } SKEL " + \
                    repr(len(curve.sol)) + " " + repr(len(curve.sol)-1)
                for n in range(len(curve.sol)):
                    GeomviewOutput += " " + repr((curve.sol[n][coords[0]]-alim[0][0])/(alim[0][1]-alim[0][0])) + \
                        " " + repr((curve.sol[n][coords[1]]-alim[1][0])/(alim[1][1]-alim[1][0])) + \
                        " " + repr((curve.sol[n][coords[2]]-alim[2][0])/(alim[2][1]-alim[2][0]))
                for n in range(len(curve.sol)-1):
                    GeomviewOutput += " 2 " + repr(n) + " " + repr(n+1) + " 0 0 0 1"

                GeomviewOutput += "})\n\n"

            GeomviewOutput += ")\n"

            f = open(filename, "w")
            f.write(GeomviewOutput)
            f.close()
        else:
            raise Warning("Coordinates not specified or not of correct dimension.")

    def display(self, coords=None, curves=None, figure=None, axes=None, stability=False, domain=False, **plot_args):
        """Plot all curves in coordinates specified by coords.

           Inputs:

               coords -- pair of coordinates (None defaults to the first free
                   parameter and the first state variable).
                   Use a 3-tuple to export to geomview.
        """
        if coords is not None and len(coords) == 3:
            self.exportGeomview(coords)
            return

        if curves is None:
            curves = self.curves.keys()

        plot_curves = []
        for curve in curves:
            if curve in self.curves:
                plot_curves.append(curve)
            else:
                print("Warning: Curve " + curve + " does not exist.")

        if len(plot_curves) > 0:
            initializeDisplay(self.plot, figure=figure, axes=axes)

        for curve in plot_curves:
            self.curves[curve].display(coords, figure=figure, axes=axes, stability=stability, domain=domain, init_display=False, **plot_args)

    def computeEigen(self):
        for curve in self.curves.values():
            curve.computeEigen()

    def info(self):
        print(self.__repr__())
        #print "  Variables : %s"%', '.join(self.model.allvars)
        #print "  Parameters: %s\n"%', '.join(self.model.pars.keys())
        print("Containing curves: ")
        for c in self.curves:
            print("  " + c + " (type " + self.curves[c].curvetype + ")")

    def update(self, args):
        """Update parameters for all curves."""
        for c in args.keys():
            if c not in curve_args_list:
                args.pop(c)

        for v in self.curves.values():
            v.update(args)

    def __repr__(self):
        return 'ContClass of model %s'%self.model.name

    __str__ = __repr__
