"""MuMoT view classes"""
import bisect
import copy
import datetime
import math
import sys
from typing import Dict, Optional, Tuple, Union

from IPython.display import display, Math
from IPython.utils import io
from ipywidgets import HTML
import ipywidgets.widgets as widgets
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import networkx as nx
import PyDSTool as dst
from scipy.integrate import odeint
import sympy
from sympy import (
    default_sort_key,
    Derivative,
    factorial,
    Function,
    lambdify,
    latex,
    simplify,
    Symbol,
    symbols,
)
from sympy.parsing.latex import parse_latex
from warnings import warn, catch_warnings, simplefilter

from . import (
    consts,
    exceptions,
    utils,
)


figureCounter = 1  # global figure counter for model views


class MuMoTview:
    """A view on a model."""

    # Model view is on
    _mumotModel = None
    # Figure/axis object to plot view to
    _figure = None
    # Unique figure number
    _figureNum = None
    # 3d axes? (False => 2d)
    _axes3d = None
    # Controller that controls this view @todo - could become None
    _controller = None
    # Summary logs of view behaviour
    _logs = None
    # parameter values when used without controller
    _fixedParams = None
    # dictionary of rates and value
    _ratesDict = None
    # total number of agents in the simulation
    _systemSize = None
    # silent flag (TRUE = do not try to acquire figure handle from pyplot)
    _silent = None
    # plot limits (for non-constant system size) @todo: not used?
    _plotLimits = None
    # command name that generates this view
    _generatingCommand = None
    # generating keyword arguments
    _generatingKwargs = None
    # x-label
    _xlab = None
    # y-label
    _ylab = None
    # defines fontsize on the axes
    _axes_font_size = None
    # legend location: combinations like 'upper left', lower right, or 'center center' are allowed (9 options in total)
    _legend_loc = None
    # legend fontsize, accepts integers
    _legend_fontsize = None
    # displayed range for vertical axis
    _chooseXrange = None
    # displayed range for horizontal axis
    _chooseYrange = None

    def __init__(self, model, controller, figure=None, params=None, **kwargs):
        self._silent = kwargs.get('silent', False)
        self._mumotModel = model
        self._controller = controller
        self._logs = []
        self._axes3d = False
        self._fixedParams = {}
        self._plotLimits = 1

        self._generatingKwargs = kwargs
        if params is not None:
            (paramNames, paramValues) = utils._process_params(params)
            self._fixedParams = dict(zip(paramNames, paramValues))

        # storing the rates for each rule
        if self._mumotModel:
            freeParamDict = self._get_argDict()
            self._ratesDict = {}
            for rule in self._mumotModel._rules:
                self._ratesDict[str(rule.rate)] = rule.rate.subs(freeParamDict)
                if self._ratesDict[str(rule.rate)] == float('inf') or self._ratesDict[str(rule.rate)] is sympy.zoo:
                    self._ratesDict[str(rule.rate)] = sys.maxsize
            # print(self._ratesDict)
        self._systemSize = self._getSystemSize()

        # storing the graphics params
        if 'fontsize' in kwargs:
            self._axes_font_size = kwargs['fontsize']
        else:
            self._axes_font_size = None
        self._legend_loc = kwargs.get('legend_loc', 'upper right')
        self._legend_fontsize = kwargs.get('legend_fontsize', None)
        self._chooseXrange = kwargs.get('choose_xrange', None)
        self._chooseYrange = kwargs.get('choose_yrange', None)

        if not self._silent:
            _buildFig(self, figure)

    def _resetErrorMessage(self) -> None:
        if self._controller is not None:
            if not self._silent:
                self._controller._errorMessage.value = ''

    def _showErrorMessage(self, message) -> None:
        if self._controller is not None:
            self._controller._errorMessage.value = self._controller._errorMessage.value + message
        else:
            print(message)

    def _show_computation_start(self) -> None:
        if self._controller is not None:
            self._controller._bookmarkWidget.style.button_color = 'pink'

    def _show_computation_stop(self) -> None:
        # ax = plt.gca()
        # ax.set_facecolor('xkcd:white')
        # print("pink off")
        if self._controller is not None:
            self._controller._bookmarkWidget.style.button_color = 'silver'

    def _setLog(self, log) -> None:
        self._logs = log

    def _log(self, analysis) -> None:
        print(f"Starting {analysis} with parameters ", end='')
        param_names = []
        param_values = []
        if self._controller is not None:
            # @todo: if the alphabetic order is not good,
            # the view could store the desired order in (param_names)
            # when the controller is constructed
            for name in sorted(self._controller._widgetsFreeParams.keys()):
                param_names.append(name)
                param_values.append(self._controller._widgetsFreeParams[name].value)
            for name in sorted(self._controller._widgetsExtraParams.keys()):
                param_names.append(name)
                param_values.append(self._controller._widgetsExtraParams[name].value)
        # if self._param_names is not None:
        #     param_names += map(str, self._param_names)
        #     param_values += self._paramValues
        #   # @todo: in soloView, this does not show the extra parameters
        #   # (we should make clearer what the use of showLogs)
        for key, value in self._fixedParams.items():
            param_names.append(str(key))
            param_values.append(value)

        for i in zip(param_names, param_values):
            print('(' + i[0] + '=' + repr(i[1]) + '), ', end='')
        print("at", datetime.datetime.now())

    def _print_standalone_view_cmd(self, includeParams: bool = True) -> str:
        log_str = self._build_bookmark(includeParams)
        if not self._silent and log_str is not None:
            with io.capture_output() as log:
                print(log_str)
            self._logs.append(log)
        return log_str

    def _set_fixedParams(self, paramDict) -> None:
        self._fixedParams = paramDict

    def _get_params(self, refModel=None):
        if refModel is not None:
            model = refModel
        else:
            model = self._mumotModel

        params = []

        paramInitCheck = []
        for reactant in model._reactants:
            if reactant not in model._constantReactants:
                paramInitCheck.append(latex(sympy.Symbol(f"Phi^0_{reactant}")))

        if self._controller:
            for name, value in self._controller._widgetsFreeParams.items():
                if name in model._ratesLaTeX:
                    name = model._ratesLaTeX[name]
                name = name.replace('(', '')
                name = name.replace(')', '')
                if name not in paramInitCheck:
                    params.append((latex(name), value.value))
        for name, value in self._fixedParams.items():
            # if name == 'systemSize' or name == 'plotLimits':
            #     continue
            if name not in self._mumotModel._get_rates_from_stoichiometry() and name not in self._mumotModel._constantReactants:
                continue
            name = repr(name)
            if name in model._ratesLaTeX:
                name = model._ratesLaTeX[name]
            name = name.replace('(', '')
            name = name.replace(')', '')
            params.append((latex(name), value))
        params.append(('plotLimits', self._getPlotLimits()))
        params.append(('systemSize', self._getSystemSize()))

        return params

    def _get_bookmarks_params(self, refModel=None) -> str:
        params = self._get_params(refModel)
        log_str = "params = ["
        for name, value in params:
            log_str += f"('{name}', {value}), "
        log_str = log_str[:-2]  # throw away last ", "
        log_str += "]"
        return log_str

    def _build_bookmark(self, _=None) -> None:
        self._resetErrorMessage()
        self._showErrorMessage(f"Bookmark functionality not implemented for class {self._generatingCommand}")

    def _getPlotLimits(self, defaultLimits: int = 1) -> int:
        # if self._paramNames is not None and 'plotLimits' in self._paramNames:
        if self._fixedParams is not None and 'plotLimits' in self._fixedParams:
            # systemSize = self._paramValues[self._paramNames.index('plotLimits')]
            plotLimits = self._fixedParams['plotLimits']
        elif self._controller is not None and self._controller._plotLimitsWidget is not None:
            plotLimits = self._controller._plotLimitsWidget.value
        else:
            plotLimits = defaultLimits

        return plotLimits

    def _getSystemSize(self, defaultSize: int = 1) -> int:
        # if self._paramNames is not None and 'systemSize' in self._paramNames:
        if self._fixedParams is not None and 'systemSize' in self._fixedParams:
            # systemSize = self._paramValues[self._paramNames.index('systemSize')]
            systemSize = self._fixedParams['systemSize']
        elif self._controller is not None and self._controller._systemSizeWidget is not None:
            systemSize = self._controller._systemSizeWidget.value
        else:
            systemSize = defaultSize

        return systemSize

    def _get_argDict(self):
        """Get names and values from widgets."""
        paramNames = []
        paramValues = []
        if self._controller is not None:
            for name, value in self._controller._widgetsFreeParams.items():
                # print("wdg-name: " + str(name) + " wdg-val: " + str(value.value))
                paramNames.append(name)
                paramValues.append(value.value)

            # if self._controller._widgetsExtraParams and 'initBifParam' in self._controller._widgetsExtraParams:
            #     paramNames.append(self._bifurcationParameter_for_get_argDict)
            #     paramValues.append(self._controller._widgetsExtraParams['initBifParam'].value)

        # if self._fixedParams and 'initBifParam' in self._fixedParams:
        #     paramNames.append(self._bifurcationParameter_for_get_argDict)
        #     paramValues.append(self._fixedParams['initBifParam'])

        if self._fixedParams is not None:
            for key, item in self._fixedParams.items():
                # print(f"fix-name: {key} fix-val: {item}")
                paramNames.append(str(key))
                paramValues.append(item)

        argNamesSymb = list(map(sympy.Symbol, paramNames))
        argDict = dict(zip(argNamesSymb, paramValues))

        if self._mumotModel._systemSize:
            argDict[self._mumotModel._systemSize] = 1

        # @todo: is this necessary? for which view?
        systemSize = sympy.Symbol('systemSize')
        argDict[systemSize] = self._getSystemSize()

        return argDict

    def _get_fixedPoints1d(self):
        """Calculate stationary states of 1D system."""
        argDict = self._get_argDict()

        EQ1 = self._mumotModel._equations[self._stateVariable1].subs(argDict)

        eps = 1e-8
        EQsol = sympy.solve((EQ1), (self._stateVariable1), dict=True)

        realEQsol = [{self._stateVariable1: sympy.re(EQsol[kk][self._stateVariable1])}
                     for kk in range(len(EQsol))
                     if sympy.Abs(sympy.im(EQsol[kk][self._stateVariable1])) <= eps]

        MAT = sympy.Matrix([EQ1])
        JAC = MAT.jacobian([self._stateVariable1])

        eigList = []
        for nn in range(len(realEQsol)):
            evSet = {}
            JACsub = JAC.subs([(self._stateVariable1, realEQsol[nn][self._stateVariable1])])
            # evSet = JACsub.eigenvals()
            eigVects = JACsub.eigenvects()
            for kk in range(len(eigVects)):
                evSet[eigVects[kk][0]] = (eigVects[kk][1], eigVects[kk][2])
            eigList.append(evSet)
        return realEQsol, eigList  # returns two lists of dictionaries

    def _get_fixedPoints2d(self):
        """Calculate stationary states of 2d system."""

        argDict = self._get_argDict()

        EQ1 = self._mumotModel._equations[self._stateVariable1].subs(argDict)
        EQ2 = self._mumotModel._equations[self._stateVariable2].subs(argDict)

        eps = 1e-8
        EQsolA = sympy.solve((EQ1, EQ2), (self._stateVariable1, self._stateVariable2), dict=True)

        EQsol = []
        addIndexToSolList = []
        for nn in range(len(EQsolA)):
            if len(EQsolA[nn]) == 2:
                addIndexToSolList.append(nn)
            # else:
            #     self._showErrorMessage('Some solutions for Fixed Points may not be unique.')

        for el in addIndexToSolList:
            EQsol.append(EQsolA[el])

        if len(EQsol) == 0:
            self._showErrorMessage('Could not compute any unique solutions for Fixed Points. ')
            return None, None
        elif len(EQsol) < len(EQsolA):
            self._showErrorMessage('Some solutions for Fixed Points may not be unique. ')

        # for nn in range(len(EQsol)):
        #     if len(EQsol[nn]) != 2:
        #         self._showErrorMessage('Some or all solutions are NOT unique.')
        #         return None, None

        realEQsol = [{self._stateVariable1: sympy.re(EQsol[kk][self._stateVariable1]),
                      self._stateVariable2: sympy.re(EQsol[kk][self._stateVariable2])}
                     for kk in range(len(EQsol))
                     if (sympy.Abs(sympy.im(EQsol[kk][self._stateVariable1])) <= eps and
                         sympy.Abs(sympy.im(EQsol[kk][self._stateVariable2])) <= eps)]

        MAT = sympy.Matrix([EQ1, EQ2])
        JAC = MAT.jacobian([self._stateVariable1, self._stateVariable2])

        eigList = []
        for nn in range(len(realEQsol)):
            evSet = {}
            JACsub = JAC.subs([(self._stateVariable1, realEQsol[nn][self._stateVariable1]),
                               (self._stateVariable2, realEQsol[nn][self._stateVariable2])])
            # evSet = JACsub.eigenvals()
            eigVects = JACsub.eigenvects()
            for kk in range(len(eigVects)):
                evSet[eigVects[kk][0]] = (eigVects[kk][1], eigVects[kk][2])
            eigList.append(evSet)
        return realEQsol, eigList  # returns two lists of dictionaries

    def _get_fixedPoints3d(self):
        """Calculate stationary states of 3d system."""
        argDict = self._get_argDict()

        EQ1 = self._mumotModel._equations[self._stateVariable1].subs(argDict)
        EQ2 = self._mumotModel._equations[self._stateVariable2].subs(argDict)
        EQ3 = self._mumotModel._equations[self._stateVariable3].subs(argDict)

        eps = 1e-8
        EQsolA = sympy.solve((EQ1, EQ2, EQ3),
                             (self._stateVariable1, self._stateVariable2, self._stateVariable3),
                             dict=True)
        EQsol = []
        addIndexToSolList = []
        for nn in range(len(EQsolA)):
            if len(EQsolA[nn]) == 3:
                addIndexToSolList.append(nn)
            # else:
            #     self._showErrorMessage('Some solutions for Fixed Points may not be unique.')

        for el in addIndexToSolList:
            EQsol.append(EQsolA[el])

        if len(EQsol) == 0:
            self._showErrorMessage('Could not compute any unique solutions for Fixed Points. ')
            return None, None
        elif len(EQsol) < len(EQsolA):
            self._showErrorMessage('Some solutions for Fixed Points may not be unique. ')

        # for nn in range(len(EQsol)):
        #     if len(EQsol[nn]) != 3:
        #         self._showErrorMessage('Some or all solutions are NOT unique.')
        #         return None, None

        realEQsol = [{self._stateVariable1: sympy.re(EQsol[kk][self._stateVariable1]),
                      self._stateVariable2: sympy.re(EQsol[kk][self._stateVariable2]),
                      self._stateVariable3: sympy.re(EQsol[kk][self._stateVariable3])}
                     for kk in range(len(EQsol))
                     if (sympy.Abs(sympy.im(EQsol[kk][self._stateVariable1])) <= eps and
                         sympy.Abs(sympy.im(EQsol[kk][self._stateVariable2])) <= eps and
                         sympy.Abs(sympy.im(EQsol[kk][self._stateVariable3])) <= eps)]

        MAT = sympy.Matrix([EQ1, EQ2, EQ3])
        JAC = MAT.jacobian([self._stateVariable1, self._stateVariable2, self._stateVariable3])

        eigList = []
        for nn in range(len(realEQsol)):
            evSet = {}
            JACsub = JAC.subs([(self._stateVariable1, realEQsol[nn][self._stateVariable1]),
                               (self._stateVariable2, realEQsol[nn][self._stateVariable2]),
                               (self._stateVariable3, realEQsol[nn][self._stateVariable3])])
            # evSet = JACsub.eigenvals()
            eigVects = JACsub.eigenvects()
            for kk in range(len(eigVects)):
                evSet[eigVects[kk][0]] = (eigVects[kk][1], eigVects[kk][2])
            eigList.append(evSet)

        return realEQsol, eigList  # returns two lists of dictionaries

    def _update_params(self) -> None:
        """Update parameters from widgets.

        If the view requires view-specific params they can be updated implementing _update_view_specific_params().

        """
        freeParamDict = self._get_argDict()
        if self._controller is not None:
            # getting the rates' value
            self._ratesDict = {}
            for rule in self._mumotModel._rules:
                self._ratesDict[str(rule.rate)] = rule.rate.subs(freeParamDict)
                if self._ratesDict[str(rule.rate)] == float('inf') or self._ratesDict[str(rule.rate)] is sympy.zoo:
                    self._ratesDict[str(rule.rate)] = sys.maxsize
                    errorMsg = (f"WARNING! Rate with division by zero. \n"
                                f"The rule has a rate with division by zero: \n"
                                f"{rule.lhsReactants} --> {rule.rhsReactants} with rate {rule.rate}.\n"
                                f"The system has run the simulation with the maximum system value: {self._ratesDict[str(rule.rate)]}")
                    self._showErrorMessage(errorMsg)
            # print("_ratesDict=" + str(self._ratesDict))
            self._systemSize = self._getSystemSize()
        self._update_view_specific_params(freeParamDict)

    def _getWidgetParamValue(self, key, dict=None):
        """Check fixedParams then generatingKwargs for a key value otherwise return from dict."""
        if self._fixedParams.get(key) is not None:
            return self._fixedParams[key]
        elif self._generatingKwargs.get(key) is not None:
            return self._generatingKwargs[key]
        elif dict is not None:
            if dict.get(key) is not None:
                return dict[key].value
            else:
                raise exceptions.MuMoTValueError(f"Could not find value for key '{key}'; "
                                                 "if using a multicontroller try moving keyword definition down to creation of constitutent controllers")
        else:
            return None

    def _getInitialState(self, state, freeParamDict):
        """Get initial state from widgets, otherwise original initial state."""
        if state in self._mumotModel._constantReactants:
            return freeParamDict[state]
        elif self._controller._widgetsExtraParams.get(f"init{state}") is not None:
            return self._controller._widgetsExtraParams[f"init{state}"].value
        else:
            return self._initialState[state]

    def _update_view_specific_params(self, freeParamDict=None):
        """Interface method to update view-specific params from widgets.

        @todo JARM: I don't see what purpose this serves - it is mostly ignored and I don't think will function as intended
        """
        if freeParamDict is None:
            freeParamDict = {}

    def _safeSymbol(self, item):
        """Used in _update_view_specific_params"""
        if type(item) is sympy.Symbol:
            return item
        else:
            return sympy.Symbol(item)

    def showLogs(self, tail: bool = False) -> None:
        """Show logs from view.

        Parameters
        ----------
        tail : bool, optional
           Flag to show only tail entries from logs. Defaults to False.

        """
        if tail:
            tailLength = 5
            print(f"Showing last {min(tailLength, len(self._logs))} of {len(self._logs)} log entries:")
            for log in self._logs[-tailLength:]:
                log.show()
        else:
            for log in self._logs:
                log.show()


class MuMoTmultiView(MuMoTview):
    """Multi-view view.

    Tied closely to :class:`MuMoTmultiController`.

    """
    # view list
    _views = None
    # axes are used for subplots ('shareAxes = True')
    _axes = None
    # number of subplots
    _subPlotNum = None
    # subplot rows
    _numRows = None
    # subplot columns
    _numColumns = None
    # use common axes for all plots (False = use subplots)
    _shareAxes = None
    # controllers (for building bookmarks)
    _controllers = None

    def __init__(self, controller, model, views, controllers, subPlotNum, **kwargs):
        super().__init__(model, controller, **kwargs)
        self._generatingCommand = "mumot.MuMoTmultiController"
        self._views = views
        self._controllers = controllers
        self._subPlotNum = subPlotNum
        self._shareAxes = kwargs.get('shareAxes', False)
        for i, view in enumerate(self._views):
            view._figure = self._figure
            view._figureNum = self._figureNum
            view._setLog(self._logs)
            view._controller = controller
            # MuMoTstochasticSimulationView could be run with the option realtimePlot, this would work only if:
            # * this view is the first of the list of views (so that realtimePlot will not erase previous plots)
            # * the multiController has shareAxes==False, in which case the update will happen only on the view subplot 
            if isinstance(view, MuMoTstochasticSimulationView) and self._shareAxes and i>0:
                view._allowRealtimePlotting = False
        if not self._shareAxes:
            self._numColumns = consts.MULTIPLOT_COLUMNS
            self._numRows = math.ceil(self._subPlotNum / self._numColumns)
            plt.gcf().set_size_inches(9, 4.5)

    def _plot(self, _=None) -> None:
        fig = plt.figure(self._figureNum)
        plt.clf()
        self._resetErrorMessage()
        if self._shareAxes:
            for func, subPlotNum, axes3d in self._controller._replotFunctions:
                func()
        else:
            # subplotNum = 1
            for func, subPlotNum, axes3d in self._controller._replotFunctions:
                with catch_warnings():
                    simplefilter("ignore")
                    if axes3d:
                        # self._figure.add_subplot(self._numRows, self._numColumns, subPlotNum, projection = '3d')
                        plt.subplot(self._numRows, self._numColumns, subPlotNum, projection='3d')
                    else:
                        plt.subplot(self._numRows, self._numColumns, subPlotNum)
                func()
            plt.subplots_adjust(left=0.12, bottom=0.25, right=0.98, top=0.9, wspace=0.45, hspace=None)
            # plt.tight_layout()
            # subplotNum += 1

    def _setLog(self, log) -> None:
        for view in self._views:
            view._setLog(log)

    def _print_standalone_view_cmd(self, includeParams: bool = False):
        model = self._views[0]._mumotModel  # @todo this suppose that all models are the same for all views
        with io.capture_output() as log:
            if not self._controller._silent:
                log_str = "bookmark = "
            else:
                log_str = ""
            log_str += self._generatingCommand + "(["
            for controller in self._controllers:
                log_str += controller._view._print_standalone_view_cmd(False) + ", "
            log_str = log_str[:-2]  # throw away last ", "
            log_str += "]"
            if includeParams:
                log_str += ", " + self._get_bookmarks_params(model)
            if len(self._generatingKwargs) > 0:
                log_str += ", "
                for key in self._generatingKwargs:
                    if type(self._generatingKwargs[key]) == str:
                        log_str += key + " = " + "\'" + str(self._generatingKwargs[key]) + "\'" + ", "
                    else:
                        log_str += key + " = " + str(self._generatingKwargs[key]) + ", "

                log_str = log_str[:-2]  # throw away last ", "
            if 'silent' not in self._generatingKwargs:
                log_str += ", silent = " + str(self._silent)
            if 'bookmark' not in self._generatingKwargs:
                log_str += ", bookmark = False"
            log_str += ")"
            # log_str = log_str.replace('\\', '\\\\')  # @todo is this necessary?

            if not self._silent and log_str is not None:
                print(log_str)
                self._logs.append(log)
            return log_str

    def _set_fixedParams(self, paramDict):
        self._fixedParams = paramDict
        for view in self._views:
            # view._set_fixedParams(paramDict)

            # This operation merges the two dictionaries with the second overriding the values of the first
            view._set_fixedParams({**paramDict, **view._fixedParams})


class MuMoTtimeEvolutionView(MuMoTview):
    """Time evolution view on model including state variables and noise.

    Specialised by :class:`MuMoTintegrateView` and :class:`MuMoTnoiseCorrelationsView`.

    """
    # list of all state variables
    _stateVarList = None
    # list of all state variables displayed in figure
    _stateVarListDisplay = None
    # 1st state variable
    _stateVariable1 = None
    # 2nd state variable
    _stateVariable2 = None
    # 3rd state variable
    _stateVariable3 = None
    # # 4th state variable
    # _stateVariable4 = None
    # # end time of numerical simulation of ODE system of the state variables
    # _tend = None
    # simulation length in time units
    _maxTime = None
    # time step of numerical simulation
    _tstep = None
    # total number of agents in the simulation
    _systemSize = None
    # the system state at the start of the simulation (timestep zero)
    _initialState = None
    # flag to plot proportions or full populations
    _plotProportions = None
    # Parameters for controller specific to this time evolution view
    _tEParams = None

    def __init__(self, model, controller, tEParams, showStateVars=None, figure=None, params=None, **kwargs):
        # if model._systemSize is None and model._constantSystemSize:
        #    print("Cannot construct time evolution -based plot until system size is set, using substitute()")
        #    return
        self._silent = kwargs.get('silent', False)
        super().__init__(model=model, controller=controller, figure=figure, params=params, **kwargs)
        # super().__init__(model, controller, figure, params, **kwargs)

        self._tEParams = tEParams

        with io.capture_output() as log:
            # if True:
            #     log=''

            self._systemSize = self._getSystemSize()

            if self._controller is None:
                # storing the initial state
                self._initialState = {}
                for state, pop in tEParams["initialState"].items():
                    if isinstance(state, str):
                        # convert string into SymPy symbol
                        self._initialState[parse_latex(state)] = pop
                    else:
                        self._initialState[state] = pop
                # # add to the _initialState the constant reactants
                # for constantReactant in self._mumotModel._getAllReactants()[1]:
                #     self._initialState[constantReactant] = freeParamDict[constantReactant]

                # Storing all values of MA-specific parameters
                self._maxTime = tEParams["maxTime"]
                self._plotProportions = tEParams["plotProportions"]
            else:
                # storing the initial state
                self._initialState = {}
                for state, pop in tEParams["initialState"][0].items():
                    if isinstance(state, str):
                        # Convert string into SymPy symbol
                        self._initialState[parse_latex(state)] = pop[0]
                    else:
                        self._initialState[state] = pop[0]

                # # add to the _initialState the constant reactants
                # for constantReactant in self._mumotModel._getAllReactants()[1]:
                #     self._initialState[constantReactant] = freeParamDict[constantReactant]

                # storing fixed params
                for key, value in tEParams.items():
                    if value[-1]:
                        if key == 'initialState':
                            self._fixedParams[key] = self._initialState
                        else:
                            self._fixedParams[key] = value[0]

            self._xlab = kwargs.get('xlab', 'time t')

            self._stateVarList = []
            for reactant in self._mumotModel._reactants:
                if reactant not in self._mumotModel._constantReactants:
                    self._stateVarList.append(reactant)

            if showStateVars:
                if type(showStateVars) == list:
                    self._stateVarListDisplay = showStateVars
                    for kk in range(len(self._stateVarListDisplay)):
                        self._stateVarListDisplay[kk] = parse_latex(self._stateVarListDisplay[kk])
                else:
                    self._showErrorMessage('Check input: should be of type list!')
            else:
                self._stateVarListDisplay = copy.deepcopy(self._stateVarList)
            self._stateVarListDisplay = sorted(self._stateVarListDisplay, key=str)

            self._stateVariable1 = self._stateVarList[0]
            if len(self._stateVarList) == 2 or len(self._stateVarList) == 3:
                self._stateVariable2 = self._stateVarList[1]
            if len(self._stateVarList) == 3:
                self._stateVariable3 = self._stateVarList[2]

            self._tstep = kwargs.get('tstep', 0.01)

            self._constructorSpecificParams(tEParams)

        self._logs.append(log)

        if not self._silent:
            self._plot_NumSolODE()

    def _get_eqsODE(self, y_old, time):
        """ Calculates right-hand side of ODE system."""
        SVsub = {}

        for kk in range(len(self._stateVarList)):
            SVsub[self._stateVarList[kk]] = y_old[kk]

        argDict = self._get_argDict()
        ode_sys = []
        for kk in range(len(self._stateVarList)):
            EQ = self._mumotModel._equations[self._stateVarList[kk]].subs(argDict)
            EQ = EQ.subs(SVsub)
            ode_sys.append(EQ)

        return ode_sys

    def _plot_NumSolODE(self):
        if not self._silent:  # @todo is this necessary?
            plt.figure(self._figureNum)
            plt.clf()
            self._resetErrorMessage()
        self._showErrorMessage(str(self))
        self._update_params()

    def _update_view_specific_params(self, freeParamDict=None):
        """getting other parameters specific to integrate"""
        if freeParamDict is None:
            freeParamDict = {}

        if self._controller is not None:
            if self._getWidgetParamValue('initialState', None) is not None:
                # self._initialState = self._getWidgetParamValue('initialState', None)
                # self._initialState = {sympy.Symbol(key): self._getWidgetParamValue('initialState', None)[key]
                #                       for key in self._getWidgetParamValue('initialState', None)}
                self._initialState = {self._safeSymbol(key): self._getWidgetParamValue('initialState', None)[key]
                                      for key in self._getWidgetParamValue('initialState', None)}
            else:
                for state in self._initialState.keys():
                    # add normal and constant reactants to the _initialState
                    self._initialState[state] = self._getInitialState(state, freeParamDict)  # freeParamDict[state] if state in self._mumotModel._constantReactants else self._controller._widgetsExtraParams['init' + str(state)].value
            if 'plotProportions' in self._tEParams:  # @todo JARM: I don't really understand logic of checking _tEParams but then retrieving the value from elsewhere
                self._plotProportions = self._getWidgetParamValue('plotProportions', self._controller._widgetsPlotOnly)  # self._fixedParams['plotProportions'] if self._fixedParams.get('plotProportions') is not None else self._controller._widgetsPlotOnly['plotProportions'].value
            self._maxTime = self._getWidgetParamValue('maxTime', self._controller._widgetsExtraParams)  # self._fixedParams['maxTime'] if self._fixedParams.get('maxTime') is not None else self._controller._widgetsExtraParams['maxTime'].value


class MuMoTintegrateView(MuMoTtimeEvolutionView):
    """Numerical solution of ODEs plot view on model."""

    # initial conditions used for proportion plot
    _y0 = None
    # save solution for redraw to switch between plotProportions = True and False
    _sol_ODE_dict = None
    # ordered list of colors to be used
    _colors = None

    def _constructorSpecificParams(self, _):
        if self._controller is not None:
            self._generatingCommand = "integrate"

        self._colors = []
        for idx, state in enumerate(sorted(self._initialState.keys(), key=str)):
            if state in self._stateVarListDisplay:
                self._colors.append(consts.LINE_COLOR_LIST[idx])
        # print(self._colors)

    def __init__(self, *args, **kwargs):
        self._ylab = kwargs.get('ylab', 'reactants')
        super().__init__(*args, **kwargs)
        # self._generatingCommand = "numSimStateVar"

    def _plot_NumSolODE(self, _=None):
        self._show_computation_start()

        super()._plot_NumSolODE()

        with io.capture_output() as log:
            self._log("numerical integration of ODE system")
        self._logs.append(log)

        # check input
        for nn in range(len(self._stateVarListDisplay)):
            if self._stateVarListDisplay[nn] not in self._stateVarList:
                self._showErrorMessage(f"Warning: {self._stateVarListDisplay[nn]} is no reactant in the current model.")
                return None

        # if self._stateVariable1 not in self._mumotModel._reactants:
        #     self._showErrorMessage('Warning:  ' + str(self._stateVariable1) + '  is no reactant in the current model.')
        # if self._stateVariable2 not in self._mumotModel._reactants:
        #     self._showErrorMessage('Warning:  ' + str(self._stateVariable2) + '  is no reactant in the current model.')
        # if self._stateVariable3:
        #     if self._stateVariable3 not in self._mumotModel._reactants:
        #         self._showErrorMessage('Warning:  ' + str(self._stateVariable3) + '  is no reactant in the current model.')
        # if self._stateVariable4:
        #     if self._stateVariable4 not in self._mumotModel._reactants:
        #         self._showErrorMessage('Warning:  ' + str(self._stateVariable4) + '  is no reactant in the current model.')

        NrDP = int(self._maxTime / self._tstep) + 1
        time = np.linspace(0, self._maxTime, NrDP)

        initDict = self._initialState  # self._get_tEParams()   # self._initialState

        # if len(initDict) < 2 or len(initDict) > 4:
        #     self._showErrorMessage("Not implemented: This feature is available only for systems with 2, 3 or 4 time-dependent reactants!")

        y0 = []
        for nn in range(len(self._stateVarList)):
            # SVi0 = initDict[sympy.Symbol(latex(sympy.Symbol('Phi^0_' + str(self._stateVarList[nn]))))]
            SVi0 = initDict[sympy.Symbol(str(self._stateVarList[nn]))]
            y0.append(SVi0)

        self._y0 = y0

        sol_ODE = odeint(self._get_eqsODE, y0, time)

        sol_ODE_dict = {}
        for nn in range(len(self._stateVarList)):
            sol_ODE_dict[str(self._stateVarList[nn])] = sol_ODE[:, nn]

        self._sol_ODE_dict = sol_ODE_dict
        # x_data = [time for kk in range(len(self._get_eqsODE(y0, time)))]
        x_data = [time for kk in range(len(self._stateVarListDisplay))]
        # y_data = [sol_ODE[:, kk] for kk in range(len(self._get_eqsODE(y0, time)))]
        y_data = [sol_ODE_dict[str(self._stateVarListDisplay[kk])]
                  for kk in range(len(self._stateVarListDisplay))]

        if not self._plotProportions:
            syst_Size = sympy.Symbol('systemSize')
            sysS = syst_Size.subs(self._get_argDict())
            # sysS = syst_Size.subs(self._getSystemSize())
            sysS = sympy.N(sysS)
            # y_scaling = np.sum(np.asarray(y0))
            # if y_scaling > 0:
            #    sysS = sysS/y_scaling
            for nn in range(len(y_data)):
                y_temp = np.copy(y_data[nn])
                for kk in range(len(y_temp)):
                    y_temp[kk] = y_temp[kk] * sysS
                y_data[nn] = y_temp
            c_labels = [r'$' + str(self._stateVarListDisplay[nn]) + '$' for nn in range(len(self._stateVarListDisplay))]
        else:
            c_labels = [r'$' + latex(sympy.Symbol('Phi_' + str(self._stateVarListDisplay[nn]))) + '$' for nn in range(len(self._stateVarListDisplay))]

        c_labels = [utils._doubleUnderscorify(utils._greekPrependify(c_labels[jj])) for jj in range(len(c_labels))]

        # c_labels = [r'$' + latex(sympy.Symbol('Phi_' + str(self._stateVariable1))) + '$', r'$' + latex(sympy.Symbol('Phi_' + str(self._stateVariable2))) + '$']
        # if self._stateVariable3:
        #     c_labels.append(r'$' + latex(sympy.Symbol('Phi_' + str(self._stateVariable3))) + '$')
        # if self._stateVariable4:
        #     c_labels.append(r'$' + latex(sympy.Symbol('Phi_' + str(self._stateVariable4))) + '$')

        if self._chooseXrange:
            choose_xrange = self._chooseXrange
        else:
            choose_xrange = [0, self._maxTime]
        _fig_formatting_2D(xdata=x_data, ydata=y_data, xlab=self._xlab,
                           ylab=self._ylab, choose_xrange=choose_xrange,
                           choose_yrange=self._chooseYrange,
                           fontsize=self._axes_font_size, curvelab=c_labels,
                           legend_loc=self._legend_loc, grid=True,
                           legend_fontsize=self._legend_fontsize,
                           line_color_list=self._colors)

        with io.capture_output() as log:
            print('Last point on curve:')
            if not self._plotProportions:
                for nn in range(len(self._stateVarListDisplay)):
                    out = latex(str(self._stateVarListDisplay[nn])) + '(t =' + str(_roundNumLogsOut(x_data[nn][-1])) + ') = ' + str(_roundNumLogsOut(y_data[nn][-1]))
                    out = utils._doubleUnderscorify(utils._greekPrependify(out))
                    display(Math(out))
            else:
                for nn in range(len(self._stateVarListDisplay)):
                    out = latex(sympy.Symbol('Phi_' + str(self._stateVarListDisplay[nn]))) + '(t =' + str(_roundNumLogsOut(x_data[nn][-1])) + ') = ' + str(_roundNumLogsOut(y_data[nn][-1]))
                    out = utils._doubleUnderscorify(utils._greekPrependify(out))
                    display(Math(out))
        self._logs.append(log)

        self._show_computation_stop()

    def _redrawOnly(self, _=None):
        super()._plot_NumSolODE()
        self._update_params()
        NrDP = int(self._maxTime / self._tstep) + 1
        time = np.linspace(0, self._maxTime, NrDP)
        # x_data = [time for kk in range(len(self._get_eqsODE(y0, time)))]
        x_data = [time for kk in range(len(self._stateVarListDisplay))]
        # y_data = [sol_ODE[:, kk] for kk in range(len(self._get_eqsODE(y0, time)))]
        y_data = [self._sol_ODE_dict[str(self._stateVarListDisplay[kk])] for kk in range(len(self._stateVarListDisplay))]

        if not self._plotProportions:
            syst_Size = sympy.Symbol('systemSize')
            sysS = syst_Size.subs(self._get_argDict())
            # sysS = syst_Size.subs(self._getSystemSize())
            sysS = sympy.N(sysS)
            # y_scaling = np.sum(np.asarray(self._y0))
            # if y_scaling > 0:
            #     sysS = sysS / y_scaling
            for nn in range(len(y_data)):
                y_temp = np.copy(y_data[nn])
                for kk in range(len(y_temp)):
                    y_temp[kk] = y_temp[kk] * sysS
                y_data[nn] = y_temp
            c_labels = [r'$' + str(self._stateVarListDisplay[nn]) + '$' for nn in range(len(self._stateVarListDisplay))]
        else:
            c_labels = [r'$' + latex(sympy.Symbol('Phi_' + str(self._stateVarListDisplay[nn]))) + '$' for nn in range(len(self._stateVarListDisplay))]

        c_labels = [utils._doubleUnderscorify(utils._greekPrependify(c_labels[jj])) for jj in range(len(c_labels))]

        # c_labels = [r'$' + latex(sympy.Symbol('Phi_' + str(self._stateVariable1))) + '$', r'$' + latex(sympy.Symbol('Phi_' + str(self._stateVariable2))) + '$']
        # if self._stateVariable3:
        #     c_labels.append(r'$' + latex(sympy.Symbol('Phi_' + str(self._stateVariable3))) + '$')
        # if self._stateVariable4:
        #     c_labels.append(r'$' + latex(sympy.Symbol('Phi_' + str(self._stateVariable4))) + '$')

        if self._chooseXrange:
            choose_xrange = self._chooseXrange
        else:
            choose_xrange = [0, self._maxTime]
        _fig_formatting_2D(xdata=x_data, ydata=y_data, xlab=self._xlab,
                           ylab=self._ylab, choose_xrange=choose_xrange,
                           choose_yrange=self._chooseYrange,
                           fontsize=self._axes_font_size, curvelab=c_labels,
                           legend_loc=self._legend_loc, grid=True,
                           legend_fontsize=self._legend_fontsize)

        with io.capture_output() as log:
            print('Last point on curve:')
            if not self._plotProportions:
                for nn in range(len(self._stateVarListDisplay)):
                    out = latex(str(self._stateVarListDisplay[nn])) + '(t =' + str(x_data[nn][-1]) + ') = ' + str(str(y_data[nn][-1]))
                    out = utils._doubleUnderscorify(utils._greekPrependify(out))
                    display(Math(out))
            else:
                for nn in range(len(self._stateVarListDisplay)):
                    out = latex(sympy.Symbol('Phi_' + str(self._stateVarListDisplay[nn]))) + '(t =' + str(x_data[nn][-1]) + ') = ' + str(str(y_data[nn][-1]))
                    out = utils._doubleUnderscorify(utils._greekPrependify(out))
                    display(Math(out))
        self._logs.append(log)

    def _build_bookmark(self, includeParams=True):
        if not self._silent:
            log_str = "bookmark = "
        else:
            log_str = ""

        log_str += "<modelName>." + self._generatingCommand + "(showStateVars=["
        for nn in range(len(self._stateVarListDisplay)):
            if nn == len(self._stateVarListDisplay) - 1:
                log_str += "'" + str(self._stateVarListDisplay[nn]) + "'], "
            else:
                log_str += "'" + str(self._stateVarListDisplay[nn]) + "', "

        if "initialState" not in self._generatingKwargs.keys():
            initState_str = {latex(state): pop
                             for state, pop in self._initialState.items()
                             if state not in self._mumotModel._constantReactants}
            log_str += "initialState = " + str(initState_str) + ", "
        if "maxTime" not in self._generatingKwargs.keys():
            log_str += "maxTime = " + str(self._maxTime) + ", "
        if "plotProportions" not in self._generatingKwargs.keys():
            log_str += "plotProportions = " + str(self._plotProportions) + ", "
        if includeParams:
            log_str += self._get_bookmarks_params() + ", "
        if len(self._generatingKwargs) > 0:
            for key in self._generatingKwargs:
                if type(self._generatingKwargs[key]) == str:
                    log_str += key + " = " + "\'" + str(self._generatingKwargs[key]) + "\'" + ", "
                else:
                    log_str += key + " = " + str(self._generatingKwargs[key]) + ", "
        log_str += "bookmark = False"
        log_str += ")"
        log_str = utils._greekPrependify(log_str)
        log_str = log_str.replace('\\', '\\\\')
        log_str = log_str.replace('\\\\\\\\', '\\\\')

        return log_str


class MuMoTnoiseCorrelationsView(MuMoTtimeEvolutionView):
    """Noise correlations around fixed points plot view on model."""

    # equations of motion for first order moments of noise variables
    _EOM_1stOrdMomDict = None
    # equations of motion for second order moments of noise variables
    _EOM_2ndOrdMomDict = None
    # upper bound of simulation time for dynamical system to reach equilibrium (can be set via keyword)
    _maxTimeDS = None
    # time step of simulation for dynamical system to reach equilibrium (can be set via keyword)
    _tstepDS = None

    def _constructorSpecificParams(self, _):
        if self._controller is not None:
            self._generatingCommand = "noiseCorrelations"

    def __init__(self, model, controller, NCParams, EOM_1stOrdMom,
                 EOM_2ndOrdMom, figure=None, params=None, **kwargs):
        self._EOM_1stOrdMomDict = EOM_1stOrdMom
        self._EOM_2ndOrdMomDict = EOM_2ndOrdMom
        self._maxTimeDS = kwargs.get('maxTimeDS', 50)
        self._tstepDS = kwargs.get('tstepDS', 0.01)
        self._ylab = kwargs.get('ylab', 'noise correlations')
        self._silent = kwargs.get('silent', False)
        super().__init__(model=model, controller=controller, tEParams=NCParams,
                         showStateVars=None, figure=figure, params=params, **kwargs)
        # super().__init__(model, controller, None, figure, params, **kwargs)

        if len(self._stateVarList) < 1 or len(self._stateVarList) > 3:
            self._showErrorMessage("Not implemented: This feature is available only for systems with 1, 2 or 3 time-dependent reactants!")
            return None

    def _plot_NumSolODE(self, _=None):
        self._show_computation_start()

        super()._plot_NumSolODE()

        with io.capture_output() as log:
            self._log("numerical integration of noise correlations")
        self._logs.append(log)

        # Check input
        for nn in range(len(self._stateVarListDisplay)):
            if self._stateVarListDisplay[nn] not in self._stateVarList:
                self._showErrorMessage(f"Warning: {self._stateVarListDisplay[nn]} is no reactant in the current model.")
                return None

        eps = 5e-3
        systemSize = sympy.Symbol('systemSize')

        NrDP = int(self._maxTimeDS / self._tstepDS) + 1
        time = np.linspace(0, self._maxTimeDS, NrDP)
        # NrDP = int(self._tend / self._tstep) + 1
        # time = np.linspace(0, self._tend, NrDP)

        initDict = self._initialState

        SV1_0 = initDict[sympy.Symbol(str(self._stateVariable1))]
        y0 = [SV1_0]
        if self._stateVariable2:
            SV2_0 = initDict[sympy.Symbol(str(self._stateVariable2))]
            y0.append(SV2_0)
        if self._stateVariable3:
            SV3_0 = initDict[sympy.Symbol(str(self._stateVariable3))]
            y0.append(SV3_0)

        sol_ODE = odeint(self._get_eqsODE, y0, time)

        if self._stateVariable3:
            realEQsol, eigList = self._get_fixedPoints3d()
        elif self._stateVariable2:
            realEQsol, eigList = self._get_fixedPoints2d()
        else:
            realEQsol, eigList = self._get_fixedPoints1d()

        y_stationary = [sol_ODE[-1, kk] for kk in range(len(y0))]

        if realEQsol != [] and realEQsol is not None:
            steadyStateReached = False
            for kk in range(len(realEQsol)):
                if self._stateVariable3:
                    if abs(realEQsol[kk][self._stateVariable1] - y_stationary[0]) <= eps and abs(realEQsol[kk][self._stateVariable2] - y_stationary[1]) <= eps and abs(realEQsol[kk][self._stateVariable3] - y_stationary[2]) <= eps:
                        if all(sympy.sign(sympy.re(lam)) < 0 for lam in eigList[kk]):
                            steadyStateReached = True
                            steadyStateDict = {self._stateVariable1: realEQsol[kk][self._stateVariable1],
                                               self._stateVariable2: realEQsol[kk][self._stateVariable2],
                                               self._stateVariable3: realEQsol[kk][self._stateVariable3]}

                elif self._stateVariable2:
                    if abs(realEQsol[kk][self._stateVariable1] - y_stationary[0]) <= eps and abs(realEQsol[kk][self._stateVariable2] - y_stationary[1]) <= eps:
                        if all(sympy.sign(sympy.re(lam)) < 0 for lam in eigList[kk]):
                            steadyStateReached = True
                            steadyStateDict = {self._stateVariable1: realEQsol[kk][self._stateVariable1],
                                               self._stateVariable2: realEQsol[kk][self._stateVariable2]}
                else:
                    if abs(realEQsol[kk][self._stateVariable1] - y_stationary[0]) <= eps:
                        if all(sympy.sign(sympy.re(lam)) < 0 for lam in eigList[kk]):
                            steadyStateReached = True
                            steadyStateDict = {self._stateVariable1: realEQsol[kk][self._stateVariable1]}

            if not steadyStateReached:
                self._show_computation_stop()
                self._showErrorMessage('ODE system could not reach stable steady state: '
                                       'Try changing the initial conditions or model parameters '
                                       'using the sliders provided, increase simulation time, or decrease timestep tstep.')
                return None
        else:
            steadyStateReached = 'uncertain'
            self._showErrorMessage('Warning: ODE system may not have reached a steady state. '
                                   'Values of state variables at t=maxTimeDS were substituted '
                                   "(maxTimeDS can be set via keyword 'maxTimeDS = <number>').")
            if self._stateVariable3:
                steadyStateDict = {self._stateVariable1: y_stationary[0],
                                   self._stateVariable2: y_stationary[1],
                                   self._stateVariable3: y_stationary[2]}
            elif self._stateVariable2:
                steadyStateDict = {self._stateVariable1: y_stationary[0],
                                   self._stateVariable2: y_stationary[1]}
            else:
                steadyStateDict = {self._stateVariable1: y_stationary[0]}

        with io.capture_output() as log:
            if steadyStateReached == 'uncertain':
                print('This plot depicts the noise-noise auto-correlation and '
                      'cross-correlation functions around the following state (this might NOT be a steady state).')
            else:
                print('This plot depicts the noise-noise auto-correlation and '
                      'cross-correlation functions around the following stable steady state:')
            for reactant in steadyStateDict:
                out = 'Phi^s_{' + latex(str(reactant)) + '} = ' + latex(_roundNumLogsOut(steadyStateDict[reactant]))
                out = utils._doubleUnderscorify(utils._greekPrependify(out))
                display(Math(out))
        self._logs.append(log)

        argDict_tmp = self._get_argDict()
        # Mutate key names of the argDict and the steadyStateDict so that they indicate concentrations via prefix Phi
        argDict = {}
        for key, val in argDict_tmp.items():
            key_phi = sympy.Symbol(f"Phi_{key}") if key in self._mumotModel._constantReactants else key
            argDict[key_phi] = val
        steadyStateDictPhi = {}
        for key, val in steadyStateDict.items():
            key_phi = sympy.Symbol(f"Phi_{key}") if key in self._mumotModel._reactants else key
            steadyStateDictPhi[key_phi] = val
        
        EOM_1stOrdMomDict = copy.deepcopy(self._EOM_1stOrdMomDict)
        for sol in EOM_1stOrdMomDict:
            EOM_1stOrdMomDict[sol] = EOM_1stOrdMomDict[sol].subs(steadyStateDictPhi)
            EOM_1stOrdMomDict[sol] = EOM_1stOrdMomDict[sol].subs(argDict)

        EOM_2ndOrdMomDict = copy.deepcopy(self._EOM_2ndOrdMomDict)

        SOL_2ndOrdMomDict = self._numericSol2ndOrdMoment(EOM_2ndOrdMomDict, steadyStateDictPhi, argDict)

        time_depend_noise = []
        for reactant in self._mumotModel._reactants:
            if reactant not in self._mumotModel._constantReactants:
                time_depend_noise.append(sympy.Symbol(f"eta_{reactant}"))

        noiseCorrEOM = []
        noiseCorrEOMdict = {}
        for sym in time_depend_noise:
            for key in EOM_1stOrdMomDict:
                noiseCorrEOMdict[sym * key] = sympy.expand(sym * EOM_1stOrdMomDict[key])

        M_1 = Function('M_1')
        M_2 = Function('M_2')
        eta_SV1 = sympy.Symbol(f"eta_{self._stateVariable1}")
        cVar1 = sympy.symbols('cVar1')
        if self._stateVariable2:
            eta_SV2 = sympy.Symbol(f"eta_{self._stateVariable2}")
            cVar2, cVar3, cVar4 = sympy.symbols('cVar2 cVar3 cVar4')
        if self._stateVariable3:
            eta_SV3 = sympy.Symbol(f"eta_{self._stateVariable3}")
            cVar5, cVar6, cVar7, cVar8, cVar9 = sympy.symbols('cVar5 cVar6 cVar7 cVar8 cVar9')

        cVarSubdict = {}
        if len(time_depend_noise) == 1:
            cVarSubdict[eta_SV1 * M_1(eta_SV1)] = cVar1
            # auto-correlation
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV1 * M_1(eta_SV1)])
        elif len(time_depend_noise) == 2:
            cVarSubdict[eta_SV1 * M_1(eta_SV1)] = cVar1
            cVarSubdict[eta_SV2 * M_1(eta_SV2)] = cVar2
            cVarSubdict[eta_SV1 * M_1(eta_SV2)] = cVar3
            cVarSubdict[eta_SV2 * M_1(eta_SV1)] = cVar4
            # auto-correlation
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV1 * M_1(eta_SV1)])
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV2 * M_1(eta_SV2)])
            # cross-correlation
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV1 * M_1(eta_SV2)])
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV2 * M_1(eta_SV1)])
        elif len(time_depend_noise) == 3:
            cVarSubdict[eta_SV1 * M_1(eta_SV1)] = cVar1
            cVarSubdict[eta_SV2 * M_1(eta_SV2)] = cVar2
            cVarSubdict[eta_SV3 * M_1(eta_SV3)] = cVar3
            cVarSubdict[eta_SV1 * M_1(eta_SV2)] = cVar4
            cVarSubdict[eta_SV2 * M_1(eta_SV1)] = cVar5
            cVarSubdict[eta_SV1 * M_1(eta_SV3)] = cVar6
            cVarSubdict[eta_SV3 * M_1(eta_SV1)] = cVar7
            cVarSubdict[eta_SV2 * M_1(eta_SV3)] = cVar8
            cVarSubdict[eta_SV3 * M_1(eta_SV2)] = cVar9
            # auto-correlation
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV1 * M_1(eta_SV1)])
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV2 * M_1(eta_SV2)])
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV3 * M_1(eta_SV3)])
            # cross-correlation
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV1 * M_1(eta_SV2)])
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV2 * M_1(eta_SV1)])
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV1 * M_1(eta_SV3)])
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV3 * M_1(eta_SV1)])
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV2 * M_1(eta_SV3)])
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV3 * M_1(eta_SV2)])
        else:
            self._showErrorMessage('Only implemented for 2 or 3 time-dependent noise variables.')

        for kk in range(len(noiseCorrEOM)):
            noiseCorrEOM[kk] = noiseCorrEOM[kk].subs(cVarSubdict)

        def noiseODEsys(yin, t):
            dydt = copy.deepcopy(noiseCorrEOM)
            for kk in range(len(dydt)):
                if self._stateVariable3:
                    dydt[kk] = dydt[kk].subs({cVar1: yin[0], cVar2: yin[1], cVar3: yin[2], cVar4: yin[3],
                                              cVar5: yin[4], cVar6: yin[5], cVar7: yin[6], cVar8: yin[7], cVar9: yin[8]})
                elif self._stateVariable2:
                    dydt[kk] = dydt[kk].subs({cVar1: yin[0], cVar2: yin[1], cVar3: yin[2], cVar4: yin[3]})
                else:
                    dydt[kk] = dydt[kk].subs({cVar1: yin[0]})
                dydt[kk] = dydt[kk].evalf()
            return dydt

        NrDP = int(self._maxTime / self._tstep) + 1
        time = np.linspace(0, self._maxTime, NrDP)

        if len(SOL_2ndOrdMomDict) > 0:
            if self._stateVariable3:
                y0 = [SOL_2ndOrdMomDict[M_2(eta_SV1**2)], SOL_2ndOrdMomDict[M_2(eta_SV2**2)], SOL_2ndOrdMomDict[M_2(eta_SV3**2)],
                      SOL_2ndOrdMomDict[M_2(eta_SV1 * eta_SV2)], SOL_2ndOrdMomDict[M_2(eta_SV1 * eta_SV2)],
                      SOL_2ndOrdMomDict[M_2(eta_SV1 * eta_SV3)], SOL_2ndOrdMomDict[M_2(eta_SV1 * eta_SV3)],
                      SOL_2ndOrdMomDict[M_2(eta_SV2 * eta_SV3)], SOL_2ndOrdMomDict[M_2(eta_SV2 * eta_SV3)]]
            elif self._stateVariable2:
                y0 = [SOL_2ndOrdMomDict[M_2(eta_SV1**2)], SOL_2ndOrdMomDict[M_2(eta_SV2**2)],
                      SOL_2ndOrdMomDict[M_2(eta_SV1 * eta_SV2)], SOL_2ndOrdMomDict[M_2(eta_SV1 * eta_SV2)]]
            else:
                y0 = [SOL_2ndOrdMomDict[M_2(eta_SV1**2)]]
        else:
            self._showErrorMessage('Could not compute Second Order Moments. '
                                   'Could not generate figure. '
                                   'Try different initial conditions in the Advanced options tab! ')
            return None

        sol_ODE = odeint(noiseODEsys, y0, time)  # sol_ODE overwritten

        x_data = [time for kk in range(len(y0))]
        y_data = [sol_ODE[:, kk] for kk in range(len(y0))]
        noiseNorm = systemSize.subs(argDict)
        noiseNorm = sympy.N(noiseNorm)
        for nn in range(len(y_data)):
            y_temp = np.copy(y_data[nn])
            for kk in range(len(y_temp)):
                y_temp[kk] = y_temp[kk] / noiseNorm
            y_data[nn] = y_temp

        if self._stateVariable3:
            c_labels = [r'$<' + latex(eta_SV1) + '(t)' + latex(eta_SV1) + '(0)' + '>$',
                        r'$<' + latex(eta_SV2) + '(t)' + latex(eta_SV2) + '(0)' + '>$',
                        r'$<' + latex(eta_SV3) + '(t)' + latex(eta_SV3) + '(0)' + '>$',
                        r'$<' + latex(eta_SV2) + '(t)' + latex(eta_SV1) + '(0)' + '>$',
                        r'$<' + latex(eta_SV1) + '(t)' + latex(eta_SV2) + '(0)' + '>$',
                        r'$<' + latex(eta_SV3) + '(t)' + latex(eta_SV1) + '(0)' + '>$',
                        r'$<' + latex(eta_SV1) + '(t)' + latex(eta_SV3) + '(0)' + '>$',
                        r'$<' + latex(eta_SV3) + '(t)' + latex(eta_SV2) + '(0)' + '>$',
                        r'$<' + latex(eta_SV2) + '(t)' + latex(eta_SV3) + '(0)' + '>$']

        elif self._stateVariable2:
            c_labels = [r'$<' + latex(eta_SV1) + '(t)' + latex(eta_SV1) + '(0)' + '>$',
                        r'$<' + latex(eta_SV2) + '(t)' + latex(eta_SV2) + '(0)' + '>$',
                        r'$<' + latex(eta_SV2) + '(t)' + latex(eta_SV1) + '(0)' + '>$',
                        r'$<' + latex(eta_SV1) + '(t)' + latex(eta_SV2) + '(0)' + '>$']
        else:
            c_labels = [r'$<' + latex(eta_SV1) + '(t)' + latex(eta_SV1) + '(0)' + '>$']

        c_labels = [utils._doubleUnderscorify(utils._greekPrependify(c_labels[jj]))
                    for jj in range(len(c_labels))]

        if self._chooseXrange:
            choose_xrange = self._chooseXrange
        else:
            choose_xrange = [0, self._maxTime]
        _fig_formatting_2D(xdata=x_data, ydata=y_data, xlab=self._xlab,
                           ylab=self._ylab, choose_xrange=choose_xrange,
                           choose_yrange=self._chooseYrange,
                           fontsize=self._axes_font_size, curvelab=c_labels,
                           legend_loc=self._legend_loc, grid=True,
                           legend_fontsize=self._legend_fontsize)

        self._show_computation_stop()

    def _numericSol2ndOrdMoment(self, EOM_2ndOrdMomDict, steadyStateDict, argDict):
        for sol in EOM_2ndOrdMomDict:
            EOM_2ndOrdMomDict[sol] = EOM_2ndOrdMomDict[sol].subs(steadyStateDict)
            EOM_2ndOrdMomDict[sol] = EOM_2ndOrdMomDict[sol].subs(argDict)

        # eta_SV1 = sympy.Symbol('eta_' + str(self._stateVarList[0]))
        # eta_SV2 = sympy.Symbol('eta_' + str(self._stateVarList[1]))
        # M_1, M_2 = sympy.symbols('M_1 M_2')
        # if len(self._stateVarList) == 3:
        #     eta_SV3 = sympy.Symbol('eta_' + str(self._stateVarList[2]))

        M_1 = Function('M_1')
        M_2 = Function('M_2')
        eta_SV1 = sympy.Symbol('eta_' + str(self._stateVariable1))
        if self._stateVariable2:
            eta_SV2 = sympy.Symbol('eta_' + str(self._stateVariable2))
        if self._stateVariable3:
            eta_SV3 = sympy.Symbol('eta_' + str(self._stateVariable3))

        SOL_2ndOrdMomDict = {}
        EQsys2ndOrdMom = []
        if self._stateVariable3:
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV1 * eta_SV1)])
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV2 * eta_SV2)])
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV3 * eta_SV3)])
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV1 * eta_SV2)])
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV1 * eta_SV3)])
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV2 * eta_SV3)])
            solEQsys2ndOrdMom = sympy.linsolve(EQsys2ndOrdMom, [M_2(eta_SV1 * eta_SV1),
                                                                M_2(eta_SV2 * eta_SV2),
                                                                M_2(eta_SV3 * eta_SV3),
                                                                M_2(eta_SV1 * eta_SV2),
                                                                M_2(eta_SV1 * eta_SV3),
                                                                M_2(eta_SV2 * eta_SV3)])
            if len(list(solEQsys2ndOrdMom)) == 0:
                return SOL_2ndOrdMomDict
            else:
                SOL_2ndOrderMom = list(solEQsys2ndOrdMom)[0]  # only one set of solutions (if any) in linear system of equations; hence index [0]
                SOL_2ndOrdMomDict[M_2(eta_SV1 * eta_SV1)] = SOL_2ndOrderMom[0]
                SOL_2ndOrdMomDict[M_2(eta_SV2 * eta_SV2)] = SOL_2ndOrderMom[1]
                SOL_2ndOrdMomDict[M_2(eta_SV3 * eta_SV3)] = SOL_2ndOrderMom[2]
                SOL_2ndOrdMomDict[M_2(eta_SV1 * eta_SV2)] = SOL_2ndOrderMom[3]
                SOL_2ndOrdMomDict[M_2(eta_SV1 * eta_SV3)] = SOL_2ndOrderMom[4]
                SOL_2ndOrdMomDict[M_2(eta_SV2 * eta_SV3)] = SOL_2ndOrderMom[5]

        elif self._stateVariable2:
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV1 * eta_SV1)])
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV2 * eta_SV2)])
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV1 * eta_SV2)])
            solEQsys2ndOrdMom = sympy.linsolve(EQsys2ndOrdMom, [M_2(eta_SV1 * eta_SV1),
                                                                M_2(eta_SV2 * eta_SV2),
                                                                M_2(eta_SV1 * eta_SV2)])

            if len(list(solEQsys2ndOrdMom)) == 0:
                return SOL_2ndOrdMomDict
            else:
                SOL_2ndOrderMom = list(solEQsys2ndOrdMom)[0]  # only one set of solutions (if any) in linear system of equations
                SOL_2ndOrdMomDict[M_2(eta_SV1 * eta_SV1)] = SOL_2ndOrderMom[0]
                SOL_2ndOrdMomDict[M_2(eta_SV2 * eta_SV2)] = SOL_2ndOrderMom[1]
                SOL_2ndOrdMomDict[M_2(eta_SV1 * eta_SV2)] = SOL_2ndOrderMom[2]

        else:
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV1 * eta_SV1)])
            solEQsys2ndOrdMom = sympy.linsolve(EQsys2ndOrdMom, [M_2(eta_SV1 * eta_SV1)])
            if len(list(solEQsys2ndOrdMom)) == 0:
                return SOL_2ndOrdMomDict
            else:
                SOL_2ndOrderMom = list(solEQsys2ndOrdMom)[0]  # only one set of solutions (if any) in linear system of equations
                SOL_2ndOrdMomDict[M_2(eta_SV1 * eta_SV1)] = SOL_2ndOrderMom[0]

        return SOL_2ndOrdMomDict

    def _build_bookmark(self, includeParams: bool = True):
        if not self._silent:
            log_str = "bookmark = "
        else:
            log_str = ""

        log_str += "<modelName>." + self._generatingCommand + "("
        if "initialState" not in self._generatingKwargs.keys():
            initState_str = {latex(state): pop
                             for state, pop in self._initialState.items()
                             if state not in self._mumotModel._constantReactants}
            log_str += "initialState = " + str(initState_str) + ", "
        if "maxTime" not in self._generatingKwargs.keys():
            log_str += "maxTime = " + str(self._maxTime) + ", "
        if includeParams:
            log_str += self._get_bookmarks_params() + ", "
        if len(self._generatingKwargs) > 0:
            for key in self._generatingKwargs:
                if type(self._generatingKwargs[key]) == str:
                    log_str += key + " = " + "\'" + str(self._generatingKwargs[key]) + "\'" + ", "
                else:
                    log_str += key + " = " + str(self._generatingKwargs[key]) + ", "
        log_str += "bookmark = False"
        log_str += ")"
        log_str = utils._greekPrependify(log_str)
        log_str = log_str.replace('\\', '\\\\')
        log_str = log_str.replace('\\\\\\\\', '\\\\')

        return log_str


class MuMoTfieldView(MuMoTview):
    """Field view on model.

    Specialised by :class:`MuMoTvectorView` and :class:`MuMoTstreamView`.

    """
    # 1st state variable (x-dimension)
    _stateVariable1 = None
    # 2nd state variable (y-dimension)
    _stateVariable2 = None
    # 3rd state variable (z-dimension)
    _stateVariable3 = None
    # stores fixed points
    _FixedPoints = None
    # stores 2nd Order moments of noise-noise correlations
    _SOL_2ndOrdMomDict = None
    # X ordinates array
    _X = None
    # Y ordinates array
    _Y = None
    # Z ordinates array
    _Z = None
    # X derivatives array
    _X = None
    # Y derivatives array
    _Y = None
    # Z derivatives array
    _Z = None
    # speed array
    _speed = None
    # class-global dictionary of memoised masks with (mesh size, dimension) as key
    _mask = {}
    # z-label
    _zlab = None
    # flag to run SSA simulations to compute noise ellipse
    _showSSANoise = None
    # flag to show Noise
    _showNoise = None
    # fixed points for logs
    _realEQsol = None
    # eigenvalues for logs
    _EV = None
    # eigenvectors for logs
    _Evects = None
    # random seed  (for computing SSA noise)
    _randomSeed = None
    # simulation length (for computing SSA noise)
    _maxTime = None
    # reactants to display on the two axes
    # _finalViewAxes = None
    # flag to plot proportions or full populations
    _plotProportions = None
    # number of runs to execute  (for computing SSA noise)
    _runs = None
    # flag to set if the results from multimple runs must be aggregated or not (for computing SSA noise)
    _aggregateResults = None

    def __init__(self, model, controller, fieldParams, SOL_2ndOrd,
                 stateVariable1, stateVariable2=None, stateVariable3=None,
                 figure=None, params=None, **kwargs):
        if model._systemSize is None and model._constantSystemSize:
            print("Cannot construct field-based plot until system size is set, using substitute()")
            return

        self._silent = kwargs.get('silent', False)
        super().__init__(model=model, controller=controller, figure=figure, params=params, **kwargs)

        with io.capture_output() as log:
            self._showFixedPoints = kwargs.get('showFixedPoints', False)
            self._xlab = r'' + kwargs.get('xlab', r'$' + r'\Phi_{' + utils._doubleUnderscorify(utils._greekPrependify(str(stateVariable1))) + '}$')
            if stateVariable2:
                self._ylab = r'' + kwargs.get('ylab', r'$' + r'\Phi_{' + utils._doubleUnderscorify(utils._greekPrependify(str(stateVariable2))) + '}$')
            if stateVariable3:
                self._zlab = r'' + kwargs.get('zlab', r'$' + r'\Phi_{' + utils._doubleUnderscorify(utils._greekPrependify(str(stateVariable3))) + '}$')

            self._stateVariable1 = parse_latex(stateVariable1)
            if stateVariable2 is not None:
                self._stateVariable2 = parse_latex(stateVariable2)
            if stateVariable3 is not None:
                self._axes3d = True
                self._stateVariable3 = parse_latex(stateVariable3)
            _mask = {}

            self._SOL_2ndOrdMomDict = SOL_2ndOrd

            self._showNoise = kwargs.get('showNoise', False)

            if self._showNoise and self._SOL_2ndOrdMomDict is None and self._stateVariable3 is None:
                self._showSSANoise = True
            else:
                self._showSSANoise = False

            if stateVariable3 is not None:
                self._chooseXrange = None
                self._chooseYrange = None

            if self._controller is None:
                # storing all values of MA-specific parameters
                self._maxTime = fieldParams["maxTime"]
                self._randomSeed = fieldParams["randomSeed"]
                # final_x = str(parse_latex(fieldParams.get("final_x", latex(sorted(self._mumotModel._getAllReactants()[0], key=str)[0]))))
                # final_y = str(parse_latex(fieldParams.get("final_y", latex(sorted(self._mumotModel._getAllReactants()[0], key=str)[0]))))
                # self._finalViewAxes = (final_x, final_y)
                # self._plotProportions = fieldParams["plotProportions"]
                self._runs = fieldParams.get('runs', 20)
                self._aggregateResults = fieldParams.get('aggregateResults', True)

            else:
                # storing fixed params
                for key, value in fieldParams.items():
                    if value[-1]:
                        self._fixedParams[key] = value[0]
                self._update_field_params_wdgt()

        self._logs.append(log)
        if not self._silent:
            self._plot_field()

    # @todo: this function will be useful to implement a solution to issue #282
    def _update_field_params_wdgt(self):
        """Update the widgets related to the _showSSANoise

        (it cannot be a :class:`MuMoTcontroller` method because with multi-controller it needs to point to the right ``_controller``)
        """
        if self._showSSANoise:
            if self._controller._advancedTabWidget is not None:
                self._controller._advancedTabWidget.layout.display = 'flex'
            if self._controller._widgetsExtraParams.get('maxTime') is not None:
                self._controller._widgetsExtraParams['maxTime'].layout.display = 'flex'
            if self._controller._widgetsExtraParams.get('maxTime') is not None:
                self._controller._widgetsExtraParams['maxTime'].layout.display = 'flex'
            if self._controller._widgetsExtraParams.get('randomSeed') is not None:
                self._controller._widgetsExtraParams['randomSeed'].layout.display = 'flex'
            if self._controller._widgetsExtraParams.get('runs') is not None:
                self._controller._widgetsExtraParams['runs'].layout.display = 'flex'
            if self._controller._widgetsExtraParams.get('aggregateResults') is not None:
                self._controller._widgetsExtraParams['aggregateResults'].layout.display = 'flex'
        else:
            if self._controller._widgetsExtraParams.get('maxTime') is not None:
                self._controller._widgetsExtraParams['maxTime'].layout.display = 'none'
            if self._controller._widgetsExtraParams.get('randomSeed') is not None:
                self._controller._widgetsExtraParams['randomSeed'].layout.display = 'none'
            if self._controller._widgetsExtraParams.get('runs') is not None:
                self._controller._widgetsExtraParams['runs'].layout.display = 'none'
            if self._controller._widgetsExtraParams.get('aggregateResults') is not None:
                self._controller._widgetsExtraParams['aggregateResults'].layout.display = 'none'
            if self._controller._advancedTabWidget is not None:
                self._controller._advancedTabWidget.layout.display = 'none'

    def _update_view_specific_params(self, freeParamDict=None):
        """Getting other parameters specific to the Field view."""
        if freeParamDict is None:
            freeParamDict = {}
        if self._controller is not None:
            self._randomSeed = self._getWidgetParamValue('randomSeed', self._controller._widgetsExtraParams)  # self._fixedParams['randomSeed'] if self._fixedParams.get('randomSeed') is not None else self._controller._widgetsExtraParams['randomSeed'].value
            # self._finalViewAxes = (self._getWidgetParamValue('final_x', self._controller._widgetsPlotOnly), self._getWidgetParamValue('final_y', self._controller._widgetsPlotOnly))
            # self._finalViewAxes = (self._fixedParams['final_x'] if self._fixedParams.get('final_x') is not None else self._controller._widgetsPlotOnly['final_x'].value, self._fixedParams['final_y'] if self._fixedParams.get('final_y') is not None else self._controller._widgetsPlotOnly['final_y'].value)
            # self._plotProportions = self._getWidgetParamValue('plotProportions', self._controller._widgetsPlotOnly) # self._fixedParams['plotProportions'] if self._fixedParams.get('plotProportions') is not None else self._controller._widgetsPlotOnly['plotProportions'].value
            self._maxTime = self._getWidgetParamValue('maxTime', self._controller._widgetsExtraParams)  # self._fixedParams['maxTime'] if self._fixedParams.get('maxTime') is not None else self._controller._widgetsExtraParams['maxTime'].value
            self._runs = self._getWidgetParamValue('runs', self._controller._widgetsExtraParams)  # self._fixedParams['runs'] if self._fixedParams.get('runs') is not None else self._controller._widgetsExtraParams['runs'].value
            self._aggregateResults = self._getWidgetParamValue('aggregateResults', self._controller._widgetsExtraParams)  # self._fixedParams['aggregateResults'] if self._fixedParams.get('aggregateResults') is not None else self._controller._widgetsPlotOnly['aggregateResults'].value

    def _build_bookmark(self, includeParams: bool = True) -> str:
        log_str = "bookmark = " if not self._silent else ""
        log_str += "<modelName>." + self._generatingCommand + "('" + str(self._stateVariable1) + "', '" + str(self._stateVariable2) + "', "
        if self._stateVariable3 is not None:
            log_str += "'" + str(self._stateVariable3) + "', "
        # todo: plotting parameters are not kept, this could be solved with
        # some work on the _generatingKwargs and should be made general for all
        # views (similar to _get_bookmarks_params() )
        # if includeParams:
        #     log_str += self._get_bookmarks_params() + ", "
        # if len(self._generatingKwargs) > 0:
        #     for key in self._generatingKwargs:
        #         if key == 'xlab' or key == 'ylab' or key == 'zlab':
        #             log_str += key + " = '" + str(self._generatingKwargs[key]) + "', "
        #         else:
        #             log_str += key + " = " + str(self._generatingKwargs[key]) + ", "
        if includeParams:
            log_str += self._get_bookmarks_params()
            log_str += ", "
        log_str = log_str.replace('\\', '\\\\')
        log_str += "showNoise = " + str(self._showNoise)
        log_str += ", showFixedPoints = " + str(self._showFixedPoints)
        log_str += ", runs = " + str(self._runs)
        log_str += ", maxTime = " + str(self._maxTime)
        log_str += ", randomSeed = " + str(self._randomSeed)
        # todo: Following commented lines are ready to implement issue #95
        # if self._visualisationType == 'final':
        #     # these loops are necessary to return the latex() format of the reactant
        #     for reactant in self._mumotModel._getAllReactants()[0]:
        #         if str(reactant) == self._finalViewAxes[0]:
        #             log_str += ", final_x = '" + latex(reactant).replace('\\', '\\\\') + "'"
        #             break
        #     for reactant in self._mumotModel._getAllReactants()[0]:
        #         if str(reactant) == self._finalViewAxes[1]:
        #             log_str += ", final_y = '" + latex(reactant).replace('\\', '\\\\') + "'"
        #             break
        # log_str += ", plotProportions = " + str(self._plotProportions) todo: ready to implement issue #283
        log_str += ", aggregateResults = " + str(self._aggregateResults)
        log_str += ", silent = " + str(self._silent)
        log_str += ", bookmark = False"
        log_str += ")"
        log_str = utils._greekPrependify(log_str)
        log_str = log_str.replace('\\', '\\\\')
        log_str = log_str.replace('\\\\\\\\', '\\\\')

        return log_str

    def _plot_field(self) -> None:
        self._update_params()
        self._show_computation_start()
        if not(self._silent):  # @todo is this necessary?
            plt.figure(self._figureNum)
            plt.clf()
            self._resetErrorMessage()
        self._showErrorMessage(str(self))

        realEQsol = None
        EV = None
        Evects = None

        if self._stateVariable2 is None:
            if self._showNoise:
                print("Please note: Currently 'showNoise' only available for 2D stream and vector plots.")
            if self._showFixedPoints:
                FixedPoints = [[], []]
                realEQsol, eigList = self._get_fixedPoints1d()
                for kk in range(len(realEQsol)):
                    for val1, key2 in zip(realEQsol[kk].values(), eigList[kk].keys()):
                        FixedPoints[0].append(val1)
                        FixedPoints[1].append(key2)
            else:
                FixedPoints = None

            self._FixedPoints = FixedPoints

        elif self._stateVariable3 is None:
            if self._showFixedPoints or self._SOL_2ndOrdMomDict is not None or self._showSSANoise:
                Phi_stateVar1 = sympy.Symbol('Phi_' + str(self._stateVariable1))
                Phi_stateVar2 = sympy.Symbol('Phi_' + str(self._stateVariable2))
                eta_stateVar1 = sympy.Symbol('eta_' + str(self._stateVariable1))
                eta_stateVar2 = sympy.Symbol('eta_' + str(self._stateVariable2))
                M_2 = sympy.Function('M_2')

                systemSize = sympy.Symbol('systemSize')
                argDict_tmp = self._get_argDict()
                # Mutate key names of argDict so that they indicate concentrations via prefix Phi
                argDict = {}
                for key, val in argDict_tmp.items():
                    key_phi = sympy.Symbol(f"Phi_{key}") if key in self._mumotModel._constantReactants else key
                    argDict[key_phi] = val

                realEQsol, eigList = self._get_fixedPoints2d()

                PhiSubList = []
                for kk in range(len(realEQsol)):
                    PhiSubDict = {}
                    for solXi in realEQsol[kk]:
                        PhiSubDict[sympy.Symbol('Phi_' + str(solXi))] = realEQsol[kk][solXi]
                    PhiSubList.append(PhiSubDict)

                Evects = []
                EvectsPlot = []
                EV = []
                EVplot = []
                for kk in range(len(eigList)):
                    EVsub = []
                    EvectsSub = []
                    for key in eigList[kk]:
                        for jj in range(len(eigList[kk][key][1])):
                            EvectsSub.append(eigList[kk][key][1][jj].evalf())
                        if eigList[kk][key][0] > 1:
                            for jj in range(eigList[kk][key][0]):
                                EVsub.append(key.evalf())
                        else:
                            EVsub.append(key.evalf())

                    EV.append(EVsub)
                    Evects.append(EvectsSub)
                    if self._mumotModel._constantSystemSize:
                        if (0 <= sympy.re(realEQsol[kk][self._stateVariable1]) <= 1 and
                            0 <= sympy.re(realEQsol[kk][self._stateVariable2]) <= 1):
                            EVplot.append(EVsub)
                            EvectsPlot.append(EvectsSub)
                    else:
                        EVplot.append(EVsub)
                        EvectsPlot.append(EvectsSub)
                # self._EV = EV
                # self._realEQsol = realEQsol
                # self._Evects = Evects
            if self._SOL_2ndOrdMomDict:
                eta_cross = eta_stateVar1 * eta_stateVar2
                for key in self._SOL_2ndOrdMomDict:
                    if key == self._SOL_2ndOrdMomDict[key] or key in self._SOL_2ndOrdMomDict[key].args:
                        for key2 in self._SOL_2ndOrdMomDict:
                            if key2 != key and key2 != M_2(eta_cross):
                                self._SOL_2ndOrdMomDict[key] = self._SOL_2ndOrdMomDict[key].subs(key, self._SOL_2ndOrdMomDict[key2])

                SOL_2ndOrdMomDictList = []
                for nn in range(len(PhiSubList)):
                    SOL_2ndOrdMomDict = copy.deepcopy(self._SOL_2ndOrdMomDict)
                    for sol in SOL_2ndOrdMomDict:
                        SOL_2ndOrdMomDict[sol] = SOL_2ndOrdMomDict[sol].subs(PhiSubList[nn])
                        SOL_2ndOrdMomDict[sol] = SOL_2ndOrdMomDict[sol].subs(argDict)
                    SOL_2ndOrdMomDictList.append(SOL_2ndOrdMomDict)

                angle_ell_list = []
                projection_angle_list = []
                vec1 = sympy.Matrix([[0], [1]])
                for nn in range(len(EvectsPlot)):
                    vec2 = EvectsPlot[nn][0]
                    vec2norm = vec2.norm()
                    vec2 = vec2 / vec2norm
                    vec3 = EvectsPlot[nn][0]
                    vec3norm = vec3.norm()
                    vec3 = vec3 / vec3norm
                    if vec2norm >= vec3norm:
                        angle_ell = sympy.acos(vec1.dot(vec2) / (vec1.norm() * vec2.norm())).evalf()
                    else:
                        angle_ell = sympy.acos(vec1.dot(vec3) / (vec1.norm() * vec3.norm())).evalf()
                    angle_ell = angle_ell.evalf()
                    projection_angle_list.append(angle_ell)
                    angle_ell_deg = 180 * angle_ell / (sympy.pi).evalf()
                    angle_ell_list.append(round(angle_ell_deg, 5))
                projection_angle_list = [(abs(projection_angle_list[kk])
                                          if abs(projection_angle_list[kk]) <= sympy.N(sympy.pi / 2)
                                          else sympy.N(sympy.pi) - abs(projection_angle_list[kk]))
                                         for kk in range(len(projection_angle_list))]

            if self._showFixedPoints or self._SOL_2ndOrdMomDict is not None or self._showSSANoise:
                if self._mumotModel._constantSystemSize:
                    FixedPoints = [[PhiSubList[kk][Phi_stateVar1]
                                    for kk in range(len(PhiSubList))
                                    if (0 <= sympy.re(PhiSubList[kk][Phi_stateVar1]) <= 1 and
                                        0 <= sympy.re(PhiSubList[kk][Phi_stateVar2]) <= 1)],
                                   [PhiSubList[kk][Phi_stateVar2]
                                    for kk in range(len(PhiSubList))
                                    if (0 <= sympy.re(PhiSubList[kk][Phi_stateVar1]) <= 1 and
                                        0 <= sympy.re(PhiSubList[kk][Phi_stateVar2]) <= 1)]]
                    if self._SOL_2ndOrdMomDict:
                        Ell_width = [2.0 * sympy.re(sympy.cos(sympy.N(sympy.pi / 2) - projection_angle_list[kk]) *
                                                    sympy.sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar2**2)] / systemSize.subs(argDict)) +
                                                    sympy.sin(sympy.N(sympy.pi / 2) - projection_angle_list[kk]) *
                                                    sympy.sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar1**2)] / systemSize.subs(argDict)))
                                     for kk in range(len(SOL_2ndOrdMomDictList))
                                     if (0 <= sympy.re(PhiSubList[kk][Phi_stateVar1]) <= 1 and
                                         0 <= sympy.re(PhiSubList[kk][Phi_stateVar2]) <= 1)]
                        Ell_height = [2.0 * sympy.re(sympy.cos(projection_angle_list[kk]) *
                                                     sympy.sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar2**2)] / systemSize.subs(argDict)) +
                                                     sympy.sin(projection_angle_list[kk]) *
                                                     sympy.sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar1**2)] / systemSize.subs(argDict)))
                                      for kk in range(len(SOL_2ndOrdMomDictList))
                                      if (0 <= sympy.re(PhiSubList[kk][Phi_stateVar1]) <= 1 and
                                          0 <= sympy.re(PhiSubList[kk][Phi_stateVar2]) <= 1)]
                    # FixedPoints=[[realEQsol[kk][self._stateVariable1]
                    #               for kk in range(len(realEQsol))
                    #               if (0 <= sympy.re(realEQsol[kk][self._stateVariable1]) <= 1 and
                    #                   0 <= sympy.re(realEQsol[kk][self._stateVariable2]) <= 1)],
                    #              [realEQsol[kk][self._stateVariable2]
                    #               for kk in range(len(realEQsol))
                    #               if (0 <= sympy.re(realEQsol[kk][self._stateVariable1]) <= 1 and
                    #                   0 <= sympy.re(realEQsol[kk][self._stateVariable2]) <= 1)]]
                else:
                    FixedPoints = [[PhiSubList[kk][Phi_stateVar1]
                                    for kk in range(len(PhiSubList))],
                                   [PhiSubList[kk][Phi_stateVar2]
                                    for kk in range(len(PhiSubList))]]
                    if self._SOL_2ndOrdMomDict:
                        Ell_width = [2.0 * sympy.re(sympy.cos(sympy.N(sympy.pi / 2) - projection_angle_list[kk]) *
                                                    sympy.sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar2**2)] / systemSize.subs(argDict)) +
                                                    sympy.sin(sympy.N(sympy.pi / 2) - projection_angle_list[kk]) *
                                                    sympy.sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar1**2)] / systemSize.subs(argDict)))
                                     for kk in range(len(SOL_2ndOrdMomDictList))]
                        Ell_height = [2.0 * sympy.re(sympy.cos(projection_angle_list[kk]) *
                                                     sympy.sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar2**2)] / systemSize.subs(argDict)) +
                                                     sympy.sin(projection_angle_list[kk]) *
                                                     sympy.sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar1**2)] / systemSize.subs(argDict)))
                                      for kk in range(len(SOL_2ndOrdMomDictList))]
                    # FixedPoints=[[realEQsol[kk][self._stateVariable1] for kk in range(len(realEQsol))],
                    #              [realEQsol[kk][self._stateVariable2] for kk in range(len(realEQsol))]]
                    # Ell_width = [SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar1**2)] for kk in range(len(SOL_2ndOrdMomDictList))]
                    # Ell_height = [SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar2**2)] for kk in range(len(SOL_2ndOrdMomDictList))]

                FixedPoints.append(EVplot)
            else:
                FixedPoints = None

            self._FixedPoints = FixedPoints

            skipEllipse = False
            if self._SOL_2ndOrdMomDict:
                for kk in range(len(Ell_width)):
                    if Ell_width[kk] == sympy.nan or Ell_width[kk] == 0 or Ell_height[kk] == sympy.nan or Ell_height[kk] == 0:
                        skipEllipse = True
                        self._showErrorMessage('Noise could not be calculated analytically. ')
                        break

            if self._SOL_2ndOrdMomDict and skipEllipse is False:
                # swap width and height of ellipse if width > height
                for kk in range(len(Ell_width)):
                    # print(Ell_width[kk])
                    # print(type(Ell_width[kk]))
                    # print(Ell_height[kk])
                    # print(type(Ell_height[kk]))
                    ell_width_temp = Ell_width[kk]
                    ell_height_temp = Ell_height[kk]
                    if ell_width_temp > ell_height_temp:
                        Ell_height[kk] = ell_width_temp
                        Ell_width[kk] = ell_height_temp

                ells = [mpatch.Ellipse(xy=[self._FixedPoints[0][nn],
                                           self._FixedPoints[1][nn]],
                                       width=Ell_width[nn] / systemSize.subs(argDict),
                                       height=Ell_height[nn] / systemSize.subs(argDict),
                                       angle=round(angle_ell_list[nn], 5))
                        for nn in range(len(self._FixedPoints[0]))]
                ax = plt.gca()
                for kk in range(len(ells)):
                    ax.add_artist(ells[kk])
                    ells[kk].set_alpha(0.5)
                    if sympy.re(EVplot[kk][0]) < 0 and sympy.re(EVplot[kk][1]) < 0:
                        Fcolor = consts.LINE_COLOR_LIST[1]
                    elif sympy.re(EVplot[kk][0]) > 0 and sympy.re(EVplot[kk][1]) > 0:
                        Fcolor = consts.LINE_COLOR_LIST[2]
                    else:
                        Fcolor = consts.LINE_COLOR_LIST[0]
                    ells[kk].set_facecolor(Fcolor)
                # self._ells = ells
            else:
                if self._showNoise:
                    self._showSSANoise = True

            if self._showSSANoise:
                # print(FixedPoints)
                # print(self._stateVariable1)
                # print(self._stateVariable2)
                # print(realEQsol)
                skipList = []
                for kk in range(len(realEQsol)):
                    # print("printing ellipse for point " + str(realEQsol[kk]) )
                    # skip values out of range [0,1] and unstable equilibria
                    skip = False
                    for p in realEQsol[kk].values():
                        # if p < 0 or p > 1:
                        if p < 0:
                            skip = True
                            # print("Skipping for out range")
                            break
                        for eigenV in EV[kk]:
                            # skip if no stable fixed points detected
                            if sympy.re(eigenV) >= 0:
                                skip = True
                                # print("Skipping for positive eigenvalue")
                                break
                        if skip:
                            break
                    if skip:
                        skipList.append('skip')
                        continue
                    # Generate proper init reactant list
                    initState = copy.deepcopy(realEQsol[kk])
                    for reactant in self._mumotModel._getAllReactants()[0]:
                        if reactant not in initState.keys():
                            # initState[reactant] = 1 - sum(initState.values())
                            iniSum = 0
                            for val in initState.values():
                                iniSum += np.real(val)
                            initState[reactant] = 1 - iniSum
                    # print(initState)
                    # print(f"Using params: {self._get_params()}")
                    getParams = []
                    for set_item in self._get_params():
                        getParams.append((utils._greekPrependify(set_item[0].replace('{', '').replace('}', '')), set_item[1]))
                    SSAView = MuMoTSSAView(self._mumotModel,
                                           None,
                                           params=getParams,
                                           SSParams={'maxTime': self._maxTime,
                                                     'runs': self._runs,
                                                     'realtimePlot': False,
                                                     'plotProportions': True,
                                                     'aggregateResults': self._aggregateResults,
                                                     'visualisationType': 'final',
                                                     'final_x': utils._greekPrependify(latex(self._stateVariable1)),
                                                     'final_y': utils._greekPrependify(latex(self._stateVariable2)),
                                                     'initialState': initState,
                                                     'randomSeed': self._randomSeed},
                                           silent=True)
                    # print(SSAView._printStandaloneViewCmd())
                    SSAView._figure = self._figure
                    SSAView._computeAndPlotSimulation()

                if len(realEQsol) == len(skipList):
                    self._showErrorMessage('No stable fixed points detected. Noise could not be calculated numerically.')

        else:
            if self._showNoise:
                print("Please note: Currently 'showNoise' only available for 2D stream and vector plots.")
            if self._showFixedPoints:
                realEQsol, eigList = self._get_fixedPoints3d()
                EV = []
                EVplot = []
                for kk in range(len(eigList)):
                    EVsub = []
                    for key in eigList[kk]:
                        if eigList[kk][key][0] > 1:
                            for jj in range(eigList[kk][key][0]):
                                EVsub.append(key.evalf())
                        else:
                            EVsub.append(key.evalf())

                    EV.append(EVsub)
                    if self._mumotModel._constantSystemSize:
                        if (0 <= sympy.re(realEQsol[kk][self._stateVariable1]) <= 1 and
                            0 <= sympy.re(realEQsol[kk][self._stateVariable2]) <= 1 and
                            0 <= sympy.re(realEQsol[kk][self._stateVariable3]) <= 1):
                            EVplot.append(EVsub)
                    else:
                        EVplot.append(EVsub)
                if self._mumotModel._constantSystemSize:
                    FixedPoints = [[realEQsol[kk][self._stateVariable1]
                                    for kk in range(len(realEQsol))
                                    if (0 <= sympy.re(realEQsol[kk][self._stateVariable1]) <= 1) and
                                       (0 <= sympy.re(realEQsol[kk][self._stateVariable2]) <= 1) and
                                       (0 <= sympy.re(realEQsol[kk][self._stateVariable3]) <= 1)],
                                   [realEQsol[kk][self._stateVariable2]
                                    for kk in range(len(realEQsol))
                                    if (0 <= sympy.re(realEQsol[kk][self._stateVariable1]) <= 1) and
                                       (0 <= sympy.re(realEQsol[kk][self._stateVariable2]) <= 1) and
                                       (0 <= sympy.re(realEQsol[kk][self._stateVariable3]) <= 1)],
                                   [realEQsol[kk][self._stateVariable3]
                                    for kk in range(len(realEQsol))
                                    if (0 <= sympy.re(realEQsol[kk][self._stateVariable1]) <= 1) and
                                       (0 <= sympy.re(realEQsol[kk][self._stateVariable2]) <= 1) and
                                       (0 <= sympy.re(realEQsol[kk][self._stateVariable3]) <= 1)]]
                else:
                    FixedPoints = [[realEQsol[kk][self._stateVariable1] for kk in range(len(realEQsol))],
                                   [realEQsol[kk][self._stateVariable2] for kk in range(len(realEQsol))],
                                   [realEQsol[kk][self._stateVariable3] for kk in range(len(realEQsol))]]
                FixedPoints.append(EVplot)

                # with io.capture_output() as log:
                #     for kk in range(len(realEQsol)):
                #         print('Fixed point' + str(kk + 1) + ':', realEQsol[kk], 'with eigenvalues:', str(EV[kk]))
                # self._logs.append(log)

            else:
                FixedPoints = None
            self._FixedPoints = FixedPoints

        self._realEQsol = realEQsol
        self._EV = EV
        self._Evects = Evects

        self._show_computation_stop()

    def _get_field(self):
        """Helper for _get_field_2d() and _get_field_3d()."""
        plotLimits = self._getPlotLimits()
        # paramNames = []
        # paramValues = []
        # if self._controller is not None:
        #     for name, value in self._controller._widgetsFreeParams.items():
        #         # throw away formatting for constant reactants
        #         # name = name.replace('(','')
        #         # name = name.replace(')','')
        #         paramNames.append(name)
        #         paramValues.append(value.value)
        # if self._paramNames is not None:
        #     paramNames += map(str, self._paramNames)
        #     paramValues += self._paramValues
        # argNamesSymb = list(map(sympy.Symbol, paramNames))
        # argDict = dict(zip(argNamesSymb, paramValues))
        argDict = self._get_argDict()
        funcs = self._mumotModel._getFuncs()

        return (funcs, argDict, plotLimits)

    def _get_field1d(self, kind, meshPoints, plotLimits=1):
        """Get 1-dimensional field for plotting."""

        (funcs, argDict, plotLimits) = self._get_field()
        self._X = np.mgrid[0:plotLimits:complex(0, meshPoints)]
        if self._mumotModel._constantSystemSize:
            mask = self._mask.get((meshPoints, 2))
            if mask is None:
                mask = np.zeros(self._X.shape, dtype=bool)
                upperright = np.triu_indices(meshPoints, m=1)  # @todo: allow user to set mesh points with keyword
                mask[upperright[0]] = True
                mask = np.flipud(mask)
                self._mask[(meshPoints, 2)] = mask
        self._Xdot = funcs[self._stateVariable1](*self._mumotModel._getArgTuple1d(argDict, self._stateVariable1, self._X))
        try:
            # self._speed = np.log(self._Xdot)
            # if np.isnan(self._speed).any():
            self._speed = None
        except:
            self._speed = None
        if self._mumotModel._constantSystemSize:
            self._Xdot = np.ma.array(self._Xdot, mask=mask)

    def _get_field2d(self, kind, meshPoints, plotLimits=1):
        """Gget 2-dimensional field for plotting."""
        with io.capture_output() as log:
            self._log(kind)
            (funcs, argDict, plotLimits) = self._get_field()
            self._Y, self._X = np.mgrid[0:plotLimits:complex(0, meshPoints), 0:plotLimits:complex(0, meshPoints)]
            if self._mumotModel._constantSystemSize:
                mask = self._mask.get((meshPoints, 2))
                if mask is None:
                    mask = np.zeros(self._X.shape, dtype=bool)
                    upperright = np.triu_indices(meshPoints)  # @todo: allow user to set mesh points with keyword
                    mask[upperright] = True
                    np.fill_diagonal(mask, False)
                    mask = np.flipud(mask)
                    self._mask[(meshPoints, 2)] = mask
            self._Xdot = funcs[self._stateVariable1](*self._mumotModel._getArgTuple2d(argDict, self._stateVariable1, self._stateVariable2, self._X, self._Y))
            self._Ydot = funcs[self._stateVariable2](*self._mumotModel._getArgTuple2d(argDict, self._stateVariable1, self._stateVariable2, self._X, self._Y))
            try:
                self._speed = np.log(np.sqrt(self._Xdot ** 2 + self._Ydot ** 2))
                if np.isnan(self._speed).any():
                    self._speed = None
            except:
                # self._speed = np.ones(self._X.shape, dtype=float)
                self._speed = None
            if self._mumotModel._constantSystemSize:
                self._Xdot = np.ma.array(self._Xdot, mask=mask)
                self._Ydot = np.ma.array(self._Ydot, mask=mask)
        # if len(self._logs) > 0:
        #    self._logs.insert(0, log)
        # else:
        self._logs.append(log)

    # get 3-dimensional field for plotting
    def _get_field3d(self, kind, meshPoints, plotLimits=1):
        with io.capture_output() as log:
            self._log(kind)
            (funcs, argDict, plotLimits) = self._get_field()
            self._Z, self._Y, self._X = np.mgrid[0:plotLimits:complex(0, meshPoints),
                                                 0:plotLimits:complex(0, meshPoints),
                                                 0:plotLimits:complex(0, meshPoints)]
            if self._mumotModel._constantSystemSize:
                mask = self._mask.get((meshPoints, 3))
                if mask is None:
                    # mask = np.zeros(self._X.shape, dtype=bool)
                    # upperright = np.triu_indices(meshPoints)  # @todo: allow user to set mesh points with keyword
                    # mask[upperright] = True
                    # np.fill_diagonal(mask, False)
                    # mask = np.flipud(mask)
                    mask = self._X + self._Y + self._Z >= 1
                    # mask = mask.astype(int)
                    self._mask[(meshPoints, 3)] = mask
            self._Xdot = funcs[self._stateVariable1](*self._mumotModel._getArgTuple3d(argDict, self._stateVariable1, self._stateVariable2, self._stateVariable3, self._X, self._Y, self._Z))
            self._Ydot = funcs[self._stateVariable2](*self._mumotModel._getArgTuple3d(argDict, self._stateVariable1, self._stateVariable2, self._stateVariable3, self._X, self._Y, self._Z))
            self._Zdot = funcs[self._stateVariable3](*self._mumotModel._getArgTuple3d(argDict, self._stateVariable1, self._stateVariable2, self._stateVariable3, self._X, self._Y, self._Z))
            try:
                self._speed = np.log(np.sqrt(self._Xdot ** 2 + self._Ydot ** 2 + self._Zdot ** 2))
            except:
                self._speed = None
            # self._Xdot = self._Xdot * mask
            # self._Ydot = self._Ydot * mask
            # self._Zdot = self._Zdot * mask
            if self._mumotModel._constantSystemSize:
                self._Xdot = np.ma.array(self._Xdot, mask=mask)
                self._Ydot = np.ma.array(self._Ydot, mask=mask)
                self._Zdot = np.ma.array(self._Zdot, mask=mask)
        # if len(self._logs) > 0:
        #     self._logs.insert(0, log)
        # else:
        self._logs.append(log)

    def _appendFixedPointsToLogs(self, realEQsol, EV, Evects):
        if realEQsol is not None:
            for kk in range(len(realEQsol)):
                realEQsolStr = ""
                for key, val in realEQsol[kk].items():
                    realEQsolStr += str(key) + ' = ' + str(_roundNumLogsOut(val)) + ', '
                evalString = ""
                for val in EV[kk]:
                    evalString += str(_roundNumLogsOut(val)) + ', '
                if Evects is None:
                    print('Fixed point' + str(kk + 1) + ': ', realEQsolStr, 'with eigenvalues: ', evalString)
                else:
                    evecString = ""
                    for matrix in Evects[kk]:
                        evecStringPart = "["
                        for entry in matrix.col(0):
                            if evecStringPart == "[":
                                evecStringPart += str(_roundNumLogsOut(entry))
                            else:
                                evecStringPart += ', ' + str(_roundNumLogsOut(entry))
                        evecStringPart += "]"
                        if evecString == "":
                            evecString += evecStringPart
                        else:
                            evecString += ' and ' + evecStringPart
                    print('Fixed point' + str(kk + 1) + ': ', realEQsolStr, 'with eigenvalues: ', evalString,
                          'and eigenvectors: ', evecString)


class MuMoTvectorView(MuMoTfieldView):
    """Vector plot view on model."""

    # dictionary containing the solutions of the second order noise moments in the stationary state
    _SOL_2ndOrdMomDict = None
    # set of all reactants
    _checkReactants = None
    # set of all constant reactants to get intersection with _checkReactants
    _checkConstReactants = None

    def __init__(self, model, controller, fieldParams, SOL_2ndOrd, stateVariable1, stateVariable2, stateVariable3=None, figure=None, params=None, **kwargs):
        # if model._systemSize is None and model._constantSystemSize:
        #    self._showErrorMessage("Cannot construct field-based plot until system size is set, using substitute()")
        #    return
        # if self._SOL_2ndOrdMomDict is None:
        #    self._showErrorMessage('Noise in the system could not be calculated: \'showNoise\' automatically disabled.')
        super().__init__(model=model, controller=controller, fieldParams=fieldParams, SOL_2ndOrd=SOL_2ndOrd, stateVariable1=stateVariable1, stateVariable2=stateVariable2, stateVariable3=stateVariable3, figure=figure, params=params, **kwargs)
        self._generatingCommand = "vector"

    def _plot_field(self, _=None):

        super()._plot_field()

        if self._stateVariable3 is None:
            self._get_field2d("2d vector plot", 10)  # @todo: allow user to set mesh points with keyword
            fig_vector = plt.quiver(self._X, self._Y, self._Xdot, self._Ydot, units='width', color='black')  # @todo: define colormap by user keyword

            if self._mumotModel._constantSystemSize:
                plt.fill_between([0, 1], [1, 0], [1, 1], color='grey', alpha=0.25)
                if self._chooseXrange:
                    choose_xrange = self._chooseXrange
                else:
                    choose_xrange = [0, 1]
                if self._chooseYrange:
                    choose_yrange = self._chooseYrange
                else:
                    choose_yrange = [0, 1]
            else:
                if self._chooseXrange:
                    choose_xrange = self._chooseXrange
                else:
                    choose_xrange = [0, self._X.max()]
                if self._chooseYrange:
                    choose_yrange = self._chooseYrange
                else:
                    choose_yrange = [0, self._Y.max()]
                # plt.xlim(0,self._X.max())
                # plt.ylim(0,self._Y.max())

            _fig_formatting_2D(figure=fig_vector, xlab=self._xlab,
                               specialPoints=self._FixedPoints,
                               showFixedPoints=self._showFixedPoints,
                               ax_reformat=False, curve_replot=False,
                               ylab=self._ylab, fontsize=self._axes_font_size,
                               aspectRatioEqual=True,
                               choose_xrange=choose_xrange,
                               choose_yrange=choose_yrange)
        else:
            self._get_field3d("3d vector plot", 10)
            ax = self._figure.gca(projection='3d')
            # @todo: define colormap by user keyword; normalise off maximum value
            # in self._speed, and meshpoints?
            fig_vec3d = ax.quiver(self._X, self._Y, self._Z, self._Xdot,
                                  self._Ydot, self._Zdot, length=0.01, color='black')

            _fig_formatting_3D(figure=fig_vec3d, xlab=self._xlab,
                               ylab=self._ylab, zlab=self._zlab,
                               specialPoints=self._FixedPoints,
                               showFixedPoints=self._showFixedPoints,
                               ax_reformat=True,
                               showPlane=self._mumotModel._constantSystemSize,
                               fontsize=self._axes_font_size)

        with io.capture_output() as log:
            self._appendFixedPointsToLogs(self._realEQsol, self._EV, self._Evects)
        self._logs.append(log)


class MuMoTstreamView(MuMoTfieldView):
    """Stream plot view on model."""

    # dictionary containing the solutions of the second order noise moments in the stationary state
    _SOL_2ndOrdMomDict = None
    # set of all reactants
    _checkReactants = None
    # set of all constant reactants to get intersection with _checkReactants
    _checkConstReactants = None
    # sets time of integration for 3d streams
    maxTime = None
    # time of integration for 3d streams
    _integrationTime = None
    # set number of 3d streams
    setNumPoints = None
    # number of 3d streams
    _numPoints = None

    def __init__(self, model, controller, fieldParams, SOL_2ndOrd,
                 stateVariable1, stateVariable2, stateVariable3=None,
                 figure=None, params=None, **kwargs):
        # if model._systemSize is None and model._constantSystemSize:
        #    self._showErrorMessage("Cannot construct field-based plot until system size is set, using substitute()")
        #    return
        # if self._SOL_2ndOrdMomDict is None:
        #    self._showErrorMessage('Noise in the system could not be calculated: \'showNoise\' automatically disabled.')

        # These might be better somewhere else?
        self._numPoints = kwargs.get('setNumPoints', 30)
        self._integrationTime = kwargs.get('maxTime', 1.0)

        self._checkReactants = model._reactants
        if model._constantReactants:
            self._checkConstReactants = model._constantReactants
        else:
            self._checkConstReactants = None
        super().__init__(model=model, controller=controller,
                         fieldParams=fieldParams, SOL_2ndOrd=SOL_2ndOrd,
                         stateVariable1=stateVariable1, stateVariable2=stateVariable2,
                         stateVariable3=stateVariable3, figure=figure, params=params,
                         **kwargs)
        self._generatingCommand = "stream"

    def _plot_field(self, _=None):

        # check number of time-dependent reactants
        checkReactants = copy.deepcopy(self._checkReactants)
        if self._checkConstReactants:
            checkConstReactants = copy.deepcopy(self._checkConstReactants)
            for reactant in checkReactants:
                if reactant in checkConstReactants:
                    checkReactants.remove(reactant)
        if len(checkReactants) > 3:
            self._showErrorMessage("Not implemented: This feature is available only for systems with 1,2 or 3 time-dependent reactants!")

        super()._plot_field()

        # if model has 1 dimension
        if self._stateVariable2 is None:
            _mesh_points = 100
            self._get_field1d("1d stream plot", _mesh_points)

            # Since plot is 2d, need every element of x-data to be plotted against 0
            ys = np.zeros(self._X.shape)

            # length of negative section of stream
            _neg_length = 0
            # length of positive section of stream
            _pos_length = 0
            # point along line where sign changes
            _sign_change_point = 0.0
            # index where sign changes
            _sign_change_index = 0
            # represents whether previous point was positive or negative
            _prev_point = 0
            # has the sign of the point changed from the previous point
            _sign_change = False

            for i in range(self._X.shape[0]):
                _Xdot_temp = np.absolute(self._Xdot[i])  # used for line shading
                if self._Xdot[i] < 0:
                    # if this is the first point, set prev point to -1
                    if (_prev_point == 0):
                        _prev_point = -1
                    # if previous _Xdot value was > 0, the sign has changed
                    elif (_prev_point == 1):
                        _prev_point = -1
                        _sign_change = True
                    # plot line segment, with color based on absolute value of _Xdot
                    fig_stream_1d = plt.plot(self._X[i:i + 2], ys[i:i + 2],
                                             color=plt.cm.Reds(_Xdot_temp))

                    _neg_length += 1

                elif self._Xdot[i] >= 0:
                    # if this is the first point, set prev point to 1
                    if (_prev_point == 0):
                        _prev_point = 1
                    # if previous _Xdot value was < 0, the sign has changed
                    elif (_prev_point == -1):
                        _prev_point = 1
                        _sign_change = True
                    # plot line segment, with color based on absolute value of _Xdot
                    fig_stream_1d = plt.plot(self._X[i:i + 2], ys[i:i + 2], color=plt.cm.Blues(_Xdot_temp))
                    _pos_length += 1

                # if sign has changed, record where sign changed
                if _sign_change:
                    _sign_change_index = i
                    _sign_change_point = self._X[i]
                    _sign_change = False

            # These values are used to plot each arrows start point
            _neg_arrow_start = 0.0
            _pos_arrow_start = 0.0

            # These values are used for the color of each arrow so it matches the color of the point of line its on
            _neg_arrow_color = 0.0
            _pos_arrow_color = 0.0

            # Currently only works if there is one fixed point within the range 0 - 1
            # If _prev_point == 1, then the last point was positive and so the pos arrow goes on the right
            if (_prev_point == 1):
                _pos_arrow_start = _sign_change_point + (_pos_length / (2 * _mesh_points))
                _neg_arrow_start = (_neg_length / (2 * _mesh_points))
                _pos_arrow_color = self._Xdot[_sign_change_index + int(_pos_length / 2)]
                _neg_arrow_color = self._Xdot[int(_neg_length / 2)]
            # If _prev_point == -1, then the last point was negative and so the neg arrow goes on the right
            else:
                _neg_arrow_start = _sign_change_point + (_neg_length / (2 * _mesh_points))
                _pos_arrow_start = (_pos_length / (2 * _mesh_points))
                _neg_arrow_color = self._Xdot[_sign_change_index + int(_neg_length / 2)]
                _pos_arrow_color = self._Xdot[int(_pos_length / 2)]

            # Need to get absolute values so color value isn't negative
            _neg_arrow_color = np.absolute(_neg_arrow_color)
            _pos_arrow_color = np.absolute(_pos_arrow_color)

            # if either length segment is zero, no arrow will be plotted
            if _neg_length != 0:
                plt.arrow(_neg_arrow_start + 0.05, 0, -0.01, 0,
                          width=0.000001, head_width=0.04, head_length=0.025,
                          color=plt.cm.Reds(_neg_arrow_color))
            if _pos_length != 0:
                plt.arrow(_pos_arrow_start - 0.05, 0, 0.01, 0,
                          width=0.000001, head_width=0.04, head_length=0.025,
                          color=plt.cm.Blues(_pos_arrow_color))

            if self._chooseXrange:
                choose_xrange = self._chooseXrange
            else:
                choose_xrange = [0, 1]
            if self._chooseYrange:
                choose_yrange = self._chooseYrange
            else:
                choose_yrange = [-0.1, 0.1]

            # Format the plot
            _fig_formatting_1D(figure=fig_stream_1d,
                               choose_xrange=choose_xrange,
                               choose_yrange=choose_yrange,
                               showFixedPoints=self._showFixedPoints,
                               specialPoints=self._FixedPoints,
                               xlab=self._xlab)

        # elif model has 2 dimensions
        elif self._stateVariable3 is None:
            self._get_field2d("2d stream plot", 100)  # @todo: allow user to set mesh points with keyword

            if self._speed is not None:
                with io.capture_output() as log:  # catch warnings from streamplot
                    # @todo: define colormap by user keyword
                    fig_stream = plt.streamplot(self._X, self._Y, self._Xdot,
                                                self._Ydot, color=self._speed, cmap='gray')
                self._logs.append(log)
            else:
                # @todo: define colormap by user keyword
                fig_stream = plt.streamplot(self._X, self._Y, self._Xdot,
                                            self._Ydot, color='k')

            if self._mumotModel._constantSystemSize:
                plt.fill_between([0, 1], [1, 0], [1, 1], color='grey', alpha=0.25)
                if self._chooseXrange:
                    choose_xrange = self._chooseXrange
                else:
                    choose_xrange = [0, 1]
                if self._chooseYrange:
                    choose_yrange = self._chooseYrange
                else:
                    choose_yrange = [0, 1]
            else:
                if self._chooseXrange:
                    choose_xrange = self._chooseXrange
                else:
                    choose_xrange = [0, self._X.max()]
                if self._chooseYrange:
                    choose_yrange = self._chooseYrange
                else:
                    choose_yrange = [0, self._Y.max()]
                # plt.xlim(0,self._X.max())
                # plt.ylim(0,self._Y.max())

            _fig_formatting_2D(figure=fig_stream, xlab=self._xlab,
                               specialPoints=self._FixedPoints,
                               showFixedPoints=self._showFixedPoints,
                               ax_reformat=False, curve_replot=False,
                               ylab=self._ylab, fontsize=self._axes_font_size,
                               aspectRatioEqual=True,
                               choose_xrange=choose_xrange,
                               choose_yrange=choose_yrange)

            with io.capture_output() as log:
                self._appendFixedPointsToLogs(self._realEQsol, self._EV,
                                              self._Evects)
            self._logs.append(log)
        else:
            self._get_field3d("3d stream plot", 10)
            ax = self._figure.gca(projection='3d')

            argDict = self._get_argDict()

            # Derived model equations stored with parameter values substituted
            # in from widgets
            eqA = self._mumotModel._equations[self._stateVariable1].subs(argDict)
            eqB = self._mumotModel._equations[self._stateVariable2].subs(argDict)
            eqC = self._mumotModel._equations[self._stateVariable3].subs(argDict)

            # Model of ODEs from model equations
            # Needed for odeint() method to integrate streams from start points
            # N is needed for the equations
            def modelODEs(states, t, eqA, eqB, eqC):
                A = states[0]
                B = states[1]
                C = states[2]

                # eval is needed to run the derived equation as a mathematical
                # formula rather than a string
                dAdt = eval(str(eqA))
                dBdt = eval(str(eqB))
                dCdt = eval(str(eqC))

                return [dAdt, dBdt, dCdt]

            # Time over which streams are integrated, longer time gives a longer stream.
            t = np.linspace(0, self._integrationTime, 20)

            # This empty array will store start points of streams
            start_points = np.empty([0, 3])
            # This is used to calculate speed value at each randomly selected point.
            # Each column and row in self._X contains the same values so we only need 1.
            # Used to find corresponding speed value for each starting point.
            speed_points = self._X[0, 0, :]

            j = 0

            # Builds array start_points of starting points for streams
            # _numPoints is a keyword for the number of start points (number of streams)
            while j < self._numPoints:
                # Randomly choose points to start streams from
                p = np.random.choice(self._X[0, 0, :])
                q = np.random.choice(self._Y[0, :, 0])
                r = np.random.choice(self._Z[:, 0, 0])
                # Use selected point as start point if they meet following criteria
                # If not, then reselect start point
                # Method is ineffecient for high number of start points
                if (p + q + r <= 1):
                    temp = np.array([[p, q, r]])
                    start_points = np.concatenate((start_points, temp), axis=0)
                    j += 1

            # Plot streams from each start point
            for i in range(start_points[:, 0].shape[0]):
                x = start_points[i, 0]
                y = start_points[i, 1]
                z = start_points[i, 2]

                # Finds the speed of stream at each start point
                # Finds index of each start point in speed_points
                # Uses indexes to find corresponding speed for start point in self._speed
                speed_x = np.where(speed_points == x)[0]
                speed_y = np.where(speed_points == y)[0]
                speed_z = np.where(speed_points == z)[0]
                speed = np.absolute(self._speed[speed_x, speed_y, speed_z])[0]

                # Initial conditions for integrations
                state0 = [x, y, z]
                state = odeint(modelODEs, state0, t, args=(eqA, eqB, eqC))

                fig_stream3d = ax.plot(state[:, 0],
                                       state[:, 1],
                                       state[:, 2],
                                       color=plt.cm.Greys(speed))

                # Adds arrow halfway along the length of each stream
                # mutation_scale is the size of the arrow head
                arrow_point = int(t.shape[0] / 2)
                arrow = Arrow3D([state[:, 0][arrow_point], state[:, 0][arrow_point + 2]],
                                [state[:, 1][arrow_point], state[:, 1][arrow_point + 2]],
                                [state[:, 2][arrow_point], state[:, 2][arrow_point + 2]],
                                mutation_scale=7,
                                color=plt.cm.Greys(speed))
                ax.add_artist(arrow)

            _fig_formatting_3D(figure=fig_stream3d, xlab=self._xlab,
                               ylab=self._ylab, zlab=self._zlab,
                               specialPoints=self._FixedPoints,
                               showFixedPoints=self._showFixedPoints,
                               ax_reformat=True,
                               showPlane=self._mumotModel._constantSystemSize,
                               fontsize=self._axes_font_size)


class MuMoTbifurcationView(MuMoTview):
    """Bifurcation view on model."""

    # model for bifurcation analysis
    _pyDSmodel = None
    # critical parameter for bifurcation analysis
    _bifurcationParameter = None
    # first state variable of 2D system
    _stateVariable1 = None
    # second state variable of 2D system
    _stateVariable2 = None
    # state variable of 2D system used for bifurcation analysis, can either be _stateVariable1 or _stateVariable2
    _stateVarBif1 = None
    # state variable of 2D system used for bifurcation analysis, can either be _stateVariable1 or _stateVariable2
    _stateVarBif2 = None
    # state variable 1 for logs output
    _stateVarBif1Print = None
    # state variable 2 for logs output
    _stateVarBif2Print = None
    # generates command for bookmark functionality
    _generatingCommand = None
    # maximum number of points in one continuation calculation
    _MaxNumPoints = None
    # information about the mathematical expression displayed on vertical axis; can be 'None', '+' or '-'
    _SVoperation = None
    # initial conditions specified on corresponding sliders, will be used when calculation of fixed points fails
    _pyDSmodel_ics = None

    # Parameters for controller specific to this MuMoTbifurcationView
    _BfcParams = None
    # bifurcation parameter prepared for use in _get_argDict function in MuMoTView
    _bifurcationParameter_for_get_argDict = None
    # bifurcation parameter to be used in bookmark function
    _bifurcationParameter_for_bookmark = None
    # the system state at the start of the simulation (timestep zero)
    _initialState = None
    # list of state variables
    _stateVariableList = None

    # list of symbols protected in PyDSTool
    _pydsProtected = ['gamma', 'Gamma']
    # bifurcation parameter symbol passsed to PyDSTool
    _bifurcationParameterPyDS = None

    def _constructorSpecificParams(self, _):
        if self._controller is not None:
            self._generatingCommand = "bifurcation"

    def __init__(self, model, controller, BfcParams, bifurcationParameter,
                 stateVarExpr1, stateVarExpr2=None, figure=None, params=None,
                 **kwargs):

        self._silent = kwargs.get('silent', False)

        self._bifurcationParameter_for_get_argDict = str(parse_latex(bifurcationParameter))
        # self._bifurcationParameter_for_bookmark = utils._greekPrependify(utils._doubleUnderscorify(self._bifurcationParameter_for_get_argDict))
        self._bifurcationParameter_for_bookmark = bifurcationParameter

        super().__init__(model=model, controller=controller, figure=figure, params=params, **kwargs)

        self._axes_font_size = kwargs.get('fontsize', None)
        if '-' in str(stateVarExpr1):
            self._ylab = kwargs.get('ylab', r'$' + r'\Phi_{' + str(stateVarExpr1)[:str(stateVarExpr1).index('-')].replace('\\\\', '\\') + '}' + '-' + r'\Phi_{' + str(stateVarExpr1)[str(stateVarExpr1).index('-') + 1:].replace('\\\\', '\\') + '}' + '$')
        elif '+' in str(stateVarExpr1):
            self._ylab = kwargs.get('ylab', r'$' + r'\Phi_{' + str(stateVarExpr1)[:str(stateVarExpr1).index('-')].replace('\\\\', '\\') + '}' + '+' + r'\Phi_{' + str(stateVarExpr1)[str(stateVarExpr1).index('-') + 1:].replace('\\\\', '\\') + '}' + '$')
        else:
            self._ylab = kwargs.get('ylab', r'$' + r'\Phi_{' + str(stateVarExpr1).replace('\\\\', '\\') + '}$')
        self._xlab = kwargs.get('xlab', r'$' + bifurcationParameter + '$')

        self._MaxNumPoints = kwargs.get('contMaxNumPoints', 100)

        self._bifurcationParameter = _pydstoolify(bifurcationParameter)
        replBifParam = {}
        if self._bifurcationParameter in self._pydsProtected:
            self._bifurcationParameterPyDS = 'A' + self._bifurcationParameter
            replBifParam[self._bifurcationParameter] = self._bifurcationParameterPyDS
        else:
            self._bifurcationParameterPyDS = self._bifurcationParameter

        self._stateVarExpr1 = stateVarExpr1
        stateVarExpr1 = _pydstoolify(stateVarExpr1)

        if stateVarExpr2:
            stateVarExpr2 = _pydstoolify(stateVarExpr2)

        self._SVoperation = None
        try:
            stateVarExpr1.index('-')
            self._stateVarBif1 = stateVarExpr1[:stateVarExpr1.index('-')]
            self._stateVarBif2 = stateVarExpr1[stateVarExpr1.index('-') + 1:]
            self._SVoperation = '-'

        except ValueError:
            try:
                stateVarExpr1.index(' + ')
                self._stateVarBif1 = stateVarExpr1[:stateVarExpr1.index('+')]
                self._stateVarBif2 = stateVarExpr1[stateVarExpr1.index('+') + 1:]
                self._SVoperation = '+'
            except ValueError:
                self._stateVarBif1 = stateVarExpr1
                self._stateVarBif2 = stateVarExpr2

        # print(self._stateVarBif1)
        # print(self._stateVarBif2)

        self._BfcParams = BfcParams

        if self._controller is None:
            # storing the initial state
            self._initialState = {}
            for state, pop in BfcParams["initialState"].items():
                if isinstance(state, str):
                    self._initialState[parse_latex(state)] = pop  # convert string into SymPy symbol
                else:
                    self._initialState[state] = pop

            # storing all values of Bfc-specific parameters
            self._initBifParam = BfcParams["initBifParam"]

        else:
            # storing the initial state
            self._initialState = {}
            for state, pop in BfcParams["initialState"][0].items():
                if isinstance(state, str):
                    self._initialState[parse_latex(state)] = pop[0]  # convert string into SymPy symbol
                else:
                    self._initialState[state] = pop[0]

            # storing fixed params
            for key, value in BfcParams.items():
                if value[-1]:
                    if key == 'initialState':
                        self._fixedParams[key] = self._initialState
                    else:
                        self._fixedParams[key] = value[0]

        self._constructorSpecificParams(BfcParams)

        # self._logs.append(log)

        self._pyDSmodel = dst.args(name='MuMoT Model' + str(id(self)))
        varspecs = {}
        stateVariableList = []
        replaceSV = {}
        for reactant in self._mumotModel._reactants:
            if reactant not in self._mumotModel._constantReactants:
                stateVariableList.append(reactant)
                reactantString = _pydstoolify(reactant)
                if reactantString[0].islower() or reactantString in self._pydsProtected:
                    replaceSV[reactantString] = 'A' + reactantString
                    varspecs['A' + reactantString] = _pydstoolify(self._mumotModel._equations[reactant])
                else:
                    varspecs[reactantString] = _pydstoolify(self._mumotModel._equations[reactant])

        for key, equation in varspecs.items():
            for replKey, replVal in replaceSV.items():
                equationNew = equation.replace(replKey, replVal)
                varspecs[key] = equationNew
        for key, equation in varspecs.items():
            for replKey, replVal in replBifParam.items():
                equationNew = equation.replace(replKey, replVal)
                varspecs[key] = equationNew

        self._pyDSmodel.varspecs = varspecs

        if len(stateVariableList) > 2:
            self._showErrorMessage('Bifurcation diagrams are currently only supported for 1D and 2D systems (1 or 2 time-dependent variables in the ODE system)!')
            return None
        self._stateVariableList = stateVariableList

        self._stateVariable1 = stateVariableList[0]
        if len(stateVariableList) == 2:
            self._stateVariable2 = stateVariableList[1]

        if self._stateVarBif2 is None:
            if self._stateVariable2:
                if self._stateVarBif1 == _pydstoolify(self._stateVariable1):
                    self._stateVarBif2 = _pydstoolify(self._stateVariable2)
                elif self._stateVarBif1 == _pydstoolify(self._stateVariable2):
                    self._stateVarBif2 = _pydstoolify(self._stateVariable1)
                self._stateVarBif2Print = self._stateVarBif2
                if self._stateVarBif2[0].islower() or self._stateVarBif2 in self._pydsProtected:
                    self._stateVarBif2 = 'A' + self._stateVarBif2
        else:
            self._stateVarBif2Print = self._stateVarBif2
            if self._stateVarBif2[0].islower() or self._stateVarBif2 in self._pydsProtected:
                self._stateVarBif2 = 'A' + self._stateVarBif2
        self._stateVarBif1Print = self._stateVarBif1
        if self._stateVarBif1[0].islower() or self._stateVarBif1 in self._pydsProtected:
            self._stateVarBif1 = 'A' + self._stateVarBif1

        if not self._silent:
            self._plot_bifurcation()

    def _plot_bifurcation(self, _=None):
        self._show_computation_start()

        self._initFigure()
        self._update_params()

        with io.capture_output() as log:
            self._log("bifurcation plot")
            if self._stateVariable2:
                print(f"State variables are: {self._stateVarBif1Print} and {self._stateVarBif2Print}.")
            else:
                print("{State variable is: {self._stateVarBif1Print}.")
            print(f"The bifurcation parameter chosen is: {self._bifurcationParameter}.")
        self._logs.append(log)

        argDict = self._get_argDict()
        paramDict = {}
        replaceRates = {}
        for arg in argDict:
            if arg in self._mumotModel._rates or arg in self._mumotModel._constantReactants or arg == self._mumotModel._systemSize:
                if _pydstoolify(arg) in self._pydsProtected:
                    paramDict['A' + _pydstoolify(arg)] = argDict[arg]
                    if _pydstoolify(arg) != self._bifurcationParameter:
                        replaceRates[_pydstoolify(arg)] = 'A' + _pydstoolify(arg)
                else:
                    paramDict[_pydstoolify(arg)] = argDict[arg]

        for key, equation in self._pyDSmodel.varspecs.items():
            for replKey, replVal in replaceRates.items():
                equationNew = equation.replace(replKey, replVal)
                self._pyDSmodel.varspecs[key] = equationNew

        with io.capture_output() as log:

            self._pyDSmodel.pars = paramDict

            xdata = []  # list of arrays containing the bifurcation-parameter data for bifurcation diagram data
            ydata = []  # list of arrays containing the state variable data (either one variable, or the sum or difference of the two SVs) for bifurcation diagram data

            initDictList = []
            self._pyDSmodel_ics = {}
            for inState in self._initialState:
                if inState in self._stateVariableList:
                    self._pyDSmodel_ics[inState] = self._initialState[inState]

            # print(self._pyDSmodel_ics
            # for ic in self._pyDSmodel_ics:
            #    if 'Phi0' in _pydstoolify(ic):
            #        self._pyDSmodel_ics[_pydstoolify(ic)[_pydstoolify(ic).index('0') + 1:]] = self._pyDSmodel_ics.pop(ic)  # {'A': 0.1, 'B': 0.9 }

            if len(self._stateVariableList) == 1:
                realEQsol, eigList = self._get_fixedPoints1d()
            elif len(self._stateVariableList) == 2:
                realEQsol, eigList = self._get_fixedPoints2d()

            if realEQsol != [] and realEQsol is not None:
                for kk in range(len(realEQsol)):
                    if all(sympy.sign(sympy.re(lam)) < 0 for lam in eigList[kk]):
                        initDictList.append(realEQsol[kk])
                # self._showErrorMessage('Stationary state(s) detected and continuated.'
                #                        'Initial conditions for state variables specified on sliders in Advanced options tab were not used.'
                #                        '(Those are only used in case the calculation of fixed points fails.) ')
                print(f"{len(initDictList)} stable steady state(s) detected and continuated. "
                      'Initial conditions for state variables specified on sliders in Advanced options tab were not used. '
                      'Those are only used in case the calculation of fixed points fails.')
            else:
                initDictList.append(self._pyDSmodel_ics)
                # self._showErrorMessage('Stationary states could not be calculated;'
                #                        'used initial conditions specified on sliders in Advanced options tab instead. '
                #                        'This means only one branch was attempted to be continuated '
                #                        'and the starting point might not have been a stationary state. ')
                print('Stationary states could not be calculated; '
                      f"used initial conditions specified on sliders in Advanced options tab instead: {self._pyDSmodel_ics}."
                      'This means only one branch was continuated and the starting point might not have been a stationary state.')

            specialPoints = []  # list of special points: LP and BP
            sPoints_X = []  # bifurcation parameter
            sPoints_Y = []  # stateVarBif1
            sPoints_Labels = []
            eigenvalues = []
            sPoints_Z = []  # stateVarBif2
            k_iter_BPlabel = 0
            k_iter_LPlabel = 0

            for nn, init_dict in enumerate(initDictList):
                # Mutate key names so they are in a form that is compatible
                # with PyDSTool
                init_dict_pyds = {}
                for k, v in init_dict.items():
                    k_pyds = _pydstoolify(k)
                    if k_pyds.islower() or k_pyds in self._pydsProtected:
                        k_pyds = 'A' + k_pyds
                    init_dict_pyds[k_pyds] = v

                #for key in initDictList[nn]:
                #    old_key = key
                #    new_key = _pydstoolify(key)
                #    if new_key[0].islower() or new_key in self._pydsProtected:
                #        new_key = 'A' + new_key
                #    initDictList[nn][new_key] = initDictList[nn].pop(old_key)

                # self._pyDSmodel.ics = init_dict_pyds
                pyDSode = dst.Generator.Vode_ODEsystem(self._pyDSmodel)
                pyDSode.set(ics=init_dict_pyds)
                # pyDSode.set(pars = self._getBifParInitCondFromSlider())
                pyDSode.set(pars={self._bifurcationParameterPyDS: self._initBifParam})

                # print(self._getBifParInitCondFromSlider())
                pyDScont = dst.ContClass(pyDSode)
                EQ_iter = 1 + nn
                k_iter_BP = 1
                k_iter_LP = 1

                # 'EP-C' stands for Equilibrium Point Curve. The branch will be labelled with the string aftr name='name'.
                pyDScontArgs = dst.args(name='EQ' + str(EQ_iter), type='EP-C')
                # control parameter(s) (should be among those specified in self._pyDSmodel.pars)
                pyDScontArgs.freepars = [self._bifurcationParameterPyDS]
                # The following 3 parameters should work for most cases, as
                # there should be a step-size adaption within PyDSTool.
                pyDScontArgs.MaxNumPoints = self._MaxNumPoints
                pyDScontArgs.MaxStepSize = 1e-1
                pyDScontArgs.MinStepSize = 1e-5
                pyDScontArgs.StepSize = 2e-3
                # 'Limit Points' and 'Branch Points may be detected'
                pyDScontArgs.LocBifPoints = ['LP', 'BP']
                # to tell unstable from stable branches
                pyDScontArgs.SaveEigen = True

                pyDScont.newCurve(pyDScontArgs)

                try:
                    try:
                        pyDScont['EQ' + str(EQ_iter)].backward()
                    except:
                        self._showErrorMessage('Continuation failure (backward) on initial branch<br>')
                    try:
                        pyDScont['EQ' + str(EQ_iter)].forward()
                    except:
                        self._showErrorMessage('Continuation failure (forward) on initial branch<br>')
                except ZeroDivisionError:
                    self._show_computation_stop()
                    self._showErrorMessage('Division by zero<br>')

                # pyDScont['EQ' + str(EQ_iter)].info()
                if self._stateVarBif2 is not None:
                    try:
                        xdata.append(pyDScont['EQ' + str(EQ_iter)].sol[self._bifurcationParameterPyDS])
                        if self._SVoperation:
                            if self._SVoperation == '-':
                                ydata.append(pyDScont['EQ' + str(EQ_iter)].sol[self._stateVarBif1] -
                                             pyDScont['EQ' + str(EQ_iter)].sol[self._stateVarBif2])
                            elif self._SVoperation == '+':
                                ydata.append(pyDScont['EQ' + str(EQ_iter)].sol[self._stateVarBif1] +
                                             pyDScont['EQ' + str(EQ_iter)].sol[self._stateVarBif2])
                            else:
                                self._showErrorMessage("Only '+' and '-' are supported operations between state variables.")
                        else:
                            ydata.append(pyDScont['EQ' + str(EQ_iter)].sol[self._stateVarBif1])

                        eigenvalues.append(np.array([pyDScont['EQ' + str(EQ_iter)].sol[kk].labels['EP']['data'].evals
                                                     for kk in range(len(pyDScont['EQ' + str(EQ_iter)].sol[self._stateVarBif1]))]))

                        while pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('LP' + str(k_iter_LP)):
                            if (round(pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('LP' + str(k_iter_LP))[self._bifurcationParameterPyDS], 4) not in [round(kk, 4) for kk in sPoints_X]
                                and round(pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('LP' + str(k_iter_LP))[self._stateVarBif1], 4) not in [round(kk, 4) for kk in sPoints_Y]
                                and round(pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('LP' + str(k_iter_LP))[self._stateVarBif2], 4) not in [round(kk, 4) for kk in sPoints_Z]):
                                sPoints_X.append(pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('LP' + str(k_iter_LP))[self._bifurcationParameterPyDS])
                                sPoints_Y.append(pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('LP' + str(k_iter_LP))[self._stateVarBif1])
                                sPoints_Z.append(pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('LP' + str(k_iter_LP))[self._stateVarBif2])
                                k_iter_LPlabel += 1
                                sPoints_Labels.append('LP' + str(k_iter_LPlabel))
                            k_iter_LP += 1

                        k_iter_BPlabel_previous = k_iter_BPlabel
                        while pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('BP' + str(k_iter_BP)):
                            if (round(pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('BP' + str(k_iter_BP))[self._bifurcationParameterPyDS], 4) not in [round(kk, 4) for kk in sPoints_X]
                                and round(pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('BP' + str(k_iter_BP))[self._stateVarBif1], 4) not in [round(kk, 4) for kk in sPoints_Y]
                                and round(pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('BP' + str(k_iter_BP))[self._stateVarBif2], 4) not in [round(kk, 4) for kk in sPoints_Z]):
                                sPoints_X.append(pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('BP' + str(k_iter_BP))[self._bifurcationParameterPyDS])
                                sPoints_Y.append(pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('BP' + str(k_iter_BP))[self._stateVarBif1])
                                sPoints_Z.append(pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('BP' + str(k_iter_BP))[self._stateVarBif2])
                                k_iter_BPlabel += 1
                                sPoints_Labels.append('BP' + str(k_iter_BPlabel))
                            k_iter_BP += 1
                        for jj in range(1, k_iter_BP):
                            if 'BP' + str(jj + k_iter_BPlabel_previous) in sPoints_Labels:
                                EQ_iter_BP = jj
                                # print(EQ_iter_BP)
                                k_iter_next = 1
                                # 'EP-C' stands for Equilibrium Point Curve. The branch will be labelled with the string aftr name='name'.
                                pyDScontArgs = dst.args(name='EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP), type='EP-C')
                                # control parameter(s) (should be among those specified in self._pyDSmodel.pars)
                                pyDScontArgs.freepars = [self._bifurcationParameterPyDS]
                                # The following 3 parameters should work for most cases, as there should be a step-size adaption within PyDSTool.
                                pyDScontArgs.MaxNumPoints = self._MaxNumPoints
                                pyDScontArgs.MaxStepSize = 1e-1
                                pyDScontArgs.MinStepSize = 1e-5
                                pyDScontArgs.StepSize = 5e-3
                                # 'Limit Points' and 'Branch Points may be detected'
                                pyDScontArgs.LocBifPoints = ['LP', 'BP']
                                # To tell unstable from stable branches
                                pyDScontArgs.SaveEigen = True
                                pyDScontArgs.initpoint = 'EQ' + str(EQ_iter) + ':BP' + str(jj)
                                pyDScont.newCurve(pyDScontArgs)

                                try:
                                    try:
                                        pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].backward()
                                    except:
                                        self._showErrorMessage('Continuation failure (backward) starting from branch point<br>')
                                    try:
                                        pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].forward()
                                    except:
                                        self._showErrorMessage('Continuation failure (forward) starting from branch point<br>')
                                except ZeroDivisionError:
                                    self._show_computation_stop()
                                    self._showErrorMessage('Division by zero<br>')

                                xdata.append(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].sol[self._bifurcationParameterPyDS])
                                if self._SVoperation:
                                    if self._SVoperation == '-':
                                        ydata.append(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].sol[self._stateVarBif1] -
                                                     pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].sol[self._stateVarBif2])
                                    elif self._SVoperation == '+':
                                        ydata.append(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].sol[self._stateVarBif1] +
                                                     pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].sol[self._stateVarBif2])
                                    else:
                                        self._showErrorMessage('Only \' +\' and \'-\' are supported operations between state variables.')
                                else:
                                    ydata.append(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].sol[self._stateVarBif1])

                                eigenvalues.append(np.array([pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].sol[kk].labels['EP']['data'].evals
                                                             for kk in range(len(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].sol[self._stateVarBif1]))]))
                                while pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('BP' + str(k_iter_next)):
                                    if (round(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('BP' + str(k_iter_next))[self._bifurcationParameterPyDS], 4)
                                            not in [round(kk, 4) for kk in sPoints_X]
                                            and round(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('BP' + str(k_iter_next))[self._stateVarBif1], 4)
                                            not in [round(kk, 4) for kk in sPoints_Y]
                                            and round(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('BP' + str(k_iter_next))[self._stateVarBif2], 4)
                                            not in [round(kk, 4) for kk in sPoints_Z]):
                                        sPoints_X.append(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('BP' + str(k_iter_next))[self._bifurcationParameterPyDS])
                                        sPoints_Y.append(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('BP' + str(k_iter_next))[self._stateVarBif1])
                                        sPoints_Z.append(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('BP' + str(k_iter_next))[self._stateVarBif2])
                                        sPoints_Labels.append('EQ_BP_BP' + str(k_iter_next))
                                    k_iter_next += 1
                                k_iter_next = 1
                                while pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('LP' + str(k_iter_next)):
                                    if (round(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('LP' + str(k_iter_next))[self._bifurcationParameterPyDS], 4)
                                            not in [round(kk, 4) for kk in sPoints_X]
                                            and round(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('LP' + str(k_iter_next))[self._stateVarBif1], 4)
                                            not in [round(kk, 4) for kk in sPoints_Y]
                                            and round(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('LP' + str(k_iter_next))[self._stateVarBif2], 4)
                                            not in [round(kk, 4) for kk in sPoints_Z]):
                                        sPoints_X.append(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('LP' + str(k_iter_next))[self._bifurcationParameterPyDS])
                                        sPoints_Y.append(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('LP' + str(k_iter_next))[self._stateVarBif1])
                                        sPoints_Z.append(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('LP' + str(k_iter_next))[self._stateVarBif2])
                                        sPoints_Labels.append('EQ_BP_LP' + str(k_iter_next))
                                    k_iter_next += 1

                    except TypeError:
                        self._show_computation_stop()
                        print("Continuation failed; "
                              "try with different parameters - use sliders. "
                              "If that does not work, try changing maximum number of continuation points using the keyword 'contMaxNumPoints'. "
                              "If not set, default value is contMaxNumPoints=100.")

                # Bifurcation routine for 1D system
                else:
                    try:
                        xdata.append(pyDScont['EQ' + str(EQ_iter)].sol[self._bifurcationParameterPyDS])
                        ydata.append(pyDScont['EQ' + str(EQ_iter)].sol[self._stateVarBif1])

                        eigenvalues.append(np.array([pyDScont['EQ' + str(EQ_iter)].sol[kk].labels['EP']['data'].evals
                                                     for kk in range(len(pyDScont['EQ' + str(EQ_iter)].sol[self._stateVarBif1]))]))

                        while pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('LP' + str(k_iter_LP)):
                            if (round(pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('LP' + str(k_iter_LP))[self._bifurcationParameterPyDS], 4)
                                    not in [round(kk, 4) for kk in sPoints_X]
                                    and round(pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('LP' + str(k_iter_LP))[self._stateVarBif1], 4)
                                    not in [round(kk, 4) for kk in sPoints_Y]):
                                sPoints_X.append(pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('LP' + str(k_iter_LP))[self._bifurcationParameterPyDS])
                                sPoints_Y.append(pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('LP' + str(k_iter_LP))[self._stateVarBif1])
                                k_iter_LPlabel += 1
                                sPoints_Labels.append('LP' + str(k_iter_LPlabel))
                            k_iter_LP += 1

                        k_iter_BPlabel_previous = k_iter_BPlabel
                        while pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('BP' + str(k_iter_BP)):
                            if (round(pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('BP' + str(k_iter_BP))[self._bifurcationParameterPyDS], 4)
                                    not in [round(kk, 4) for kk in sPoints_X]
                                    and round(pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('BP' + str(k_iter_BP))[self._stateVarBif1], 4)
                                    not in [round(kk, 4) for kk in sPoints_Y]):
                                sPoints_X.append(pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('BP' + str(k_iter_BP))[self._bifurcationParameterPyDS])
                                sPoints_Y.append(pyDScont['EQ' + str(EQ_iter)].getSpecialPoint('BP' + str(k_iter_BP))[self._stateVarBif1])
                                k_iter_BPlabel += 1
                                sPoints_Labels.append('BP' + str(k_iter_BPlabel))
                            k_iter_BP += 1
                        for jj in range(1, k_iter_BP):
                            if 'BP' + str(jj + k_iter_BPlabel_previous) in sPoints_Labels:
                                EQ_iter_BP = jj
                                print(EQ_iter_BP)
                                k_iter_next = 1
                                # 'EP-C' stands for Equilibrium Point Curve. The branch will be labelled with the string aftr name='name'.
                                pyDScontArgs = dst.args(name='EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP), type='EP-C')
                                # Control parameter(s) (should be among those specified in self._pyDSmodel.pars)
                                pyDScontArgs.freepars = [self._bifurcationParameterPyDS]
                                # The following 3 parameters should work for most cases, as there should be a step-size adaption within PyDSTool.
                                pyDScontArgs.MaxNumPoints = self._MaxNumPoints
                                pyDScontArgs.MaxStepSize = 1e-1
                                pyDScontArgs.MinStepSize = 1e-5
                                pyDScontArgs.StepSize = 5e-3
                                # 'Limit Points' and 'Branch Points may be detected'
                                pyDScontArgs.LocBifPoints = ['LP', 'BP']
                                # To tell unstable from stable branches
                                pyDScontArgs.SaveEigen = True
                                pyDScontArgs.initpoint = 'EQ' + str(EQ_iter) + ':BP' + str(jj)
                                pyDScont.newCurve(pyDScontArgs)

                                try:
                                    try:
                                        pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].backward()
                                    except:
                                        self._showErrorMessage('Continuation failure (backward) starting from branch point<br>')
                                    try:
                                        pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].forward()
                                    except:
                                        self._showErrorMessage('Continuation failure (forward) starting from branch point<br>')
                                except ZeroDivisionError:
                                    self._show_computation_stop()
                                    self._showErrorMessage('Division by zero<br>')

                                xdata.append(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].sol[self._bifurcationParameterPyDS])
                                ydata.append(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].sol[self._stateVarBif1])

                                eigenvalues.append(np.array([pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].sol[kk].labels['EP']['data'].evals
                                                             for kk in range(len(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].sol[self._stateVarBif1]))]))
                                while pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('BP' + str(k_iter_next)):
                                    if (round(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('BP' + str(k_iter_next))[self._bifurcationParameterPyDS], 4)
                                            not in [round(kk, 4) for kk in sPoints_X]
                                            and round(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('BP' + str(k_iter_next))[self._stateVarBif1], 4)
                                            not in [round(kk, 4) for kk in sPoints_Y]):
                                        sPoints_X.append(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('BP' + str(k_iter_next))[self._bifurcationParameterPyDS])
                                        sPoints_Y.append(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('BP' + str(k_iter_next))[self._stateVarBif1])
                                        sPoints_Labels.append('EQ_BP_BP' + str(k_iter_next))
                                    k_iter_next += 1
                                k_iter_next = 1
                                while pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('LP' + str(k_iter_next)):
                                    if (round(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('LP' + str(k_iter_next))[self._bifurcationParameterPyDS], 4)
                                            not in [round(kk, 4) for kk in sPoints_X]
                                            and round(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('LP' + str(k_iter_next))[self._stateVarBif1], 4)
                                            not in [round(kk, 4) for kk in sPoints_Y]):
                                        sPoints_X.append(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('LP' + str(k_iter_next))[self._bifurcationParameterPyDS])
                                        sPoints_Y.append(pyDScont['EQ' + str(EQ_iter) + 'BP' + str(EQ_iter_BP)].getSpecialPoint('LP' + str(k_iter_next))[self._stateVarBif1])
                                        sPoints_Labels.append('EQ_BP_LP' + str(k_iter_next))
                                    k_iter_next += 1

                    except TypeError:
                        self._show_computation_stop()
                        print("Continuation failed; "
                              "try with different parameters - use sliders. "
                              "If that does not work, try changing maximum number of continuation points using the keyword 'contMaxNumPoints'. "
                              "If not set, default value is contMaxNumPoints=100.")

                del(pyDScontArgs)
                del(pyDScont)
                del(pyDSode)
            if self._SVoperation:
                if self._SVoperation == '-':
                    specialPoints = [sPoints_X, np.asarray(sPoints_Y) - np.asarray(sPoints_Z), sPoints_Labels]
                elif self._SVoperation == '+':
                    specialPoints = [sPoints_X, np.asarray(sPoints_Y) + np.asarray(sPoints_Z), sPoints_Labels]
                else:
                    self._showErrorMessage("Only '+' and '-' are supported operations between state variables.")
            else:
                specialPoints = [sPoints_X, np.asarray(sPoints_Y), sPoints_Labels]

            # print('Special Points on curve: ', specialPoints)
            print('Special Points on curve:')
            if len(specialPoints[0]) == 0:
                print('No special points could be detected.')
            else:
                for kk in range(len(specialPoints[0])):
                    print(specialPoints[2][kk], ': ', self._bifurcationParameter, '=', str(_roundNumLogsOut(specialPoints[0][kk])), ',',
                          self._stateVarExpr1, '=', str(_roundNumLogsOut(specialPoints[1][kk])))

            if self._chooseYrange is None:
                if self._mumotModel._systemSize:
                    self._chooseYrange = [-self._getSystemSize(), self._getSystemSize()]

            if xdata != [] and self._chooseXrange is None:
                xmaxbif = np.max([np.max(xdata[kk]) for kk in range(len(xdata))])
                self._chooseXrange = [0, xmaxbif]

            if xdata != [] and ydata != []:
                # plt.clf()
                _fig_formatting_2D(xdata=xdata,
                                   ydata=ydata,
                                   xlab=self._xlab,
                                   ylab=self._ylab,
                                   specialPoints=specialPoints,
                                   eigenvalues=eigenvalues,
                                   choose_xrange=self._chooseXrange, choose_yrange=self._chooseYrange,
                                   ax_reformat=False, curve_replot=False, fontsize=self._axes_font_size)

            else:
                self._showErrorMessage("Bifurcation diagram could not be computed. "
                                       "Try changing parameter values on the sliders.")
                return None

        self._logs.append(log)
        self._show_computation_stop()

    def _initFigure(self) -> None:
        # self._show_computation_start()
        if not self._silent:
            plt.figure(self._figureNum)
            plt.clf()
            self._resetErrorMessage()
        self._showErrorMessage(str(self))

    def _build_bookmark(self, includeParams: bool = True) -> str:
        if not self._silent:
            log_str = "bookmark = "
        else:
            log_str = ""
        log_str += "<modelName>." + self._generatingCommand + "('" + str(self._bifurcationParameter_for_bookmark) + "', '" + str(self._stateVarExpr1) + "', "
        # if self._stateVarBif2 is not None:
        #    log_str += "'" + str(self._stateVarBif2) + "', "
        if "initialState" not in self._generatingKwargs.keys():
            initState_str = {latex(state): pop for state, pop in self._initialState.items() if state not in self._mumotModel._constantReactants}
            log_str += "initialState = " + str(initState_str) + ", "
        if "initBifParam" not in self._generatingKwargs.keys():
            log_str += "initBifParam = " + str(self._initBifParam) + ", "
        if includeParams:
            log_str += self._get_bookmarks_params() + ", "
        if len(self._generatingKwargs) > 0:
            for key in self._generatingKwargs:
                if type(self._generatingKwargs[key]) == str:
                    log_str += key + " = " + "\'" + str(self._generatingKwargs[key]) + "\'" + ", "
                else:
                    log_str += key + " = " + str(self._generatingKwargs[key]) + ", "
        log_str += "bookmark = False"
        log_str += ")"
        log_str = utils._greekPrependify(log_str)
        log_str = log_str.replace('\\', '\\\\')
        log_str = log_str.replace('\\\\\\\\', '\\\\')

        return log_str

    def _update_view_specific_params(self, freeParamDict: Optional[Dict[object, object]] = None) -> None:
        """Get other parameters specific to bifurcation()"""
        if freeParamDict is None:
            freeParamDict = {}

        if self._controller is not None:
            if self._getWidgetParamValue('initialState', None) is not None:
                # self._initialState = self._getWidgetParamValue('initialState', None)
                # self._initialState = {sympy.Symbol(key): self._getWidgetParamValue('initialState', None)[key]
                #                       for key in self._getWidgetParamValue('initialState', None)}
                self._initialState = {self._safeSymbol(key): self._getWidgetParamValue('initialState', None)[key]
                                      for key in self._getWidgetParamValue('initialState', None)}
            else:
                for state in self._initialState.keys():
                    # add normal and constant reactants to the _initialState
                    self._initialState[state] = self._getInitialState(state, freeParamDict)  # freeParamDict[state] if state in self._mumotModel._constantReactants else self._controller._widgetsExtraParams['init'+str(state)].value

            self._initBifParam = self._getWidgetParamValue('initBifParam', self._controller._widgetsExtraParams)  # self._fixedParams['initBifParam'] if self._fixedParams.get('initBifParam') is not None else self._controller._widgetsExtraParams['initBifParam'].value

    def _get_argDict(self):
        """Get and return names and values from widgets, overrides method defined in parent class MuMoTview."""
        paramNames = []
        paramValues = []
        if self._controller is not None:
            for name, value in self._controller._widgetsFreeParams.items():
                # print("wdg-name: " + str(name) + " wdg-val: " + str(value.value))
                paramNames.append(name)
                paramValues.append(value.value)

            if self._controller._widgetsExtraParams and 'initBifParam' in self._controller._widgetsExtraParams:
                paramNames.append(self._bifurcationParameter_for_get_argDict)
                paramValues.append(self._controller._widgetsExtraParams['initBifParam'].value)

        if self._fixedParams and 'initBifParam' in self._fixedParams:
            paramNames.append(self._bifurcationParameter_for_get_argDict)
            paramValues.append(self._fixedParams['initBifParam'])

        if self._fixedParams is not None:
            for key, item in self._fixedParams.items():
                # print("fix-name: " + str(key) + " fix-val: " + str(item))
                paramNames.append(str(key))
                paramValues.append(item)

        argNamesSymb = list(map(sympy.Symbol, paramNames))
        argDict = dict(zip(argNamesSymb, paramValues))

        if self._mumotModel._systemSize:
            argDict[self._mumotModel._systemSize] = 1

        # @todo: is this necessary? for which view?
        systemSize = sympy.Symbol('systemSize')
        argDict[systemSize] = self._getSystemSize()

        return argDict


class MuMoTstochasticSimulationView(MuMoTview):
    """Stochastic-simulations-view view.

    For views that allow for multiple runs with different random-seeds.

    """
    # the system state at the start of the simulation (timestep zero) described as proportion of _systemSize
    _initialState = None
    # variable to link a color to each reactant
    _colors = None
    # list of colors to pass to the _fig_formatting method (that does not include constant reactants)
    _colors_list = None
    # random seed
    _randomSeed = None
    # simulation length (in the same time unit of the rates)
    _maxTime = None
    # visualisation type
    _visualisationType = None
    # reactants to display on the two axes
    _finalViewAxes = None
    # flag to plot proportions or full populations
    _plotProportions = None
    # realtimePlot flag (TRUE = the plot is updated each timestep of the simulation; FALSE = it is updated once at the end of the simulation)
    _realtimePlot = None
    # latest computed results
    _latestResults = None
    # number of runs to execute
    _runs = None
    # flag to set if the results from multimple runs must be aggregated or not
    _aggregateResults = None
    # variable to store simulation time during simulation
    _t = 0
    # variable to store the current simulation state
    _currentState = 0
    # variable to store the time evolution of the simulation
    _evo = None
    # progress bar
    _progressBar = None
    # variable that is set to False only by the multiController managing this view (when shareAxes == True and not first view to be run)
    _allowRealtimePlotting = True

    def __init__(self, model, controller, SSParams, figure=None, params=None, **kwargs):
        # Loading bar (useful to give user progress status for long executions)
        self._progressBar = widgets.FloatProgress(
            value=0,
            min=0,
            max=1,
            description='Loading:',
            bar_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            style={'description_width': 'initial'},
            orientation='horizontal'
        )
        self._silent = kwargs.get('silent', False)
        self._xlab = kwargs.get('xlab', 'time t')
        self._ylab = kwargs.get('ylab', 'reactants')
        if not self._silent:
            display(self._progressBar)

        super().__init__(model=model, controller=controller, figure=figure, params=params, **kwargs)

        with io.capture_output() as log:
            freeParamDict = self._get_argDict()
            if self._controller is None:
                # Storing the initial state
                self._initialState = {}
                for state, pop in SSParams["initialState"].items():
                    if isinstance(state, str):
                        self._initialState[parse_latex(state)] = pop  # convert string into SymPy symbol
                    else:
                        self._initialState[state] = pop
                # Add to the _initialState the constant reactants
                for constantReactant in self._mumotModel._getAllReactants()[1]:
                    self._initialState[constantReactant] = freeParamDict[constantReactant]
                # Storing all values of MA-specific parameters
                self._maxTime = SSParams["maxTime"]
                self._randomSeed = SSParams["randomSeed"]
                self._visualisationType = SSParams["visualisationType"]
                final_x = str(parse_latex(SSParams.get("final_x", latex(sorted(self._mumotModel._getAllReactants()[0], key=str)[0]))))
                # if isinstance(final_x, str): final_x = parse_latex(final_x)
                final_y = str(parse_latex(SSParams.get("final_y", latex(sorted(self._mumotModel._getAllReactants()[0], key=str)[0]))))
                # if isinstance(final_y, str): final_y = parse_latex(final_y)
                self._finalViewAxes = (final_x, final_y)
                self._plotProportions = SSParams["plotProportions"]
                self._realtimePlot = SSParams.get('realtimePlot', False)
                self._runs = SSParams.get('runs', 1)
                self._aggregateResults = SSParams.get('aggregateResults', True)

            else:
                # storing the initial state
                self._initialState = {}
                for state, pop in SSParams["initialState"][0].items():
                    if isinstance(state, str):
                        self._initialState[parse_latex(state)] = pop[0]  # convert string into SymPy symbol
                    else:
                        self._initialState[state] = pop[0]
                # add to the _initialState the constant reactants
                for constantReactant in self._mumotModel._getAllReactants()[1]:
                    self._initialState[constantReactant] = freeParamDict[constantReactant]
                # storing fixed params
                for key, value in SSParams.items():
                    if value[-1]:
                        if key == 'initialState':
                            self._fixedParams[key] = self._initialState
                        else:
                            self._fixedParams[key] = value[0]

            self._constructorSpecificParams(SSParams)

            # map colouts to each reactant
            # colors = cm.rainbow(np.linspace(0, 1, len(self._mumotModel._reactants) ))  # @UndefinedVariable
            self._colors = {}
            self._colors_list = []
            for idx, state in enumerate(sorted(self._initialState.keys(), key=str)):
                self._colors[state] = consts.LINE_COLOR_LIST[idx]
                if state not in self._mumotModel._constantReactants:
                    self._colors_list.append(consts.LINE_COLOR_LIST[idx])

        self._logs.append(log)
        if not self._silent:
            self._computeAndPlotSimulation()

    def _constructorSpecificParams(self, _) -> None:
        pass

    def _computeAndPlotSimulation(self, _=None) -> None:
        with io.capture_output() as log:
            self._show_computation_start()
            self._update_params()
            self._log("Stochastic Simulation")
            if not self._allowRealtimePlotting: self._realtimePlot = False # if realtimePlot is not allowed, then the _realtimePlot param is forced to False (regardless the widget or input value)
            # if you need to access the standalone view, you can use the
            # command self._printStandaloneViewCmd(), this is very useful for
            # developer and advanced users as indicated in issue #92

            # Clearing the plot and setting the axes
            self._initFigure()

            self._latestResults = []
            for r in range(self._runs):
                runID = f"[{r + 1}/{self._runs}] " if self._runs > 1 else ''
                self._latestResults.append(self._runSingleSimulation(self._randomSeed + r,
                                                                     runID=runID))

            # Final plot
            if not self._realtimePlot or self._aggregateResults:
                # for results in self._latestResults:
                #     self._updateSimultationFigure(results, fullPlot=True)
                self._updateSimultationFigure(self._latestResults, fullPlot=True)

            self._show_computation_stop()
        self._logs.append(log)
        if self._controller is not None:
            self._updateDownloadLink()

    def _update_view_specific_params(self, freeParamDict: Optional[Dict[object, object]] = None) -> None:
        """Get other parameters specific to SSA."""

        if freeParamDict is None:
            freeParamDict = {}
        if self._controller is not None:
            if self._getWidgetParamValue('initialState', None) is not None:
                # self._initialState = self._getWidgetParamValue('initialState', None)
                # self._initialState = {sympy.Symbol(key): self._getWidgetParamValue('initialState', None)[key]
                #                       for key in self._getWidgetParamValue('initialState', None)}
                self._initialState = {self._safeSymbol(key): self._getWidgetParamValue('initialState', None)[key]
                                      for key in self._getWidgetParamValue('initialState', None)}
            else:
                for state in self._initialState.keys():
                    # Add normal and constant reactants to the _initialState
                    self._initialState[state] = self._getInitialState(state, freeParamDict)  # freeParamDict[state] if state in self._mumotModel._constantReactants else self._controller._widgetsExtraParams['init'+str(state)].value
            self._randomSeed = self._getWidgetParamValue('randomSeed', self._controller._widgetsExtraParams)  # self._fixedParams['randomSeed'] if self._fixedParams.get('randomSeed') is not None else self._controller._widgetsExtraParams['randomSeed'].value
            self._visualisationType = self._getWidgetParamValue('visualisationType', self._controller._widgetsPlotOnly)  # self._fixedParams['visualisationType'] if self._fixedParams.get('visualisationType') is not None else self._controller._widgetsPlotOnly['visualisationType'].value
            if self._visualisationType == 'final':
                self._finalViewAxes = (self._getWidgetParamValue('final_x',
                                       self._controller._widgetsPlotOnly),
                                       self._getWidgetParamValue('final_y',
                                       self._controller._widgetsPlotOnly))
                # self._finalViewAxes = (self._fixedParams['final_x'] if self._fixedParams.get('final_x') is not None else self._controller._widgetsPlotOnly['final_x'].value, self._fixedParams['final_y'] if self._fixedParams.get('final_y') is not None else self._controller._widgetsPlotOnly['final_y'].value)
            self._plotProportions = self._getWidgetParamValue('plotProportions', self._controller._widgetsPlotOnly)  # self._fixedParams['plotProportions'] if self._fixedParams.get('plotProportions') is not None else self._controller._widgetsPlotOnly['plotProportions'].value
            self._maxTime = self._getWidgetParamValue('maxTime', self._controller._widgetsExtraParams)  # self._fixedParams['maxTime'] if self._fixedParams.get('maxTime') is not None else self._controller._widgetsExtraParams['maxTime'].value
            self._realtimePlot = self._getWidgetParamValue('realtimePlot', self._controller._widgetsExtraParams)  # self._fixedParams['realtimePlot'] if self._fixedParams.get('realtimePlot') is not None else self._controller._widgetsExtraParams['realtimePlot'].value
            self._runs = self._getWidgetParamValue('runs', self._controller._widgetsExtraParams)  # self._fixedParams['runs'] if self._fixedParams.get('runs') is not None else self._controller._widgetsExtraParams['runs'].value
            self._aggregateResults = self._getWidgetParamValue('aggregateResults', self._controller._widgetsPlotOnly)  # self._fixedParams['aggregateResults'] if self._fixedParams.get('aggregateResults') is not None else self._controller._widgetsPlotOnly['aggregateResults'].value

    def _initSingleSimulation(self) -> None:
        self._progressBar.max = self._maxTime

        # Initialise populations by multiplying proportion with _systemSize
        # currentState = copy.deepcopy(self._initialState)
        # currentState = {s:p * self._systemSize for s,p in self._initialState.items()}
        self._currentState = {}
        leftOvers = {}
        for state, prop in self._initialState.items():
            pop = prop * self._systemSize
            if (not utils._almostEqual(pop, math.floor(pop))) and (state not in self._mumotModel._constantReactants):
                leftOvers[state] = pop - math.floor(pop)
            self._currentState[state] = math.floor(pop)
        # If approximations resulted in one agent less, it is added randomly (with probability proportional to the rounding quantities)
        sumReactants = sum([self._currentState[state]
                            for state in self._currentState.keys()
                            if state not in self._mumotModel._constantReactants])
        if sumReactants < self._systemSize:
            rnd = np.random.rand() * sum(leftOvers.values())
            bottom = 0.0
            for state, prob in leftOvers.items():
                if rnd >= bottom and rnd < (bottom + prob):
                    self._currentState[state] += 1
                    break
                bottom += prob

        # Create logging structs
        self._evo = {}
        self._evo['time'] = [0]
        for state, pop in self._currentState.items():
            self._evo[state] = []
            self._evo[state].append(pop)

        # initialise time
        self._t = 0

    def _runSingleSimulation(self, randomSeed, runID=''):
        # init the random seed
        np.random.seed(randomSeed)

        self._initSingleSimulation()

        while self._t < self._maxTime:
            # Update progress bar
            self._progressBar.value = self._t
            self._progressBar.description = f"Loading {runID}{round(self._t / self._maxTime*100)}%:"

            timeInterval, self._currentState = self._simulationStep()
            # increment time
            self._t += timeInterval
            # log step
            for state, pop in self._currentState.items():
                if state in self._mumotModel._constantReactants:
                    continue
                self._evo[state].append(pop)
            self._evo['time'].append(self._t)

            # Print (self._evo)
            # Plotting each timestep
            if self._realtimePlot:
                self._updateSimultationFigure(allResults=self._latestResults,
                                              fullPlot=False,
                                              currentEvo=self._evo)

        self._progressBar.value = self._progressBar.max
        self._progressBar.description = "Completed 100%:"
        # print("Temporal evolution per state: " + str(self._evo))
        return self._evo

    def _updateSimultationFigure(self, allResults, fullPlot: bool = True, currentEvo: Optional[Dict] = None) -> None:
        if (self._visualisationType == "evo"):

            # if incremental plot is requested, but it's the first item, we operate as fullPlot (to allow legend)
            # if not fullPlot and len(currentEvo['time']) <= 2:
            #     fullPlot = True

            # If fullPlot, plot all time-evolution
            if fullPlot or len(currentEvo['time']) <= 2:
                y_max = 1.0 if self._plotProportions else self._systemSize

                # plot in aggregate mode only if there's enough data
                if self._aggregateResults and len(allResults) > 1:
                    self._initFigure()
                    steps = 10
                    timesteps = list(np.arange(0, self._maxTime, step=self._maxTime / steps))
                    # if timesteps[-1] - self._maxTime > self._maxTime / (steps * 2):
                    #     timesteps.append(self._maxTime)
                    # else:
                    #     timesteps[-1] = self._maxTime
                    if not utils._almostEqual(timesteps[-1], self._maxTime):
                        timesteps.append(self._maxTime)

                    for state in sorted(self._initialState.keys(), key=str):
                        if state == 'time' or state in self._mumotModel._constantReactants:
                            continue
                        boxesData = []
                        avgs = []
                        for timestep in timesteps:
                            boxData = []
                            for results in allResults:
                                idx = max(0, bisect.bisect_left(results['time'], timestep))
                                if self._plotProportions:
                                    boxData.append(results[state][idx] / self._systemSize)
                                else:
                                    boxData.append(results[state][idx])
                            y_max = max(y_max, max(boxData))
                            boxesData.append(boxData)
                            avgs.append(np.mean(boxData))
                            # bplot = plt.boxplot(boxData, patch_artist=True, positions=[timestep],
                            #                     manage_ticks=False, widths=self._maxTime / (steps * 3) )
                            # print(f"Plotting bxplt at positions {timestep} generated from idx = {idx}")
                        plt.plot(timesteps, avgs, color=self._colors[state])
                        bplots = plt.boxplot(boxesData, patch_artist=True, positions=timesteps,
                                             manage_ticks=False, widths=self._maxTime / (steps * 3))
                        # for patch, color in zip(bplots['boxes'], [self._colors[state]] * len(timesteps)):
                        #     patch.set_facecolor(color)
                        # bplot['boxes'].set_facecolor(self._colors[state])
                        # plt.setp(bplots['boxes'], color=self._colors[state])
                        wdt = 2
                        for box in bplots['boxes']:
                            # change outline color
                            box.set(color=self._colors[state], linewidth=wdt)
                            # box.set( color='black', linewidth=2)
                            # change fill color
                            box.set(facecolor='None')
                            # box.set( facecolor = self._colors[state] )
                        plt.setp(bplots['whiskers'], color=self._colors[state], linewidth=wdt)
                        plt.setp(bplots['caps'], color=self._colors[state], linewidth=wdt)
                        plt.setp(bplots['medians'], color=self._colors[state], linewidth=wdt)
                        # plt.setp(bplots['fliers'], color=self._colors[state], marker='o', alpha=0.5)
                        for flier in bplots['fliers']:
                            flier.set_markerfacecolor(self._colors[state])
                            flier.set_markeredgecolor("None")
                            flier.set(marker='o', alpha=0.5)

                    padding_x = self._maxTime / 20.0
                    padding_y = y_max / 20.0
                else:
                    for state in sorted(self._initialState.keys(), key=str):
                        if (state == 'time') or state in self._mumotModel._constantReactants:
                            continue
                        # xdata = []
                        # xdata.append( results['time'] )
                        for results in allResults:
                            # ydata = []
                            if self._plotProportions:
                                ydata = [y / self._systemSize for y in results[state]]
                                # ydata.append(ytmp)
                                y_max = max(y_max, max(ydata))
                            else:
                                ydata = results[state]
                                y_max = max(y_max, max(results[state]))
                            # xdata=[list(np.arange(len(list(evo.values())[0])))] * len(evo.values()), ydata=list(evo.values()), curvelab=list(evo.keys())
                            plt.plot(results['time'], ydata, color=self._colors[state], lw=2)
                    # _fig_formatting_2D(xdata=xdata, ydata=ydata, curvelab=labels, curve_replot=False,
                    #                    choose_xrange=(0, self._maxTime), choose_yrange=(0, y_max))
                    padding_x = self._maxTime / 100.0
                    padding_y = y_max / 100.0

                # plot legend
                if self._plotProportions:
                    stateNamesLabel = [r'$' + utils._doubleUnderscorify(utils._greekPrependify(str(sympy.Symbol('Phi_{' + str(state) + '}')))) + '$'
                                       for state in sorted(self._initialState.keys(), key=str)
                                       if state not in self._mumotModel._constantReactants]
                    # stateNamesLabel = [r'$'+latex(sympy.Symbol('Phi_'+str(state))) +'$'
                    #                    for state in sorted(self._initialState.keys(), key=str)
                    #                    if state not in self._mumotModel._constantReactants]
                else:
                    stateNamesLabel = [r'$' + utils._doubleUnderscorify(utils._greekPrependify(str(sympy.Symbol(str(state))))) + '$'
                                       for state in sorted(self._initialState.keys(), key=str)
                                       if state not in self._mumotModel._constantReactants]
                    # stateNamesLabel = [r'$'+latex(sympy.Symbol(str(state)))+'$'
                    #                    for state in sorted(self._initialState.keys(), key=str)
                    #                    if state not in self._mumotModel._constantReactants]
                markers = [plt.Line2D([0, 0], [0, 0], color=self._colors[state], marker='s', linestyle='', markersize=10)
                           for state in sorted(self._initialState.keys(), key=str)
                           if state not in self._mumotModel._constantReactants]
                plt.legend(markers, stateNamesLabel, loc=self._legend_loc, borderaxespad=0., numpoints=1, fontsize=self._legend_fontsize)  # bbox_to_anchor=(0.885, 1),
                xrange = (0 - padding_x, self._maxTime + padding_x) if self._chooseXrange is None else self._chooseXrange 
                yrange = (0 - padding_y, y_max + padding_y) if self._chooseYrange is None else self._chooseYrange 
                _fig_formatting_2D(figure=self._figure, xlab=self._xlab, ylab=self._ylab,
                                   choose_xrange=xrange,
                                   choose_yrange=yrange,
                                   #choose_xrange=choose_xrange,
                                   #choose_yrange=self._chooseYrange,
                                   fontsize=self._axes_font_size, 
                                   aspectRatioEqual=False, grid=True)

            if not fullPlot:  # If realtime-plot mode, draw only the last timestep rather than overlay all
                xdata = []
                ydata = []
                y_max = 1.0 if self._plotProportions else self._systemSize
                for state in sorted(self._initialState.keys(), key=str):
                    if (state == 'time') or self._mumotModel._constantReactants:
                        continue
                    xdata.append(currentEvo['time'][-2:])
                    # modify if plotProportions
                    ytmp = ([y / self._systemSize
                             for y in currentEvo[state][-2:]]
                            if self._plotProportions else currentEvo[state][-2:])
                    y_max = max(y_max, max(ytmp))
                    ydata.append(ytmp)
                xrange = (0, self._maxTime) if self._chooseXrange is None else self._chooseXrange 
                yrange = (0, y_max) if self._chooseYrange is None else self._chooseYrange 
                _fig_formatting_2D(xdata=xdata, ydata=ydata, curve_replot=False,
                                   choose_xrange=xrange,
                                   choose_yrange=yrange,
                                   aspectRatioEqual=False, LineThickness=2,
                                   fontsize=self._axes_font_size, 
                                   grid=True, line_color_list=self._colors_list)

                # y_max = 1.0 if self._plotProportions else self._systemSize
                # for state in sorted(self._initialState.keys(), key=str):
                #     if (state == 'time'):
                #         continue
                #     # modify if plotProportions
                #     ytmp = [y / self._systemSize for y in currentEvo[state][-2:] ] if self._plotProportions else currentEvo[state][-2:]

                #     y_max = max(y_max, max(ytmp))
                #     plt.plot(currentEvo['time'][-2:], ytmp, color=self._colors[state], lw=2)
                # padding_x = 0
                # padding_y = 0
                # _fig_formatting_2D(figure=self._figure, xlab="Time", ylab="Reactants",
                #                    choose_xrange=(0-padding_x, self._maxTime+padding_x),
                #                    choose_yrange=(0-padding_y, y_max+padding_y),
                #                    aspectRatioEqual=False)
        elif (self._visualisationType == "final"):
            points_x = []
            points_y = []

            if not fullPlot:  # if it's a runtime plot
                self._initFigure()  # the figure must be cleared each timestep
                for state in self._mumotModel._getAllReactants()[0]:  # the current point added to the list of points
                    if str(state) == self._finalViewAxes[0]:
                        points_x.append(currentEvo[state][-1] / self._systemSize
                                        if self._plotProportions
                                        else currentEvo[state][-1])
                        trajectory_x = ([x / self._systemSize
                                         for x in currentEvo[state]]
                                        if self._plotProportions
                                        else currentEvo[state])
                    if str(state) == self._finalViewAxes[1]:
                        points_y.append(currentEvo[state][-1] / self._systemSize
                                        if self._plotProportions
                                        else currentEvo[state][-1])
                        trajectory_y = ([y / self._systemSize
                                         for y in currentEvo[state]]
                                        if self._plotProportions
                                        else currentEvo[state])

            if self._aggregateResults and len(allResults) > 2:  # plot in aggregate mode only if there's enough data
                self._initFigure()
                samples_x = []
                samples_y = []
                for state in self._mumotModel._getAllReactants()[0]:
                    if str(state) == self._finalViewAxes[0]:
                        for results in allResults:
                            samples_x.append(results[state][-1] / self._systemSize
                                             if self._plotProportions
                                             else results[state][-1])
                    if str(state) == self._finalViewAxes[1]:
                        for results in allResults:
                            samples_y.append(results[state][-1] / self._systemSize
                                             if self._plotProportions
                                             else results[state][-1])
                samples = np.column_stack((samples_x, samples_y))
                _plot_point_cov(samples, nstd=1, alpha=0.5, color='green')
            else:
                for state in self._mumotModel._getAllReactants()[0]:
                    if str(state) == self._finalViewAxes[0]:
                        for results in allResults:
                            points_x.append(results[state][-1] / self._systemSize
                                            if self._plotProportions
                                            else results[state][-1])
                    if str(state) == self._finalViewAxes[1]:
                        for results in allResults:
                            points_y.append(results[state][-1] / self._systemSize
                                            if self._plotProportions
                                            else results[state][-1])

            # _fig_formatting_2D(xdata=[xdata], ydata=[ydata], curve_replot=False,
            #                   xlab=self._finalViewAxes[0], ylab=self._finalViewAxes[1])
            if not fullPlot:
                plt.plot(trajectory_x, trajectory_y, '-', c='0.6')
            plt.plot(points_x, points_y, 'ro')
            if self._plotProportions:
                xlab = r'$' + r'\Phi_{' + utils._doubleUnderscorify(utils._greekPrependify(str(self._finalViewAxes[0]))) + '}$'
                ylab = r'$' + r'\Phi_{' + utils._doubleUnderscorify(utils._greekPrependify(str(self._finalViewAxes[1]))) + '}$'
            else:
                xlab = r'$' + utils._doubleUnderscorify(utils._greekPrependify(str(self._finalViewAxes[0]))) + '$'
                ylab = r'$' + utils._doubleUnderscorify(utils._greekPrependify(str(self._finalViewAxes[1]))) + '$'
            _fig_formatting_2D(figure=self._figure, aspectRatioEqual=True, xlab=xlab, ylab=ylab, fontsize=self._axes_font_size,
                               choose_xrange=self._chooseXrange, choose_yrange=self._chooseYrange)
        elif self._visualisationType == "barplot":
            self._initFigure()

            finaldata = []
            colors = []
            stdev = []
            y_max = 1.0 if self._plotProportions else self._systemSize

            if fullPlot:
                for state in sorted(self._initialState.keys(), key=str):
                    if state == 'time' or self._mumotModel._constantReactants:
                        continue
                    if self._aggregateResults and len(allResults) > 0:
                        points = []
                        for results in allResults:
                            points.append(results[state][-1] / self._systemSize
                                          if self._plotProportions
                                          else results[state][-1])
                        avg = np.mean(points)
                        stdev.append(np.std(points))
                    else:
                        if allResults:
                            avg = (allResults[-1][state][-1] / self._systemSize
                                   if self._plotProportions
                                   else allResults[-1][state][-1])
                        else:
                            avg = 0
                        stdev.append(0)
                    finaldata.append(avg)
                    # labels.append(state)
                    colors.append(self._colors[state])
            else:
                for state in sorted(self._initialState.keys(), key=str):
                    if state == 'time' or self._mumotModel._constantReactants:
                        continue
                    finaldata.append(currentEvo[state][-1] / self._systemSize
                                     if self._plotProportions
                                     else currentEvo[state][-1])
                    stdev.append(0)
                    # labels.append(state)
                    colors.append(self._colors[state])

            # plt.pie(finaldata, labels=labels, autopct=utils._make_autopct(piedata),
            #         colors=colors) # shadow=True, startangle=90,
            xpos = np.arange(len(finaldata))  # the x locations for the bars
            width = 1  # the width of the bars
            plt.bar(xpos, finaldata, width, color=colors, yerr=stdev, ecolor='black')
            # set axes
            ax = plt.gca()
            ax.set_xticks(xpos)  # for matplotlib < 2 ---> ax.set_xticks(xpos - (width / 2) )
            y_max = max(y_max, max(finaldata))
            padding_y = y_max / 100.0 if self._runs <= 1 else y_max / 20.0
            # set lables
            if self._plotProportions:
                stateNamesLabel = [r'$' + utils._doubleUnderscorify(utils._greekPrependify(str(sympy.Symbol('Phi_{' + str(state) + '}')))) + '$'
                                   for state in sorted(self._initialState.keys(), key=str)
                                   if state not in self._mumotModel._constantReactants]
                # stateNamesLabel = [r'$'+latex(sympy.Symbol('Phi_'+str(state))) +'$'
                #                    for state in sorted(self._initialState.keys(), key=str)]
            else:
                stateNamesLabel = [r'$' + utils._doubleUnderscorify(utils._greekPrependify(str(sympy.Symbol(str(state))))) + '$'
                                   for state in sorted(self._initialState.keys(), key=str)
                                   if state not in self._mumotModel._constantReactants]
                # stateNamesLabel = [r'$'+latex(sympy.Symbol(str(state)))+'$'
                #                    for state in sorted(self._initialState.keys(), key=str)]
            ax.set_xticklabels(stateNamesLabel)
            _fig_formatting_2D(figure=self._figure, xlab=self._ylab, ylab="population proportion"
                               if self._plotProportions
                               else "population size", aspectRatioEqual=False, fontsize=self._axes_font_size)
            # @todo: to fix the choose_yrange of _fig_formatting_2D (issue #104)
            plt.ylim((0, y_max + padding_y))
        # update the figure
        if not self._silent or self._realtimePlot:
            self._figure.canvas.draw()

    def _redrawOnly(self, _=None):
        self._update_params()
        self._initFigure()
        self._updateSimultationFigure(self._latestResults, fullPlot=True)
        # for results in self._latestResults:
        #     self._updateSimultationFigure(results, fullPlot=True)

    def _initFigure(self):
        if not self._silent or self._realtimePlot:
            plt.figure(self._figureNum)
            plt.cla()

        if (self._visualisationType == 'evo'):
            pass
        elif (self._visualisationType == "final"):
            # plt.axes().set_aspect('equal')
            if self._plotProportions:
                plt.xlim((0, 1.0))
                plt.ylim((0, 1.0))
            else:
                plt.xlim((0, self._systemSize))
                plt.ylim((0, self._systemSize))
            # plt.axes().set_xlabel(self._finalViewAxes[0])
            # plt.axes().set_ylabel(self._finalViewAxes[1])
        elif (self._visualisationType == "barplot"):
            # plt.axes().set_aspect('equal') # for piechart
            # plt.axes().set_aspect('auto') # for barchart
            pass

    def _convertLatestDataIntoCSV(self):
        """Formatting of the latest simulation data in the CSV format"""
        csv_results = ""
        line = 'runID' + ',' + 'time'
        for state in sorted(self._initialState.keys(), key=str):
            line += ',' + str(state)
        line += '\n'
        csv_results += line
        for runID, runData in enumerate(self._latestResults):
            for t, timestep in enumerate(runData['time']):
                line = str(runID) + ',' + str(timestep)
                for state in sorted(self._initialState.keys(), key=str):
                    if state in self._mumotModel._constantReactants:
                        continue
                    line += ',' + str(runData[state][t])
                line += '\n'
                csv_results += line
        return csv_results

    def _updateDownloadLink(self):
        """Update the link with the latest results"""
        self._controller._downloadWidgetLink.value = self._controller._create_download_link(self._convertLatestDataIntoCSV(), title="Download results", filename="simulationData.txt")

    def downloadResults(self):
        """Create a download link to access the latest results."""
        return HTML(self._controller._create_download_link(self._convertLatestDataIntoCSV(), title="Download results", filename="simulationData.txt"))


class MuMoTmultiagentView(MuMoTstochasticSimulationView):
    """Agent on networks view on model."""

    # structure to store the communication network
    _graph = None
    # type of network used in the M-A simulation
    _netType = None
    # network parameter which varies with respect to each type of network
    # (for Erdos-Renyi is linking probability, for Barabasi-Albert is the number of edge per new node, for spatial is the communication range)
    _netParam = None
    # speed of the particle on one timestep (only for dynamic netType)
    _particleSpeed = None
    # corelatedness in the random walk motion (only for dynamic netType)
    _motionCorrelatedness = None
    # list of agents involved in the simulation
    _agents = None
    # list of agents' positions
    _positions = None
    _positionHistory = None
    # Arena size: width
    _arena_width = 1
    # Arena size: height
    _arena_height = 1
    # number of simulation timesteps
    _maxTimeSteps = None
    # time scaling (i.e. lenght of each timestep)
    _timestepSize = None
    # visualise the agent trace (on moving particles)
    _showTrace = None
    # visualise the agent trace (on moving particles)
    _showInteractions = None

    def _constructorSpecificParams(self, MAParams):
        if self._controller is None:
            self._timestepSize = MAParams.get('timestepSize', 1)
            self._netType = utils._decodeNetworkTypeFromString(MAParams['netType'])
            if self._netType != consts.NetworkType.FULLY_CONNECTED:
                self._netParam = MAParams['netParam']
                if self._netType == consts.NetworkType.DYNAMIC:
                    self._motionCorrelatedness = MAParams['motionCorrelatedness']
                    self._particleSpeed = MAParams['particleSpeed']
                    self._showTrace = MAParams.get('showTrace', False)
                    self._showInteractions = MAParams.get('showInteractions', False)
        else:
            # needed for bookmarking
            self._generatingCommand = "multiagent"
            # storing fixed params
            if self._fixedParams.get('netType') is not None:
                self._fixedParams['netType'] = utils._decodeNetworkTypeFromString(self._fixedParams['netType'])
            self._netType = utils._decodeNetworkTypeFromString(MAParams['netType'][0])
            self._update_net_params(False)

        self._mumotModel._getSingleAgentRules()
        # print(self._mumotModel._agentProbabilities)

        # check if any network is available or only moving particles
        onlyDynamic = False
        (_, allConstantReactants) = self._mumotModel._getAllReactants()
        for rule in self._mumotModel._rules:
            if consts.EMPTYSET_SYMBOL in rule.lhsReactants + rule.rhsReactants:
                onlyDynamic = True
                break
            for react in rule.lhsReactants + rule.rhsReactants:
                if react in allConstantReactants:
                    onlyDynamic = True
            if onlyDynamic:
                break

        if onlyDynamic:
            # if (not self._controller) and (not self._netType == consts.NetworkType.DYNAMIC): # if the user has specified the network type, we notify him/her through error-message
            #    self._errorMessage.value = "Only Moving-Particle netType is available when rules contain the emptyset."
            if not self._netType == consts.NetworkType.DYNAMIC:
                wrnMsg = "Only Moving-Particle netType is available when rules contain the emptyset or constant reactants."
                warn(wrnMsg, exceptions.MuMoTWarning)
            self._netType = consts.NetworkType.DYNAMIC
            if self._controller:  # updating value and disabling widget
                if self._controller._widgetsExtraParams.get('netType') is not None:
                    self._controller._widgetsExtraParams['netType'].value = consts.NetworkType.DYNAMIC
                    self._controller._widgetsExtraParams['netType'].disabled = True
                else:
                    self._fixedParams['netType'] = consts.NetworkType.DYNAMIC
                self._controller._update_net_params()
            else:  # this is a standalone view
                # if the assigned value of net-param is not consistent with the input, raise a WARNING and set the default value to 0.1
                if self._netParam < 0 or self._netParam > 1:
                    wrnMsg = "WARNING! net-param value " + str(self._netParam) + " is invalid for Moving-Particles. Valid range is [0,1] indicating the particles' communication range. \n"
                    self._netParam = 0.1
                    wrnMsg += "New default values is '_netParam'=" + str(self._netParam)
                    warn(wrnMsg, exceptions.MuMoTWarning)

    def _build_bookmark(self, includeParams=True) -> str:
        log_str = "bookmark = " if not self._silent else ""
        log_str += "<modelName>." + self._generatingCommand + "("
        # log_str += _find_obj_names(self._mumotModel)[0] + "." + self._generatingCommand + "("
        if includeParams:
            log_str += self._get_bookmarks_params()
            log_str += ", "
        log_str = log_str.replace('\\', '\\\\')

        init_state_str = {latex(state): pop
                          for state, pop in self._initialState.items()
                          if state not in self._mumotModel._constantReactants}
        log_str += "initialState = " + str(init_state_str)
        log_str += ", maxTime = " + str(self._maxTime)
        log_str += ", timestepSize = " + str(self._timestepSize)
        log_str += ", randomSeed = " + str(self._randomSeed)
        log_str += ", netType = '" + utils._encodeNetworkTypeToString(self._netType) + "'"
        if not self._netType == consts.NetworkType.FULLY_CONNECTED:
            log_str += ", netParam = " + str(self._netParam)
        if self._netType == consts.NetworkType.DYNAMIC:
            log_str += ", motionCorrelatedness = " + str(self._motionCorrelatedness)
            log_str += ", particleSpeed = " + str(self._particleSpeed)
            log_str += ", showTrace = " + str(self._showTrace)
            log_str += ", showInteractions = " + str(self._showInteractions)
        log_str += ", visualisationType = '" + str(self._visualisationType) + "'"
        if self._visualisationType == 'final':
            # these loops are necessary to return the latex() format of the reactant
            for reactant in self._mumotModel._getAllReactants()[0]:
                if str(reactant) == self._finalViewAxes[0]:
                    log_str += ", final_x = '" + latex(reactant).replace('\\', '\\\\') + "'"
                    break
            for reactant in self._mumotModel._getAllReactants()[0]:
                if str(reactant) == self._finalViewAxes[1]:
                    log_str += ", final_y = '" + latex(reactant).replace('\\', '\\\\') + "'"
                    break
        log_str += ", plotProportions = " + str(self._plotProportions)
        log_str += ", realtimePlot = " + str(self._realtimePlot)
        log_str += ", runs = " + str(self._runs)
        log_str += ", aggregateResults = " + str(self._aggregateResults)
        log_str += ", silent = " + str(self._silent)
        log_str += ", bookmark = False"
        # if len(self._generatingKwargs) > 0:
        #     for key in self._generatingKwargs:
        #         if type(self._generatingKwargs[key]) == str:
        #             log_str += key + " = " + "\'"+ str(self._generatingKwargs[key]) + "\'" + ", "
        #         else:
        #             log_str += key + " = " + str(self._generatingKwargs[key]) + ", "

        log_str += ")"
        return log_str

    def _printStandaloneViewCmd(self) -> None:
        MAParams = {}
        MAParams["initialState"] = {latex(state): pop
                                    for state, pop in self._initialState.items()
                                    if state not in self._mumotModel._constantReactants}
        MAParams["maxTime"] = self._maxTime
        MAParams['timestepSize'] = self._timestepSize
        MAParams["randomSeed"] = self._randomSeed
        MAParams['netType'] = utils._encodeNetworkTypeToString(self._netType)
        if self._netType != consts.NetworkType.FULLY_CONNECTED:
            MAParams['netParam'] = self._netParam
        if self._netType == consts.NetworkType.DYNAMIC:
            MAParams['motionCorrelatedness'] = self._motionCorrelatedness
            MAParams['particleSpeed'] = self._particleSpeed
            MAParams['showTrace'] = self._showTrace
            MAParams['showInteractions'] = self._showInteractions
        MAParams["visualisationType"] = self._visualisationType
        if self._visualisationType == 'final':
            # this loop is necessary to return the latex() format of the reactant
            for reactant in self._mumotModel._getAllReactants()[0]:
                if str(reactant) == self._finalViewAxes[0]:
                    MAParams['final_x'] = latex(reactant)
                if str(reactant) == self._finalViewAxes[1]:
                    MAParams['final_y'] = latex(reactant)
        MAParams["plotProportions"] = self._plotProportions
        MAParams["realtimePlot"] = self._realtimePlot
        MAParams["runs"] = self._runs
        MAParams["aggregateResults"] = self._aggregateResults
        # sortedDict = "{"
        # for key,value in sorted(MAParams.items()):
        #     sortedDict += "'" + key + "': " + str(value) + ", "
        # sortedDict += "}"
        print("mumot.MuMoTmultiagentView(<modelName>, None, " + self._get_bookmarks_params().replace('\\', '\\\\') + ", SSParams = " + str(MAParams) + " )")

    def _update_view_specific_params(self, freeParamDict=None) -> None:
        """Read the new parameters (in case they changed in the controller) specific to multiagent().

        This function should only update local parameters and not compute data.

        """
        if freeParamDict is None:
            freeParamDict = {}

        super()._update_view_specific_params(freeParamDict)
        self._adjust_barabasi_network_range()
        if self._controller is not None:
            self._netType = self._getWidgetParamValue('netType', self._controller._widgetsExtraParams)
            if self._netType != consts.NetworkType.FULLY_CONNECTED:  # this used to refer only to value in self._fixedParams; possible bug?
                self._netParam = self._getWidgetParamValue('netParam', self._controller._widgetsExtraParams)  # self._fixedParams['netParam'] if self._fixedParams.get('netParam') is not None else self._controller._widgetsExtraParams['netParam'].value
                if self._netType is None or self._netType == consts.NetworkType.DYNAMIC:  # this used to refer only to value in self._fixedParams; possible bug?
                    self._motionCorrelatedness = self._getWidgetParamValue('motionCorrelatedness', self._controller._widgetsExtraParams)  # self._fixedParams['motionCorrelatedness'] if self._fixedParams.get('motionCorrelatedness') is not None else self._controller._widgetsExtraParams['motionCorrelatedness'].value
                    self._particleSpeed = self._getWidgetParamValue('particleSpeed', self._controller._widgetsExtraParams)  # self._fixedParams['particleSpeed'] if self._fixedParams.get('particleSpeed') is not None else self._controller._widgetsExtraParams['particleSpeed'].value
                    self._showTrace = self._getWidgetParamValue('showTrace', self._controller._widgetsPlotOnly)  # self._fixedParams['showTrace'] if self._fixedParams.get('showTrace') is not None else self._controller._widgetsPlotOnly['showTrace'].value
                    self._showInteractions = self._getWidgetParamValue('showInteractions', self._controller._widgetsPlotOnly)  # self._fixedParams['showInteractions'] if self._fixedParams.get('showInteractions') is not None else self._controller._widgetsPlotOnly['showInteractions'].value
            self._timestepSize = self._getWidgetParamValue('timestepSize', self._controller._widgetsExtraParams)  # self._fixedParams['timestepSize'] if self._fixedParams.get('timestepSize') is not None else self._controller._widgetsExtraParams['timestepSize'].value

        self._computeScalingFactor()

    def _initSingleSimulation(self) -> None:
        super()._initSingleSimulation()
        # init the network
        self._initGraph()
        # init the agents
        self._initMultiagent()

    def _initFigure(self) -> None:
        super()._initFigure()
        if self._visualisationType == "graph":
            # plt.axes().set_aspect('equal')
            ax = plt.gca()
            ax.set_aspect('equal')

    # def _updateSimultationFigure(self, evo, fullPlot=True):
    def _updateSimultationFigure(self, allResults, fullPlot=True, currentEvo=None):
        if (self._visualisationType == "graph"):
            self._initFigure()
            # plt.clf()
            # plt.axes().set_aspect('equal')
            if self._netType == consts.NetworkType.DYNAMIC:
                plt.xlim((0, 1))
                plt.ylim((0, 1))
                # plt.axes().set_aspect('equal')
                # xs = [p[0] for p in positions]
                # ys = [p[1] for p in positions]
                # plt.plot(xs, ys, 'o' )
                xs = {}
                ys = {}
                for state in self._initialState.keys():
                    xs[state] = []
                    ys[state] = []
                for a in np.arange(len(self._positions)):
                    xs[self._agents[a]].append(self._positions[a][0])
                    ys[self._agents[a]].append(self._positions[a][1])

                    if self._showInteractions:
                        agent_p = [self._positions[a][0], self._positions[a][1]]
                        for n in self._getNeighbours(a, self._positions, self._netParam):
                            neigh_p = [self._positions[n][0], self._positions[n][1]]
                            jump_boudaries = False
                            if abs(agent_p[0] - neigh_p[0]) > self._netParam:
                                jump_boudaries = True
                                if agent_p[0] > neigh_p[0]:
                                    neigh_p[0] += self._arena_width
                                else:
                                    neigh_p[0] -= self._arena_width
                            if abs(agent_p[1] - neigh_p[1]) > self._netParam:
                                jump_boudaries = True
                                if agent_p[1] > neigh_p[1]:
                                    neigh_p[1] += self._arena_height
                                else:
                                    neigh_p[1] -= self._arena_height
                            plt.plot((agent_p[0], neigh_p[0]), (agent_p[1], neigh_p[1]), '-', c='orange' if jump_boudaries else 'y')
                            # plt.plot((self._positions[a][0], self._positions[n][0]),(self._positions[a][1], self._positions[n][1]), '-', c='y')

                    if self._showTrace:
                        trace_xs = []
                        trace_ys = []
                        trace_xs.append(self._positions[a][0])
                        trace_ys.append(self._positions[a][1])
                        for p in reversed(self._positionHistory[a]):
                            # check if the trace is making a jump from one side to the other of the screen
                            if abs(trace_xs[-1] - p[0]) > self._particleSpeed or abs(trace_ys[-1] - p[1]) > self._particleSpeed:
                                tmp_start = [trace_xs[-1], trace_ys[-1]]
                                if abs(trace_xs[-1] - p[0]) > self._particleSpeed:
                                    if trace_xs[-1] > p[0]:
                                        trace_xs.append(self._arena_width)
                                        tmp_start[0] = 0
                                    else:
                                        trace_xs.append(0)
                                        tmp_start[0] = self._arena_width
                                else:
                                    trace_xs.append(p[0])
                                if abs(trace_ys[-1] - p[1]) > self._particleSpeed:
                                    if trace_ys[-1] > p[1]:
                                        trace_ys.append(self._arena_height)
                                        tmp_start[1] = 0
                                    else:
                                        trace_ys.append(0)
                                        tmp_start[1] = self._arena_height
                                else:
                                    trace_ys.append(p[1])
                                plt.plot(trace_xs, trace_ys, '-', c='0.6')
                                trace_xs = []
                                trace_ys = []
                                trace_xs.append(tmp_start[0])
                                trace_ys.append(tmp_start[1])
                            trace_xs.append(p[0])
                            trace_ys.append(p[1])
                        plt.plot(trace_xs, trace_ys, '-', c='0.6')
                for state in self._initialState.keys():
                    plt.plot(xs.get(state, []), ys.get(state, []), 'o', c=self._colors[state])
            else:
                stateColors = []
                for n in self._graph.nodes():
                    stateColors.append(self._colors.get(self._agents[n], 'w'))
                nx.draw_networkx(self._graph, self._positionHistory, node_color=stateColors, with_labels=True)
                plt.axis('off')
            # plot legend
            stateNamesLabel = [r'$' + utils._doubleUnderscorify(utils._greekPrependify(str(sympy.Symbol(str(state))))) + '$' for state in sorted(self._initialState.keys(), key=str) if state not in self._mumotModel._constantReactants]
            # stateNamesLabel = [r'$' + latex(sympy.Symbol(str(state))) + '$' for state in sorted(self._initialState.keys(), key=str)]
            markers = [plt.Line2D([0, 0], [0, 0], color=self._colors[state], marker='o', linestyle='', markersize=10) for state in sorted(self._initialState.keys(), key=str)]
            plt.legend(markers, stateNamesLabel, bbox_to_anchor=(1, 1), loc=self._legend_loc, borderaxespad=0., numpoints=1, fontsize=self._legend_fontsize)

        super()._updateSimultationFigure(allResults, fullPlot, currentEvo)

    def _computeScalingFactor(self):
        # Determining the minimum speed of the process (thus the max-scaling factor)
        maxRatesAll = 0
        for reactant, reactions in self._mumotModel._agentProbabilities.items():
            if reactant == consts.EMPTYSET_SYMBOL:
                continue  # not considering the spontaneous births as limiting component for simulation step
            sumRates = 0
            for reaction in reactions:
                sumRates += self._ratesDict[str(reaction[1])]
            # print("self._ratesDict " + str(self._ratesDict) )
            # print("reactant " + str(reactant) + " has sum rates: " + str(sumRates))
            if sumRates > maxRatesAll:
                maxRatesAll = sumRates

        if maxRatesAll > 0:
            maxTimestepSize = 1 / maxRatesAll
        else:
            maxTimestepSize = 1
        # if the timestep size is too small (and generated a too large number of timesteps, it returns an error!)
        if math.ceil(self._maxTime / maxTimestepSize) > 10000000:
            errorMsg = "ERROR! Invalid rate values. The current rates limit the agent timestep to be too small and would correspond to more than 10 milions simulation timesteps.\n"\
                       "Please modify the free parameters value to allow quicker simulations."
            self._showErrorMessage(errorMsg)
            raise exceptions.MuMoTValueError(errorMsg)
        if self._timestepSize > maxTimestepSize:
            self._timestepSize = maxTimestepSize
        self._maxTimeSteps = math.ceil(self._maxTime / self._timestepSize)
        if self._controller is not None and self._controller._widgetsExtraParams.get('timestepSize'):
            self._update_timestepSize_widget(self._timestepSize, maxTimestepSize, self._maxTimeSteps)
        else:
            if self._fixedParams.get('timestepSize') != self._timestepSize:
                self._showErrorMessage(f"Time step size was fixed to {self._fixedParams.get('timestepSize')} but needs to be updated to {self._timestepSize}")
                self._fixedParams['timestepSize'] = self._timestepSize

    def _update_timestepSize_widget(self, timestepSize, maxTimestepSize, maxTimeSteps):
        if not self._controller._widgetsExtraParams['timestepSize'].value == timestepSize:
            if self._controller._replotFunction:
                self._controller._widgetsExtraParams['timestepSize'].unobserve(self._controller._replotFunction, 'value')
            if (self._controller._widgetsExtraParams['timestepSize'].max < timestepSize):
                self._controller._widgetsExtraParams['timestepSize'].max = maxTimestepSize
            if (self._controller._widgetsExtraParams['timestepSize'].min > timestepSize):
                self._controller._widgetsExtraParams['timestepSize'].min = timestepSize
            self._controller._widgetsExtraParams['timestepSize'].value = timestepSize
            if self._controller._replotFunction:
                self._controller._widgetsExtraParams['timestepSize'].observe(self._controller._replotFunction, 'value')
        if not self._controller._widgetsExtraParams['timestepSize'].max == maxTimestepSize:
            self._controller._widgetsExtraParams['timestepSize'].max = maxTimestepSize
            self._controller._widgetsExtraParams['timestepSize'].min = min(maxTimestepSize / 100, timestepSize)
            self._controller._widgetsExtraParams['timestepSize'].step = self._controller._widgetsExtraParams['timestepSize'].min
            self._controller._widgetsExtraParams['timestepSize'].readout_format = '.' + str(utils._count_sig_decimals(str(self._controller._widgetsExtraParams['timestepSize'].step))) + 'f'
        if self._controller._widgetsExtraParams.get('maxTime'):
            self._controller._widgetsExtraParams['maxTime'].description = f"Simulation time (equivalent to {maxTimeSteps} simulation timesteps)"
            self._controller._widgetsExtraParams['maxTime'].layout = widgets.Layout(width='70%')
        else:
            self._controller._widgetsExtraParams['timestepSize'].description = f"Timestep size (total time is {self._fixedParams['maxTime']} = {maxTimeSteps} timesteps)"
            self._controller._widgetsExtraParams['timestepSize'].layout = widgets.Layout(width='70%')

    def _initGraph(self):
        numNodes = sum(self._currentState.values())
        if (self._netType == consts.NetworkType.FULLY_CONNECTED):
            # print("Generating full graph")
            self._graph = nx.complete_graph(numNodes)  # np.repeat(0, self.numNodes)
        elif (self._netType == consts.NetworkType.ERSOS_RENYI):
            # print("Generating Erdos-Renyi graph (connected)")
            if self._netParam is not None and self._netParam > 0 and self._netParam <= 1:
                self._graph = nx.erdos_renyi_graph(numNodes, self._netParam, np.random.randint(consts.MAX_RANDOM_SEED))
                i = 0
                while (not nx.is_connected(self._graph)):
                    if i > 100000:
                        errorMsg = (f"ERROR! Invalid network parameter (link probability={self._netParam} for E-R networks."
                                    f"After {i} attempts of network initialisation, the network is never connected.\n"
                                    "Please increase the network parameter value.")
                        print(errorMsg)
                        raise exceptions.MuMoTValueError(errorMsg)
                    # print("Graph was not connected; Resampling!")
                    i = i + 1
                    self._graph = nx.erdos_renyi_graph(numNodes, self._netParam, np.random.randint(consts.MAX_RANDOM_SEED))
            else:
                errorMsg = ("ERROR! Invalid network parameter (link probability) for E-R networks. "
                            f"It must be between 0 and 1; input is {self._netParam}")
                raise exceptions.MuMoTValueError(errorMsg)
        elif (self._netType == consts.NetworkType.BARABASI_ALBERT):
            # print("Generating Barabasi-Albert graph")
            netParam = int(self._netParam)
            if netParam is not None and netParam > 0 and netParam <= numNodes:
                self._graph = nx.barabasi_albert_graph(numNodes, netParam, np.random.randint(consts.MAX_RANDOM_SEED))
            else:
                errorMsg = ("ERROR! Invalid network parameter (number of edges per new node) for B-A networks."
                            f"It must be an integer between 1 and {numNodes}; input is {self._netParam}")
                raise exceptions.MuMoTValueError(errorMsg)
        elif (self._netType == consts.NetworkType.SPACE):
            # @todo: implement network generate by placing points (with local communication range) randomly in 2D space
            errorMsg = "ERROR: Graphs of type SPACE are not implemented yet."
            raise exceptions.MuMoTValueError(errorMsg)
        elif (self._netType == consts.NetworkType.DYNAMIC):
            self._positions = []
            for _ in range(numNodes):
                x = np.random.rand() * self._arena_width
                y = np.random.rand() * self._arena_height
                o = np.random.rand() * np.pi * 2.0
                self._positions.append((x, y, o))
            return

    def _initMultiagent(self):
        # init the agents list
        self._agents = []
        for state, pop in self._currentState.items():
            self._agents.extend([state] * pop)
        self._agents = np.random.permutation(self._agents).tolist()  # random shuffling of elements (useful to avoid initial clusters in networks)

        # init the positionHistory lists
        dynamicNetwork = self._netType == consts.NetworkType.DYNAMIC
        if dynamicNetwork:
            self._positionHistory = []
            for _ in np.arange(sum(self._currentState.values())):
                self._positionHistory.append([])
        else:  # store the graph layout (only for 'graph' visualisation)
            self._positionHistory = nx.circular_layout(self._graph)

    def _simulationStep(self):
        tmp_agents = copy.deepcopy(self._agents)
        dynamic = self._netType == consts.NetworkType.DYNAMIC
        if dynamic:
            tmp_positions = copy.deepcopy(self._positions)
            communication_range = self._netParam
            # store the position history
            for idx, _ in enumerate(self._agents):  # second element _ is the agent (unused)
                self._positionHistory[idx].append(self._positions[idx])
        children = []
        activeAgents = [True] * len(self._agents)
        # for idx, a in enumerate(self._agents):
        # to execute in random order the agents I just create a shuffled list of idx and I follow that
        indexes = np.arange(0, len(self._agents))
        indexes = np.random.permutation(indexes).tolist()  # shuffle the indexes
        for idx in indexes:
            a = self._agents[idx]
            # if moving-particles the agent moves
            if dynamic:
                # print("Agent " + str(idx) + " moved from " + str(self._positions[idx]) )
                self._positions[idx] = self._updatePosition(self._positions[idx][0], self._positions[idx][1], self._positions[idx][2], self._particleSpeed, self._motionCorrelatedness)
                # print("to position " + str(self._positions[idx]) )

            # the step is executed only if the agent is active
            if not activeAgents[idx]:
                continue

            # computing the list of neighbours for the given agent
            if dynamic:
                neighNodes = self._getNeighbours(idx, tmp_positions, communication_range)
            else:
                neighNodes = list(nx.all_neighbors(self._graph, idx))
            neighNodes = np.random.permutation(neighNodes).tolist()  # random shuffling of neighNodes (to randomise interactions)
            neighAgents = [tmp_agents[x] for x in neighNodes]  # creating the list of neighbours' states
            neighActive = [activeAgents[x] for x in neighNodes]  # creating the list of neighbour' activity-status

            # print("Neighs of agent " + str(idx) + " are " + str(neighNodes) + " with states " + str(neighAgents) )
            # run one simulation step for agent a
            oneStepOutput = self._stepOneAgent(a, neighAgents, neighActive)
            self._agents[idx] = oneStepOutput[0][0]
            # check for new particles generated in the step
            if len(oneStepOutput[0]) > 1:  # new particles must be created
                for particle in oneStepOutput[0][1:]:
                    children.append((particle, tmp_positions[idx]))
            for idx_c, neighChange in enumerate(oneStepOutput[1]):
                if neighChange:
                    activeAgents[neighNodes[idx_c]] = False
                    self._agents[neighNodes[idx_c]] = neighChange

        # add the new agents coming from splitting (possible only for moving-particles view)
        for child in children:
            self._agents.append(child[0])
            self._positions.append(child[1])
            self._positionHistory.append([])
            idx = len(self._positions) - 1
            self._positionHistory[idx].append(self._positions[idx])
            self._positions[idx] = (self._positions[idx][0], self._positions[idx][1], np.random.rand() * np.pi * 2.0)  # set random orientation
            # self._positions[idx][2] = np.random.rand() * np.pi * 2.0 # set random orientation
            self._positions[idx] = self._updatePosition(self._positions[idx][0], self._positions[idx][1], self._positions[idx][2], self._particleSpeed, self._motionCorrelatedness)

        # compute self birth (possible only for moving-particles view)
        for birth in self._mumotModel._agentProbabilities[consts.EMPTYSET_SYMBOL]:
            birthRate = self._ratesDict[str(birth[1])] * self._timestepSize  # scale the rate
            decimal = birthRate % 1
            birthsNum = int(birthRate - decimal)
            np.random.rand()
            if (np.random.rand() < decimal):
                birthsNum += 1
            # print ( "Birth rate " + str(birth[1]) + " triggers " + str(birthsNum) + " newborns")
            for _ in range(birthsNum):
                for newborn in birth[2]:
                    self._agents.append(newborn)
                    self._positions.append((np.random.rand() * self._arena_width, np.random.rand() * self._arena_height, np.random.rand() * np.pi * 2.0))
                    self._positionHistory.append([])
                    self._positionHistory[len(self._positions) - 1].append(self._positions[len(self._positions) - 1])

        # Remove from lists (_agents, _positions, and _positionHistory) the 'dead' agents (possible only for moving-particles view)
        deads = [idx for idx, a in enumerate(self._agents) if a == consts.EMPTYSET_SYMBOL]
        # print("Dead list is " + str(deads))
        for dead in reversed(deads):
            del self._agents[dead]
            del self._positions[dead]
            del self._positionHistory[dead]

        currentState = {state: self._agents.count(state) for state in self._initialState.keys()}  # if state not in self._mumotModel._constantReactants} # self._mumotModel._reactants | self._mumotModel._constantReactants}
        return (self._timestepSize, currentState)

    def _stepOneAgent(self, agent, neighs, activeNeighs):
        """One timestep for one agent."""
        rnd = np.random.rand()
        lastVal = 0
        neighChanges = [None] * len(neighs)
        # counting how many neighbours for each state (to be uses for the interaction probabilities)
        neighCount = {x: neighs.count(x) for x in self._initialState.keys()}  # self._mumotModel._reactants | self._mumotModel._constantReactants}
        for idx, neigh in enumerate(neighs):
            if not activeNeighs[idx]:
                neighCount[neigh] -= 1
        # print("Agent " + str(agent) + " with probSet=" + str(probSets))
        # print("nc:" + str(neighCount))
        for reaction in self._mumotModel._agentProbabilities[agent]:
            popScaling = 1
            rate = self._ratesDict[str(reaction[1])] * self._timestepSize  # scaling the rate by the timeStep size
            if len(neighs) >= len(reaction[0]):
                j = 0
                for reagent in reaction[0]:
                    popScaling *= (neighCount[reagent] / (len(neighs) - j)
                                   if neighCount[reagent] >= reaction[0].count(reagent)
                                   else 0)
                    j += 1
            else:
                popScaling = 0
            val = popScaling * rate
            # print(f"For reaction: {agent}+{reaction[0]} the popScaling is {popScaling}")
            if (rnd < val + lastVal):
                # A state change happened!
                # print(f"Reaction: {reaction[1]} by agent {agent} with agent(s) {reaction[0]} becomes {reaction[2]} &others: {reaction[3]}")
                # print("Val was: {val} lastVal: {lastVal} and rand: {rnd}")

                # Locking the other reagents involved in the reaction
                for idx_r, reagent in enumerate(reaction[0]):
                    for idx_n, neigh in enumerate(neighs):
                        if neigh == reagent and activeNeighs[idx_n] and neighChanges[idx_n] is None:
                            neighChanges[idx_n] = reaction[3][idx_r]
                            break

                return (reaction[2], neighChanges)
            else:
                lastVal += val
        # No state change happened
        return ([agent], neighChanges)

    def _updatePosition(self, x, y, o, speed, correlatedness):
        # random component
        rand_o = np.random.rand() * np.pi * 2.0
        rand_x = speed * np.cos(rand_o) * (1 - correlatedness)
        rand_y = speed * np.sin(rand_o) * (1 - correlatedness)
        # persistance component
        corr_x = speed * np.cos(o) * correlatedness
        corr_y = speed * np.sin(o) * correlatedness
        # movement
        move_x = rand_x + corr_x
        move_y = rand_y + corr_y
        # new orientation
        o = np.arctan2(move_y, move_x)
        # new position
        x = x + move_x
        y = y + move_y

        # Implement the periodic boundary conditions
        x = x % self._arena_width
        y = y % self._arena_height
        # CODE FOR A BOUNDED ARENA
        # if a.position.x < 0:
        #     a.position.x = 0
        # elif a.position.x > self.dimensions.x:
        #     a.position.x = self.dimensions.x
        # if a.position.y < 0:
        #     a.position.y = 0
        # elif a.position.y > self.dimensions.y:
        #     a.position.x = self.dimensions.x
        return (x, y, o)

    def _getNeighbours(self, agent, positions, distance_range):
        """Return the (index) list of neighbours of ``agent``."""
        neighbour_list = []
        for neigh in np.arange(len(positions)):
            if (not neigh == agent) and (self._distance_on_torus(positions[agent][0], positions[agent][1], positions[neigh][0], positions[neigh][1]) < distance_range):
                neighbour_list.append(neigh)
        return neighbour_list

    def _distance_on_torus(self, x_1, y_1, x_2, y_2):
        """Returns the minimum distance calculated on the torus given by periodic boundary conditions."""
        return np.sqrt(min(abs(x_1 - x_2), self._arena_width - abs(x_1 - x_2))**2 +
                       min(abs(y_1 - y_2), self._arena_height - abs(y_1 - y_2))**2)

    def _update_net_params(self, resetValueAndRange):
        """Update the widgets related to the netType

        (it cannot be a :class:`MuMoTcontroller` method because with multi-controller it needs to point to the right ``_controller``)
        """
        # if netType is fixed, no update is necessary
        if self._fixedParams.get('netParam') is not None:
            return
        self._netType = (self._fixedParams['netType']
                         if self._fixedParams.get('netType') is not None
                         else self._controller._widgetsExtraParams['netType'].value)

        # oder of assignment is important (first, update the min and max, later, the value)
        toLinkPlotFunction = False
        if self._controller._replotFunction:
            try:
                self._controller._widgetsExtraParams['netParam'].unobserve(self._controller._replotFunction, 'value')
                toLinkPlotFunction = True
            except ValueError:
                pass
        if resetValueAndRange:
            self._controller._widgetsExtraParams['netParam'].max = float("inf")  # temp to avoid min > max exception
        if (self._netType == consts.NetworkType.FULLY_CONNECTED):
            # self._controller._widgetsExtraParams['netParam'].min = 0
            # self._controller._widgetsExtraParams['netParam'].max = 1
            # self._controller._widgetsExtraParams['netParam'].step = 1
            # self._controller._widgetsExtraParams['netParam'].value = 0
            # self._controller._widgetsExtraParams['netParam'].disabled = True
            # self._controller._widgetsExtraParams['netParam'].description = "None"
            self._controller._widgetsExtraParams['netParam'].layout.display = 'none'
        elif (self._netType == consts.NetworkType.ERSOS_RENYI):
            # self._controller._widgetsExtraParams['netParam'].disabled = False
            self._controller._widgetsExtraParams['netParam'].layout.display = 'flex'
            if resetValueAndRange:
                self._controller._widgetsExtraParams['netParam'].min = 0.1
                self._controller._widgetsExtraParams['netParam'].max = 1
                self._controller._widgetsExtraParams['netParam'].step = 0.1
                self._controller._widgetsExtraParams['netParam'].value = 0.5
            self._controller._widgetsExtraParams['netParam'].description = "Network connectivity parameter (link probability)"
        elif (self._netType == consts.NetworkType.BARABASI_ALBERT):
            # self._controller._widgetsExtraParams['netParam'].disabled = False
            self._controller._widgetsExtraParams['netParam'].layout.display = 'flex'
            maxVal = self._systemSize - 1
            if resetValueAndRange:
                self._controller._widgetsExtraParams['netParam'].min = 1
                self._controller._widgetsExtraParams['netParam'].max = maxVal
                self._controller._widgetsExtraParams['netParam'].step = 1
                self._controller._widgetsExtraParams['netParam'].value = min(maxVal, 3)
            self._controller._widgetsExtraParams['netParam'].description = "Network connectivity parameter (new edges)"
        elif (self._netType == consts.NetworkType.SPACE):
            self._controller._widgetsExtraParams['netParam'].value = -1

        if (self._netType == consts.NetworkType.DYNAMIC):
            # self._controller._widgetsExtraParams['netParam'].disabled = False
            self._controller._widgetsExtraParams['netParam'].layout.display = 'flex'
            if resetValueAndRange:
                self._controller._widgetsExtraParams['netParam'].min = 0.0
                self._controller._widgetsExtraParams['netParam'].max = 1
                self._controller._widgetsExtraParams['netParam'].step = 0.05
                self._controller._widgetsExtraParams['netParam'].value = 0.1
            self._controller._widgetsExtraParams['netParam'].description = "Interaction range"
            # self._controller._widgetsExtraParams['particleSpeed'].disabled = False
            # self._controller._widgetsExtraParams['motionCorrelatedness'].disabled = False
            # self._controller._widgetsPlotOnly['showTrace'].disabled = False
            # self._controller._widgetsPlotOnly['showInteractions'].disabled = False
            if self._controller._widgetsExtraParams.get('particleSpeed') is not None:
                self._controller._widgetsExtraParams['particleSpeed'].layout.display = 'flex'
            if self._controller._widgetsExtraParams.get('motionCorrelatedness') is not None:
                self._controller._widgetsExtraParams['motionCorrelatedness'].layout.display = 'flex'
            if self._controller._widgetsPlotOnly.get('showTrace') is not None:
                self._controller._widgetsPlotOnly['showTrace'].layout.display = 'flex'
            if self._controller._widgetsPlotOnly.get('showInteractions') is not None:
                self._controller._widgetsPlotOnly['showInteractions'].layout.display = 'flex'
        else:
            # self._controller._widgetsExtraParams['particleSpeed'].disabled = True
            # self._controller._widgetsExtraParams['motionCorrelatedness'].disabled = True
            # self._controller._widgetsPlotOnly['showTrace'].disabled = True
            # self._controller._widgetsPlotOnly['showInteractions'].disabled = True
            if self._controller._widgetsExtraParams.get('particleSpeed') is not None:
                self._controller._widgetsExtraParams['particleSpeed'].layout.display = 'none'
            if self._controller._widgetsExtraParams.get('motionCorrelatedness') is not None:
                self._controller._widgetsExtraParams['motionCorrelatedness'].layout.display = 'none'
            if self._controller._widgetsPlotOnly.get('showTrace') is not None:
                self._controller._widgetsPlotOnly['showTrace'].layout.display = 'none'
            if self._controller._widgetsPlotOnly.get('showInteractions') is not None:
                self._controller._widgetsPlotOnly['showInteractions'].layout.display = 'none'

        self._controller._widgetsExtraParams['netParam'].readout_format = \
            '.' + str(utils._count_sig_decimals(str(self._controller._widgetsExtraParams['netParam'].step))) + 'f'
        if toLinkPlotFunction:
            self._controller._widgetsExtraParams['netParam'].observe(self._controller._replotFunction, 'value')

    def _adjust_barabasi_network_range(self) -> None:
        """Adjust the widget of the number of edges of the Barabasi-Albert network when the system size slider is changed."""
        if self._controller is None or not self._netType == consts.NetworkType.BARABASI_ALBERT or self._controller._widgetsExtraParams.get('netParam') is None:
            return
        maxVal = self._systemSize - 1
        if self._controller._widgetsExtraParams['netParam'].max == maxVal:
            # the value is correct and no action is necessary
            return

        toLinkPlotFunction = False
        if self._controller._replotFunction:
            try:
                self._controller._widgetsExtraParams['netParam'].unobserve(self._controller._replotFunction, 'value')
                toLinkPlotFunction = True
            except ValueError:
                pass

        self._controller._widgetsExtraParams['netParam'].max = maxVal

        if toLinkPlotFunction:
            self._controller._widgetsExtraParams['netParam'].observe(self._controller._replotFunction, 'value')


class MuMoTSSAView(MuMoTstochasticSimulationView):
    """View for computational simulations of the Gillespie algorithm to approximate the Master Equation solution."""

    # A matrix form of the left-handside of the rules
    _reactantsMatrix = None
    # The effect of each rule
    _ruleChanges = None

    def _constructorSpecificParams(self, _) -> None:
        if self._controller is not None:
            self._generatingCommand = "SSA"

    def _build_bookmark(self, includeParams=True) -> str:
        log_str = "bookmark = " if not self._silent else ""
        log_str += "<modelName>." + self._generatingCommand + "("
        if includeParams:
            log_str += self._get_bookmarks_params()
            log_str += ", "
        log_str = log_str.replace('\\', '\\\\')
        # initState_str = {self._mumotModel._reactantsLaTeX.get(str(state), str(state)): pop
        #                  for state,pop in self._initialState.items()
        #                  if state not in self._mumotModel._constantReactants}
        initState_str = {latex(state): pop
                         for state, pop in self._initialState.items()
                         if state not in self._mumotModel._constantReactants}
        log_str += "initialState = " + str(initState_str)
        log_str += ", maxTime = " + str(self._maxTime)
        log_str += ", randomSeed = " + str(self._randomSeed)
        log_str += ", visualisationType = '" + str(self._visualisationType) + "'"
        if self._visualisationType == 'final':
            # these loops are necessary to return the latex() format of the reactant
            for reactant in self._mumotModel._getAllReactants()[0]:
                if str(reactant) == self._finalViewAxes[0]:
                    log_str += ", final_x = '" + latex(reactant).replace('\\', '\\\\') + "'"
                    break
            for reactant in self._mumotModel._getAllReactants()[0]:
                if str(reactant) == self._finalViewAxes[1]:
                    log_str += ", final_y = '" + latex(reactant).replace('\\', '\\\\') + "'"
                    break
        log_str += ", plotProportions = " + str(self._plotProportions)
        log_str += ", realtimePlot = " + str(self._realtimePlot)
        log_str += ", runs = " + str(self._runs)
        log_str += ", aggregateResults = " + str(self._aggregateResults)
        log_str += ", silent = " + str(self._silent)
        log_str += ", bookmark = False"
        # if len(self._generatingKwargs) > 0:
        #     for key in self._generatingKwargs:
        #         if type(self._generatingKwargs[key]) == str:
        #             log_str += key + " = " + "\'"+ str(self._generatingKwargs[key]) + "\'" + ", "
        #         else:
        #             log_str += key + " = " + str(self._generatingKwargs[key]) + ", "
        log_str += ")"
        return log_str

    def _printStandaloneViewCmd(self) -> None:
        ssa_params = {}
        # initState_str = {}
        # for state,pop in self._initialState.items():
        #     initState_str[str(state)] = pop
        ssa_params["initialState"] = {latex(state): pop
                                      for state, pop in self._initialState.items()
                                      if state not in self._mumotModel._constantReactants}
        ssa_params["maxTime"] = self._maxTime
        ssa_params["randomSeed"] = self._randomSeed
        ssa_params["visualisationType"] = self._visualisationType
        if self._visualisationType == 'final':
            # this loop is necessary to return the latex() format of the reactant
            for reactant in self._mumotModel._getAllReactants()[0]:
                if str(reactant) == self._finalViewAxes[0]:
                    ssa_params['final_x'] = latex(reactant)
                if str(reactant) == self._finalViewAxes[1]:
                    ssa_params['final_y'] = latex(reactant)
        ssa_params["plotProportions"] = self._plotProportions
        ssa_params['realtimePlot'] = self._realtimePlot
        ssa_params['runs'] = self._runs
        ssa_params['aggregateResults'] = self._aggregateResults
        # str( list(self._ratesDict.items()) )
        print("mumot.MuMoTSSAView(<modelName>, None, " + str(self._get_bookmarks_params().replace('\\', '\\\\')) + ", SSParams = " + str(ssa_params) + " )")

    def _simulationStep(self) -> Tuple[float, object]:
        """Update transition probabilities accounting for the current state."""
        probabilitiesOfChange = {}
        for reaction_id, reaction in self._mumotModel._stoichiometry.items():
            prob = self._ratesDict[str(reaction["rate"])]
            numReagents = 0
            for reactant, re_stoch in reaction.items():
                if reactant == 'rate':
                    continue
                if re_stoch == 'const':
                    reactantOccurencies = 1
                else:
                    reactantOccurencies = re_stoch[0]
                if reactantOccurencies > 0:
                    prob *= self._currentState[reactant] ** reactantOccurencies
                numReagents += reactantOccurencies
            # print("for reaction " + str(reaction) + " counted " + str(numReagents) + " reagents (prob:" + str(prob) +" ) (rate: " + str(self._ratesDict[str(reaction["rate"])]) + ")")
            if prob > 0 and numReagents > 1:
                prob /= sum(self._currentState.values())**(numReagents - 1)
            probabilitiesOfChange[reaction_id] = prob
        # for rule in self._reactantsMatrix:
        #     prob = sum([a * b for a,b in zip(rule,currentState)])
        #     numReagents = sum(x > 0 for x in rule)
        #     if numReagents > 1:
        #         prob /= sum(currentState)**( numReagents -1 )
        #     probabilitiesOfChange.append(prob)
        probSum = sum(probabilitiesOfChange.values())
        if probSum == 0:  # no reaction are possible (the execution terminates with this population)
            infiniteTime = self._maxTime - self._t
            return (infiniteTime, self._currentState)
        # computing when is happening next reaction
        timeInterval = np.random.exponential(1 / probSum)

        # Selecting the occurred reaction at random, with probability proportional to each reaction probabilities
        bottom = 0.0
        # Get a random between [0,1) (but we don't want 0!)
        reaction = 0.0
        while reaction == 0.0:
            reaction = np.random.random_sample()
        # Normalising probOfChange in the range [0,1]
        #  probabilitiesOfChange = [pc / probSum for pc in probabilitiesOfChange]
        probabilitiesOfChange = {r_id: pc / probSum
                                 for r_id, pc in probabilitiesOfChange.items()}
        # print("Prob of Change: " + str(probabilitiesOfChange))
        # index = -1
        # for i, prob in enumerate(probabilitiesOfChange):
        #     if ( reaction >= bottom and reaction < (bottom + prob)):
        #         index = i
        #         break
        #     bottom += prob

        # if (index == -1):
        #     print("ERROR! Transition not found. Error in the algorithm execution.")
        #     sys.exit()
        reaction_id = -1
        for r_id, prob in probabilitiesOfChange.items():
            if bottom <= reaction < (bottom + prob):
                reaction_id = r_id
                break
            bottom += prob

        if reaction_id == -1:
            raise exceptions.MuMoTError("ERROR! Transition not found. Error in the algorithm execution.")
            sys.exit()
        # print("current state: " + str(self._currentState))
        # print("triggered change: " + str(self._mumotModel._stoichiometry[reaction_id]))
        # # apply the change
        # currentState = list(np.array(self._currentState) + np.array(self._ruleChanges[index]) )
        for reactant, re_stoch in self._mumotModel._stoichiometry[reaction_id].items():
            if reactant == 'rate' or re_stoch == 'const':
                continue
            self._currentState[reactant] += re_stoch[1] - re_stoch[0]
            if self._currentState[reactant] < 0:
                raise exceptions.MuMoTError(f"ERROR! Population size became negative: {self._currentState}; Error in the algorithm execution.")
                sys.exit()
        # print(self._currentState)

        return (timeInterval, self._currentState)


class Arrow3D(mpatch.FancyArrowPatch):
    """Enable arrows to be added to 3D stream plot.

    Adapted from https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot

    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        mpatch.FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer) -> None:
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        mpatch.FancyArrowPatch.draw(self, renderer)


def _roundNumLogsOut(number: Union[sympy.Add, float]) -> str:
    """ Round numerical output in Logs to 3 decimal places. """
    # if number is complex
    if isinstance(number, sympy.Add):
        return str(sympy.re(number).round(4)) + str(sympy.im(number).round(4)) + 'j'
    # if number is real
    else:
        return str(number.round(4))


def _buildFig(object, figure=None):
    """Generic function for constructing figure objects in :class:`MuMoTview` and :class:`MuMoTmultiController` classes.

    This constructs figure objects with a global figure number.
    To avoid superfluous drawings, ``plt.ion``  and ``plt.ioff``
    are used to turn interactive mode on and off,
    i.e. choosing when data and figure objects are drawn and displayed or not
    to avoid replotting of previous figure windows in addition to updated plots.

    Parameters
    ----------
    object : MuMoTview figure object
        Generated internally.
    figure : optional
        If ``None` then switch interactive mode on ONLY when global figure number larger than 2
        to avoid superfluous matplotlib figure.

    Returns
    -------
    value : Numbered figure object
    """
    global figureCounter
    object._figureNum = figureCounter
    if figureCounter == 1:
        plt.ion()
    else:
        plt.ioff()
    figureCounter += 1
    if figure is None:
        if figureCounter > 2:
            plt.ion()
        object._figure = plt.figure(object._figureNum)
    else:
        object._figure = figure


def _fig_formatting_3D(figure, xlab=None, ylab=None, zlab=None, ax_reformat=False,
                       specialPoints=None, showFixedPoints=False, **kwargs):
    """Function for editing properties of 3D plots.

    Called by :class:`MuMoTvectorView` and :class:`MuMoTstreamView`

    """
    fig = plt.gcf()
    # fig.set_size_inches(10,8)
    ax = fig.gca(projection='3d')

    if kwargs.get('showPlane', False) is True:
        # pointsMesh = np.linspace(0, 1, 11)
        # Xdat, Ydat = np.meshgrid(pointsMesh, pointsMesh)
        # Zdat = 1 - Xdat - Ydat
        # Zdat[Zdat<0] = 0
        # ax.plot_surface(Xdat, Ydat, Zdat, rstride=20, cstride=20, color='grey', alpha=0.25)
        # ax.plot_wireframe(Xdat, Ydat, Zdat, rstride=1, cstride=1, color='grey', alpha=0.5)
        ax.plot([1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], linewidth=2, c='k')

    if xlab is None:
        try:
            xlabelstr = ax.xaxis.get_label_text()
            if len(ax.xaxis.get_label_text()) == 0:
                xlabelstr = 'choose x-label'
        except:
            xlabelstr = 'choose x-label'
    else:
        xlabelstr = xlab

    if ylab is None:
        try:
            ylabelstr = ax.yaxis.get_label_text()
            if len(ax.yaxis.get_label_text()) == 0:
                ylabelstr = 'choose y-label'
        except:
            ylabelstr = 'choose y-label'
    else:
        ylabelstr = ylab

    if zlab is None:
        try:
            zlabelstr = ax.yaxis.get_label_text()
            if len(ax.zaxis.get_label_text()) == 0:
                zlabelstr = 'choose z-label'
        except:
            zlabelstr = 'choose z-label'
    else:
        zlabelstr = zlab

    x_lim_left = ax.get_xbound()[0]  # ax.xaxis.get_data_interval()[0]
    x_lim_right = ax.get_xbound()[1]  # ax.xaxis.get_data_interval()[1]
    y_lim_bot = ax.get_ybound()[0]  # ax.yaxis.get_data_interval()[0]
    y_lim_top = ax.get_ybound()[1]
    z_lim_bot = ax.get_zbound()[0]  # ax.zaxis.get_data_interval()[0]
    z_lim_top = ax.get_zbound()[1]
    if ax_reformat is False:
        xmajortickslocs = ax.xaxis.get_majorticklocs()
        # xminortickslocs = ax.xaxis.get_minorticklocs()
        ymajortickslocs = ax.yaxis.get_majorticklocs()
        # yminortickslocs = ax.yaxis.get_minorticklocs()
        zmajortickslocs = ax.zaxis.get_majorticklocs()
        # zminortickslocs = ax.zaxis.get_minorticklocs()
        # plt.cla()
        ax.set_xticks(xmajortickslocs)
        # ax.set_xticks(xminortickslocs, minor = True)
        ax.set_yticks(ymajortickslocs)
        # ax.set_yticks(yminortickslocs, minor = True)
        ax.set_zticks(zmajortickslocs)
        # ax.set_zticks(zminortickslocs, minor = True)
    else:
        max_xrange = x_lim_right - x_lim_left
        max_yrange = y_lim_top - y_lim_bot
        if kwargs.get('showPlane', False) is True:
            max_zrange = z_lim_top
        else:
            max_zrange = z_lim_top - z_lim_bot

        if max_xrange < 1.0:
            xMLocator_major = utils._round_to_1(max_xrange / 4)
        else:
            xMLocator_major = utils._round_to_1(max_xrange / 6)
        # xMLocator_minor = xMLocator_major / 2
        if max_yrange < 1.0:
            yMLocator_major = utils._round_to_1(max_yrange / 4)
        else:
            yMLocator_major = utils._round_to_1(max_yrange / 6)
        # yMLocator_minor = yMLocator_major / 2
        if max_zrange < 1.0:
            zMLocator_major = utils._round_to_1(max_zrange / 4)
        else:
            zMLocator_major = utils._round_to_1(max_zrange / 6)
        # zMLocator_minor = yMLocator_major / 2
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xMLocator_major))
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(xMLocator_minor))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(yMLocator_major))
        # ax.yaxis.set_minor_locator(ticker.MultipleLocator(yMLocator_minor))
        ax.zaxis.set_major_locator(ticker.MultipleLocator(zMLocator_major))
        # ax.zaxis.set_minor_locator(ticker.MultipleLocator(zMLocator_minor))

    ax.set_xlim3d(x_lim_left, x_lim_right)
    ax.set_ylim3d(y_lim_bot, y_lim_top)
    if kwargs.get('showPlane', False) is True:
        ax.set_zlim3d(0, z_lim_top)
    else:
        ax.set_zlim3d(z_lim_bot, z_lim_top)

    if showFixedPoints is True:
        if not specialPoints[0] == []:
            for jj in range(len(specialPoints[0])):
                try:
                    len(specialPoints[3][jj]) == 3
                    lam1 = specialPoints[3][jj][0]
                    lam2 = specialPoints[3][jj][1]
                    lam3 = specialPoints[3][jj][2]
                    if (sympy.re(lam1) < 0 and sympy.re(lam2) < 0 and sympy.re(lam3) < 0):
                        FPcolor = 'g'
                        FPmarker = 'o'
                    else:
                        FPcolor = 'r'
                        FPmarker = '>'
                except:
                    print('Check input!')
                    FPcolor = 'k'

                ax.scatter([specialPoints[0][jj]], [specialPoints[1][jj]], [specialPoints[2][jj]],
                           marker=FPmarker, s=100, c=FPcolor)

    if kwargs.get('fontsize', None) is not None:
        axes_font_size = kwargs['fontsize']
    elif len(xlabelstr) > 40 or len(ylabelstr) > 40 or len(zlabelstr) > 40:
        axes_font_size = 12
    elif 31 <= len(xlabelstr) <= 40 or 31 <= len(ylabelstr) <= 40 or 31 <= len(zlabelstr) <= 40:
        axes_font_size = 16
    elif 26 <= len(xlabelstr) <= 30 or 26 <= len(ylabelstr) <= 30 or 26 <= len(zlabelstr) <= 30:
        axes_font_size = 22
    else:
        axes_font_size = 26

    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = 20
    ax.set_xlabel(r'' + str(xlabelstr), fontsize=axes_font_size)
    ax.set_ylabel(r'' + str(ylabelstr), fontsize=axes_font_size)
    if len(str(zlabelstr)) > 1:
        ax.set_zlabel(r'' + str(zlabelstr), fontsize=axes_font_size, rotation=90)
    else:
        ax.set_zlabel(r'' + str(zlabelstr), fontsize=axes_font_size)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax.zaxis.get_major_ticks():
        tick.set_pad(8)
        tick.label.set_fontsize(18)

    plt.tight_layout(pad=4)


def _fig_formatting_2D(figure=None, xdata=None, ydata=None, choose_xrange=None, choose_yrange=None, eigenvalues=None,
                       curve_replot=False, ax_reformat=False, showFixedPoints=False, specialPoints=None,
                       xlab=None, ylab=None, curvelab=None, aspectRatioEqual=False, line_color_list=consts.LINE_COLOR_LIST,
                       **kwargs):
    """Format 2D plots.

    Called by :class:`MuMoTvectorView`, :class:`MuMoTstreamView` and :class:`MuMoTbifurcationView`

    """
    showLegend = kwargs.get('showLegend', False)

    linestyle_list = ['solid', 'dashed', 'dashdot', 'dotted', 'solid', 'dashed', 'dashdot', 'dotted', 'solid']

    if xdata and ydata:
        if len(xdata) == len(ydata):
            # plt.figure(figsize=(8,6), dpi=80)
            # ax = plt.axes()
            ax = plt.gca()
            data_x = xdata
            data_y = ydata

        else:
            print('CHECK input:')
            print('xdata and ydata are lists of arrays and must have same lengths!')
            print('Array pairs xdata[k] and ydata[k] (k = 0, ..., N-1) must have same lengths too!')

    elif figure:
        plt.gcf()
        ax = plt.gca()
        data_x = [ax.lines[kk].get_xdata() for kk in range(len(ax.lines))]
        data_y = [ax.lines[kk].get_ydata() for kk in range(len(ax.lines))]

    else:
        print('Choose either figure or dataset(s)')
    # print(data_x)

    if xlab is None:
        try:
            xlabelstr = ax.xaxis.get_label_text()
            if len(ax.xaxis.get_label_text()) == 0:
                xlabelstr = 'choose x-label'
        except:
            xlabelstr = 'choose x-label'
    else:
        xlabelstr = xlab

    if ylab is None:
        try:
            ylabelstr = ax.yaxis.get_label_text()
            if len(ax.yaxis.get_label_text()) == 0:
                ylabelstr = 'choose y-label'
        except:
            ylabelstr = 'choose y-label'
    else:
        ylabelstr = ylab

    if ax_reformat is False and figure is not None:
        xmajortickslocs = ax.xaxis.get_majorticklocs()
        xminortickslocs = ax.xaxis.get_minorticklocs()
        ymajortickslocs = ax.yaxis.get_majorticklocs()
        yminortickslocs = ax.yaxis.get_minorticklocs()
        x_lim_left = ax.get_xbound()[0]  # ax.xaxis.get_data_interval()[0]
        x_lim_right = ax.get_xbound()[1]  # ax.xaxis.get_data_interval()[1]
        y_lim_bot = ax.get_ybound()[0]  # ax.yaxis.get_data_interval()[0]
        y_lim_top = ax.get_ybound()[1]  # ax.yaxis.get_data_interval()[1]
        # print(ax.yaxis.get_data_interval())

    if curve_replot is True:
        plt.cla()

    if ax_reformat is False and figure is not None:
        ax.set_xticks(xmajortickslocs)
        ax.set_xticks(xminortickslocs, minor=True)
        ax.set_yticks(ymajortickslocs)
        ax.set_yticks(yminortickslocs, minor=True)
        ax.tick_params(axis='both', which='major', length=5, width=2)
        ax.tick_params(axis='both', which='minor', length=3, width=1)
        plt.xlim(x_lim_left, x_lim_right)
        plt.ylim(y_lim_bot, y_lim_top)

    if figure is None or curve_replot is True:

        if 'LineThickness' in kwargs:
            LineThickness = kwargs['LineThickness']
        else:
            LineThickness = 4

        if eigenvalues:
            showLegend = False
            round_digit = 10
            # solX_dict={} #bifurcation parameter
            # solY_dict={} #state variable 1
            # solX_dict['solX_unst']=[]
            # solY_dict['solY_unst']=[]
            # solX_dict['solX_stab']=[]
            # solY_dict['solY_stab']=[]
            # solX_dict['solX_saddle']=[]
            # solY_dict['solY_saddle']=[]

            # nr_sol_unst=0
            # nr_sol_saddle=0
            # nr_sol_stab=0
            # data_x_tmp=[]
            # data_y_tmp=[]
            # #print(specialPoints)

            Nr_unstable = 0
            Nr_stable = 0
            Nr_saddle = 0

            for nn in range(len(data_x)):
                solX_dict = {}  # bifurcation parameter
                solY_dict = {}  # state variable 1
                solX_dict['solX_unst'] = []
                solY_dict['solY_unst'] = []
                solX_dict['solX_stab'] = []
                solY_dict['solY_stab'] = []
                solX_dict['solX_saddle'] = []
                solY_dict['solY_saddle'] = []

                nr_sol_unst = 0
                nr_sol_saddle = 0
                nr_sol_stab = 0
                data_x_tmp = []
                data_y_tmp = []
                # sign_change=0
                for kk in range(len(eigenvalues[nn])):
                    if kk > 0:
                        if len(eigenvalues[0][0]) == 1:
                            if (np.sign(np.round(np.real(eigenvalues[nn][kk][0]), round_digit)) * np.sign(np.round(np.real(eigenvalues[nn][kk - 1][0]), round_digit)) <= 0):
                                # print('sign change')
                                # sign_change+=1
                                # print(sign_change)
                                # if specialPoints is not None and specialPoints[0]!=[]:
                                #     data_x_tmp.append(specialPoints[0][sign_change-1])
                                #     data_y_tmp.append(specialPoints[1][sign_change-1])

                                if nr_sol_unst == 1:
                                    solX_dict['solX_unst'].append(data_x_tmp)
                                    solY_dict['solY_unst'].append(data_y_tmp)
                                elif nr_sol_saddle == 1:
                                    solX_dict['solX_saddle'].append(data_x_tmp)
                                    solY_dict['solY_saddle'].append(data_y_tmp)
                                elif nr_sol_stab == 1:
                                    solX_dict['solX_stab'].append(data_x_tmp)
                                    solY_dict['solY_stab'].append(data_y_tmp)
                                else:
                                    print('Something went wrong!')

                                data_x_tmp_first = data_x_tmp[-1]
                                data_y_tmp_first = data_y_tmp[-1]
                                nr_sol_stab = 0
                                nr_sol_saddle = 0
                                nr_sol_unst = 0
                                data_x_tmp = []
                                data_y_tmp = []
                                data_x_tmp.append(data_x_tmp_first)
                                data_y_tmp.append(data_y_tmp_first)
                                # if specialPoints is not None and specialPoints[0]!=[]:
                                #     data_x_tmp.append(specialPoints[0][sign_change-1])
                                #     data_y_tmp.append(specialPoints[1][sign_change-1])
                        elif len(eigenvalues[0][0]) == 2:
                            if (np.sign(np.round(np.real(eigenvalues[nn][kk][0]), round_digit)) * np.sign(np.round(np.real(eigenvalues[nn][kk - 1][0]), round_digit)) <= 0
                                or np.sign(np.round(np.real(eigenvalues[nn][kk][1]), round_digit)) * np.sign(np.round(np.real(eigenvalues[nn][kk - 1][1]), round_digit)) <= 0):
                                # print('sign change')
                                # sign_change+=1
                                # print(sign_change)
                                # if specialPoints is not None and specialPoints[0]!=[]:
                                #     data_x_tmp.append(specialPoints[0][sign_change-1])
                                #     data_y_tmp.append(specialPoints[1][sign_change-1])

                                if nr_sol_unst == 1:
                                    solX_dict['solX_unst'].append(data_x_tmp)
                                    solY_dict['solY_unst'].append(data_y_tmp)
                                elif nr_sol_saddle == 1:
                                    solX_dict['solX_saddle'].append(data_x_tmp)
                                    solY_dict['solY_saddle'].append(data_y_tmp)
                                elif nr_sol_stab == 1:
                                    solX_dict['solX_stab'].append(data_x_tmp)
                                    solY_dict['solY_stab'].append(data_y_tmp)
                                else:
                                    print('Something went wrong!')

                                data_x_tmp_first = data_x_tmp[-1]
                                data_y_tmp_first = data_y_tmp[-1]
                                nr_sol_stab = 0
                                nr_sol_saddle = 0
                                nr_sol_unst = 0
                                data_x_tmp = []
                                data_y_tmp = []
                                data_x_tmp.append(data_x_tmp_first)
                                data_y_tmp.append(data_y_tmp_first)
                                # if specialPoints is not None and specialPoints[0]!=[]:
                                #     data_x_tmp.append(specialPoints[0][sign_change-1])
                                #     data_y_tmp.append(specialPoints[1][sign_change-1])

                    if len(eigenvalues[0][0]) == 1:
                        if np.sign(np.round(np.real(eigenvalues[nn][kk][0]), round_digit)) == -1:
                            nr_sol_stab = 1
                        else:
                            nr_sol_unst = 1

                    elif len(eigenvalues[0][0]) == 2:
                        if np.sign(np.round(np.real(eigenvalues[nn][kk][0]), round_digit)) == -1 and np.sign(np.round(np.real(eigenvalues[nn][kk][1]), round_digit)) == -1:
                            nr_sol_stab = 1
                        elif np.sign(np.round(np.real(eigenvalues[nn][kk][0]), round_digit)) in [0, 1] and np.sign(np.round(np.real(eigenvalues[nn][kk][1]), round_digit)) == -1:
                            nr_sol_saddle = 1
                        elif np.sign(np.round(np.real(eigenvalues[nn][kk][0]), round_digit)) == -1 and np.sign(np.round(np.real(eigenvalues[nn][kk][1]), round_digit)) in [0, 1]:
                            nr_sol_saddle = 1
                        else:
                            nr_sol_unst = 1
                        # if np.real(eigenvalues[nn][kk][0]) < 0 and np.real(eigenvalues[nn][kk][1]) < 0:
                        #     nr_sol_stab=1
                        # elif np.real(eigenvalues[nn][kk][0]) >= 0 and np.real(eigenvalues[nn][kk][1]) < 0:
                        #     nr_sol_saddle=1
                        # elif np.real(eigenvalues[nn][kk][0]) < 0 and np.real(eigenvalues[nn][kk][1]) >= 0:
                        #     nr_sol_saddle=1
                        # else:
                        #     nr_sol_unst=1

                    data_x_tmp.append(data_x[nn][kk])
                    data_y_tmp.append(data_y[nn][kk])

                    if kk == len(eigenvalues[nn]) - 1:
                        if nr_sol_unst == 1:
                            solX_dict['solX_unst'].append(data_x_tmp)
                            solY_dict['solY_unst'].append(data_y_tmp)
                        elif nr_sol_saddle == 1:
                            solX_dict['solX_saddle'].append(data_x_tmp)
                            solY_dict['solY_saddle'].append(data_y_tmp)
                        elif nr_sol_stab == 1:
                            solX_dict['solX_stab'].append(data_x_tmp)
                            solY_dict['solY_stab'].append(data_y_tmp)
                        else:
                            print('Something went wrong!')

                if not solX_dict['solX_unst'] == []:
                    for jj in range(len(solX_dict['solX_unst'])):
                        if jj == 0 and Nr_unstable == 0:
                            label_description = r'unstable'
                            Nr_unstable = 1
                        else:
                            label_description = '_nolegend_'
                        plt.plot(solX_dict['solX_unst'][jj],
                                 solY_dict['solY_unst'][jj],
                                 c=line_color_list[2],
                                 ls=linestyle_list[1], lw=LineThickness, label=label_description)
                if not solX_dict['solX_stab'] == []:
                    for jj in range(len(solX_dict['solX_stab'])):
                        if jj == 0 and Nr_stable == 0:
                            label_description = r'stable'
                            Nr_stable = 1
                        else:
                            label_description = '_nolegend_'
                        plt.plot(solX_dict['solX_stab'][jj],
                                 solY_dict['solY_stab'][jj],
                                 c=line_color_list[1],
                                 ls=linestyle_list[0], lw=LineThickness, label=label_description)
                if not solX_dict['solX_saddle'] == []:
                    for jj in range(len(solX_dict['solX_saddle'])):
                        if jj == 0 and Nr_saddle == 0:
                            label_description = r'saddle'
                            Nr_saddle = 1
                        else:
                            label_description = '_nolegend_'
                        plt.plot(solX_dict['solX_saddle'][jj],
                                 solY_dict['solY_saddle'][jj],
                                 c=line_color_list[0],
                                 ls=linestyle_list[3], lw=LineThickness, label=label_description)

            # if not solX_dict['solX_unst'] == []:
            #     for jj in range(len(solX_dict['solX_unst'])):
            #         plt.plot(solX_dict['solX_unst'][jj],
            #                  solY_dict['solY_unst'][jj],
            #                  c = line_color_list[2],
            #                  ls = linestyle_list[3], lw = LineThickness, label = r'unstable')
            # if not solX_dict['solX_stab'] == []:
            #     for jj in range(len(solX_dict['solX_stab'])):
            #         plt.plot(solX_dict['solX_stab'][jj],
            #                  solY_dict['solY_stab'][jj],
            #                  c = line_color_list[1],
            #                  ls = linestyle_list[0], lw = LineThickness, label = r'stable')
            # if not solX_dict['solX_saddle'] == []:
            #     for jj in range(len(solX_dict['solX_saddle'])):
            #         plt.plot(solX_dict['solX_saddle'][jj],
            #                  solY_dict['solY_saddle'][jj],
            #                  c = line_color_list[0],
            #                  ls = linestyle_list[1], lw = LineThickness, label = r'saddle')

        else:
            for nn in range(len(data_x)):
                try:
                    plt.plot(data_x[nn], data_y[nn], c=line_color_list[nn],
                             ls=linestyle_list[nn], lw=LineThickness, label=r'' + str(curvelab[nn]))
                except:
                    plt.plot(data_x[nn], data_y[nn], c=line_color_list[nn],
                             ls=linestyle_list[nn], lw=LineThickness)

    if len(xlabelstr) > 40 or len(ylabelstr) > 40:
        axes_font_size = 10  # 16
    elif 31 <= len(xlabelstr) <= 40 or 31 <= len(ylabelstr) <= 40:
        axes_font_size = 14  # 20
    elif 26 <= len(xlabelstr) <= 30 or 26 <= len(ylabelstr) <= 30:
        axes_font_size = 18  # 26
    else:
        axes_font_size = 24  # 30

    if 'fontsize' in kwargs:
        if not kwargs['fontsize'] is None:
            axes_font_size = kwargs['fontsize']

    plt.xlabel(r'' + str(xlabelstr), fontsize=axes_font_size)
    plt.ylabel(r'' + str(ylabelstr), fontsize=axes_font_size)

    if figure is None or ax_reformat is True or choose_xrange is not None or choose_yrange is not None:
        if choose_xrange:
            max_xrange = choose_xrange[1] - choose_xrange[0]
        else:
            # xrange = [np.max(data_x[kk]) - np.min(data_x[kk]) for kk in range(len(data_x))]
            XaxisMax = np.max([np.max(data_x[kk]) for kk in range(len(data_x))])
            XaxisMin = np.min([np.min(data_x[kk]) for kk in range(len(data_x))])
            max_xrange = XaxisMax - XaxisMin  # max(xrange)

        if choose_yrange:
            max_yrange = choose_yrange[1] - choose_yrange[0]
        else:
            # yrange = [np.max(data_y[kk]) - np.min(data_y[kk]) for kk in range(len(data_y))]
            # max_yrange = np.max(data_y) - np.min(data_y) #max(yrange)
            YaxisMax = np.max([np.max(data_y[kk]) for kk in range(len(data_y))])
            YaxisMin = np.min([np.min(data_y[kk]) for kk in range(len(data_y))])
            max_yrange = YaxisMax - YaxisMin

        if max_xrange < 1.0:
            xMLocator_major = utils._round_to_1(max_xrange / 5)
        else:
            xMLocator_major = utils._round_to_1(max_xrange / 10)
        xMLocator_minor = xMLocator_major / 2
        if max_yrange < 1.0:
            yMLocator_major = utils._round_to_1(max_yrange / 5)
        else:
            yMLocator_major = utils._round_to_1(max_yrange / 10)
        yMLocator_minor = yMLocator_major / 2

        if choose_xrange:
            plt.xlim(choose_xrange[0] - xMLocator_minor / 10.0, choose_xrange[1] + xMLocator_minor / 10.0)
        else:
            plt.xlim(XaxisMin - xMLocator_minor / 10.0, XaxisMax + xMLocator_minor / 10.0)
        if choose_yrange:
            plt.ylim(choose_yrange[0] - yMLocator_minor / 10.0, choose_yrange[1] + yMLocator_minor / 10.0)
        else:
            plt.ylim(YaxisMin - yMLocator_minor / 10.0, YaxisMax + yMLocator_minor / 10.0)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(xMLocator_major))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(xMLocator_minor))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(yMLocator_major))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(yMLocator_minor))
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)

        ax.tick_params('both', length=5, width=2, which='major')
        ax.tick_params('both', length=3, width=1, which='minor')

    if eigenvalues:
        if specialPoints != []:
            if specialPoints[0] != []:
                for jj in range(len(specialPoints[0])):
                    plt.plot([specialPoints[0][jj]], [specialPoints[1][jj]], marker='o', markersize=8,
                             c=line_color_list[-1])
                for a, b, c in zip(specialPoints[0], specialPoints[1], specialPoints[2]):
                    if a > plt.xlim()[0] + (plt.xlim()[1] - plt.xlim()[0]) / 2:
                        x_offset = -(plt.xlim()[1] - plt.xlim()[0]) * 0.02
                    else:
                        x_offset = (plt.xlim()[1] - plt.xlim()[0]) * 0.02
                    if b > plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) / 2:
                        y_offset = -(plt.ylim()[1] - plt.ylim()[0]) * 0.05
                    else:
                        y_offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.05
                    plt.text(a + x_offset, b + y_offset, c, fontsize=18)

    if showFixedPoints:
        if not specialPoints[0] == []:
            for jj in range(len(specialPoints[0])):
                try:
                    len(specialPoints[2][jj]) == 2
                    lam1 = specialPoints[2][jj][0]
                    lam2 = specialPoints[2][jj][1]
                    if sympy.re(lam1) < 0 and sympy.re(lam2) < 0:
                        FPcolor = line_color_list[1]
                        FPfill = 'full'
                    elif sympy.re(lam1) > 0 and sympy.re(lam2) > 0:
                        FPcolor = line_color_list[2]
                        FPfill = 'none'
                    else:
                        FPcolor = line_color_list[0]
                        FPfill = 'none'
                except:
                    print('Check input!')
                    FPcolor = line_color_list[-1]
                    FPfill = 'none'
                if sympy.re(lam1) != 0 and sympy.re(lam2) != 0:
                    plt.plot([specialPoints[0][jj]], [specialPoints[1][jj]], marker='o', markersize=9,
                             c=FPcolor, fillstyle=FPfill, mew=3, mec=FPcolor)

    plt.grid(kwargs.get('grid', False))

    if curvelab is not None or showLegend:
        # if 'legend_loc' in kwargs:
        #     legend_loc = kwargs['legend_loc']
        # else:
        #     legend_loc = 'upper left'
        legend_fontsize = kwargs.get('legend_fontsize', 14)
        legend_loc = kwargs.get('legend_loc', 'upper left')
        plt.legend(loc=str(legend_loc), fontsize=legend_fontsize, ncol=2)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(13)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(13)

    plt.tight_layout()

    if aspectRatioEqual:
        ax.set_aspect('equal')


def _fig_formatting_1D(figure=None, xdata=None, choose_xrange=None,
                       choose_yrange=None, eigenvalues=None,
                       showFixedPoints=False, specialPoints=None, xlab=None,
                       **kwargs) -> None:
    """Format 1D plots.

    Called by :class:`MuMoTstreamView`.

    """
    if xdata:
        ax = plt.gca()
        data_x = xdata
    elif figure:
        # plt.gcf()
        fig1dStream = plt.gcf()
        fig1dStream.set_size_inches(6, 2)
        ax = plt.gca()
        data_x = [ax.lines[kk].get_xdata() for kk in range(len(ax.lines))]
    else:
        print('Choose either figure or dataset(s)')

    if xlab is None:
        try:
            xlabelstr = ax.xaxis.get_label_text()
            if len(ax.xaxis.get_label_text()) == 0:
                xlabelstr = 'choose x-label'
        except:
            xlabelstr = 'choose x-label'
    else:
        xlabelstr = xlab

    if figure is not None:
        xmajortickslocs = ax.xaxis.get_majorticklocs()
        xminortickslocs = ax.xaxis.get_minorticklocs()
        x_lim_left = ax.get_xbound()[0]  # ax.xaxis.get_data_interval()[0]
        x_lim_right = ax.get_xbound()[1]  # ax.xaxis.get_data_interval()[1]
        ax.set_xticks(xmajortickslocs)
        ax.set_xticks(xminortickslocs, minor=True)
        ax.tick_params(axis='both', which='major', length=5, width=2)
        ax.tick_params(axis='both', which='minor', length=3, width=1)
        plt.xlim(x_lim_left, x_lim_right)
        plt.yticks([])

    if len(xlabelstr) > 40:
        axes_font_size = 10  # 16
    elif 31 <= len(xlabelstr) <= 40:
        axes_font_size = 14  # 20
    elif 26 <= len(xlabelstr) <= 30:
        axes_font_size = 18  # 26
    else:
        axes_font_size = 24  # 30

    if 'fontsize' in kwargs:
        if not kwargs['fontsize'] is None:
            axes_font_size = kwargs['fontsize']

    plt.xlabel(r'' + str(xlabelstr), fontsize=axes_font_size)
    plt.ylabel('')

    if figure is None or choose_xrange is not None:
        if choose_yrange:
            plt.ylim(choose_yrange[0], choose_yrange[1])
        if choose_xrange:
            max_xrange = choose_xrange[1] - choose_xrange[0]
        else:
            XaxisMax = np.max([np.max(data_x[kk]) for kk in range(len(data_x))])
            XaxisMin = np.min([np.min(data_x[kk]) for kk in range(len(data_x))])
            max_xrange = XaxisMax - XaxisMin  # max(xrange)

        if max_xrange < 1.0:
            xMLocator_major = utils._round_to_1(max_xrange / 5)
        else:
            xMLocator_major = utils._round_to_1(max_xrange / 10)
        xMLocator_minor = xMLocator_major / 2

        if choose_xrange:
            plt.xlim(choose_xrange[0] - xMLocator_minor / 10.0, choose_xrange[1] + xMLocator_minor / 10.0)
        else:
            plt.xlim(XaxisMin - xMLocator_minor / 10.0, XaxisMax + xMLocator_minor / 10.0)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(xMLocator_major))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(xMLocator_minor))
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)

        ax.tick_params('both', length=5, width=2, which='major')
        ax.tick_params('both', length=3, width=1, which='minor')

    if showFixedPoints is True:
        if not specialPoints[0] == []:
            for jj in range(len(specialPoints[0])):
                try:
                    lam1 = specialPoints[1][jj]
                    if sympy.re(lam1) < 0:
                        FPfill = 'full'
                        circleColor = 'green'
                    else:
                        FPfill = 'none'
                        circleColor = 'red'
                except:
                    print('Check input!')
                    FPfill = 'none'
                if sympy.re(lam1) != 0:
                    plt.plot([specialPoints[0][jj]], 0.0, marker='o', markersize=9,
                             c=circleColor, fillstyle=FPfill, mew=3)

    plt.grid(kwargs.get('grid', False))

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(13)

    plt.tight_layout()


def _plot_point_cov(
        points: np.ndarray,
        nstd: Optional[float] = 2,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs) -> matplotlib.patches.Ellipse:
    """
    Plots an ``nstd`` sigma ellipse based on the mean and covariance of a point
    "cloud" (``points``, an Nx2 array).

    Parameters
    ----------
    points : numpy.ndarray
        An Nx2 array of the data points.
    nstd : float, optional
        The radius of the ellipse in numbers of standard deviations.
        Defaults to 2 standard deviations.
    ax : matplotlib.axes.Axes, optional
        The axis that the ellipse will be plotted on. Defaults to the
        current axis.

    Returns
    -------
    :class:`matplotlib.patches.Ellipse`
        A matplotlib ellipse artist

    Notes
    -----
    Additional keyword arguments are pass on to the ellipse patch.

    Credit
    ------
    https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py
    """
    # Copyright (c) 2012 Free Software Foundation
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy of
    # this software and associated documentation files (the "Software"), to deal in
    # the Software without restriction, including without limitation the rights to
    # use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
    # of the Software, and to permit persons to whom the Software is furnished to do
    # so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return _plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


def _plot_cov_ellipse(
        cov: np.ndarray,
        pos: np.ndarray, nstd:
        Optional[float] = 2.0,
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs) -> matplotlib.patches.Ellipse:
    """
    Plots an ``nstd`` sigma error ellipse based on the specified covariance
    matrix (``cov``). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
    cov : numpy.ndarray
        The 2x2 covariance matrix to base the ellipse on.
    pos : list of float
        The location of the center of the ellipse. Expects a 2-element
        sequence of [x0, y0].
    nstd : float, optional
        The radius of the ellipse in numbers of standard deviations.
        Defaults to 2 standard deviations.
    ax : matplotlib.axes.Axes, optional
        The axis that the ellipse will be plotted on. Defaults to the
        current axis.

    Returns
    -------
    :class:`matplotlib`.patches.Ellipse
        A matplotlib ellipse artist

    Notes
    -----
    Additional keyword arguments are pass on to the ellipse patch.

    Credit
    ------
    https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py
    """
    # Copyright (c) 2012 Free Software Foundation
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy of
    # this software and associated documentation files (the "Software"), to deal in
    # the Software without restriction, including without limitation the rights to
    # use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
    # of the Software, and to permit persons to whom the Software is furnished to do
    # so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.
    def _eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = _eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = mpatch.Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def _deriveODEsFromRules(reactants, rules):
    # @todo: replace with principled derivation via Master Equation and van Kampen expansion
    equations = {}
    terms = []
    for rule in rules:
        term = None
        for reactant in rule.lhsReactants:
            if term is None:
                term = reactant
            else:
                term = term * reactant
        term = term * rule.rate
        terms.append(term)
    for reactant in reactants:
        rhs = None
        for rule, term in zip(rules, terms):
            # I love Python!
            factor = rule.rhsReactants.count(reactant) - rule.lhsReactants.count(reactant)
            if factor != 0:
                if rhs is None:
                    rhs = factor * term
                else:
                    rhs = rhs + factor * term
        equations[reactant] = rhs

    return equations


def _deriveMasterEquation(stoichiometry):
    """Derive the Master equation

    Returns dictionary used in :method:`MuMoTmodel.showMasterEquation`.
    """
    substring = None

    x, y, v, w, t, m = symbols('x y v w t m')
    E_op = Function('E_op')
    z = Function('z')
    P = Function('P')
    V = Symbol(r'\overline{V}', real=True, constant=True)

    stoich = stoichiometry
    nvec = []
    for key1 in stoich:
        for key2 in stoich[key1]:
            if key2 != 'rate' and stoich[key1][key2] != 'const':
                if key2 not in nvec:
                    nvec.append(key2)
                if len(stoich[key1][key2]) == 3:
                    substring = stoich[key1][key2][2]
    nvec = sorted(nvec, key=default_sort_key)

    if len(nvec) < 1 or len(nvec) > 4:
        print("Derivation of Master Equation works for 1, 2, 3 or 4 different reactants only")

        return None, None

    # assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'

    rhs = 0
    sol_dict_rhs = {}
    f = lambdify(z(y, v - w), z(y, v - w), modules='sympy')
    g = lambdify((x, y, v), (factorial(x) / factorial(x - y)) / v**y, modules='sympy')
    for key1 in stoich:
        prod1 = 1
        prod2 = 1
        rate_fact = 1
        for key2 in stoich[key1]:
            if key2 != 'rate' and stoich[key1][key2] != 'const':
                prod1 *= f(E_op(key2, stoich[key1][key2][0] - stoich[key1][key2][1]))
                prod2 *= g(key2, stoich[key1][key2][0], V)
            if stoich[key1][key2] == 'const':
                rate_fact *= key2 / V

        if len(nvec) == 1:
            sol_dict_rhs[key1] = (prod1, simplify(prod2 * V), P(nvec[0], t), stoich[key1]['rate'] * rate_fact)
        elif len(nvec) == 2:
            sol_dict_rhs[key1] = (prod1, simplify(prod2 * V), P(nvec[0], nvec[1], t), stoich[key1]['rate'] * rate_fact)
        elif len(nvec) == 3:
            sol_dict_rhs[key1] = (prod1, simplify(prod2 * V), P(nvec[0], nvec[1], nvec[2], t), stoich[key1]['rate'] * rate_fact)
        else:
            sol_dict_rhs[key1] = (prod1, simplify(prod2 * V), P(nvec[0], nvec[1], nvec[2], nvec[3], t), stoich[key1]['rate'] * rate_fact)

    return sol_dict_rhs, substring


def _doVanKampenExpansion(rhs, stoich):
    """Return the left-hand side and right-hand side of van Kampen expansion."""
    x, y, v, w, t, m = symbols('x y v w t m')
    E_op = Function('E_op')
    P = Function('P')
    V = Symbol(r'\overline{V}', real=True, constant=True)
    nvec = []
    nconstvec = []
    for key1 in stoich:
        for key2 in stoich[key1]:
            if key2 != 'rate' and stoich[key1][key2] != 'const':
                if key2 not in nvec:
                    nvec.append(key2)
            elif key2 != 'rate' and stoich[key1][key2] == 'const':
                if key2 not in nconstvec:
                    nconstvec.append(key2)

    nvec = sorted(nvec, key=default_sort_key)
    if len(nvec) < 1 or len(nvec) > 4:
        print("van Kampen expansion works for 1, 2, 3 or 4 different reactants only")

        return None, None, None
    # assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'

    NoiseDict = {}
    PhiDict = {}
    PhiConstDict = {}

    for kk in range(len(nvec)):
        NoiseDict[nvec[kk]] = Symbol(f"eta_{nvec[kk]}")
        PhiDict[nvec[kk]] = Symbol(f"Phi_{nvec[kk]}")

    for kk in range(len(nconstvec)):
        PhiConstDict[nconstvec[kk]] = V * Symbol(f"Phi_{nconstvec[kk]}")

    rhs_dict, substring = rhs(stoich)
    rhs_vKE = 0

    if len(nvec) == 1:
        lhs_vKE = (Derivative(P(nvec[0], t), t).subs({nvec[0]: NoiseDict[nvec[0]]}) -
                   sympy.sqrt(V) * Derivative(PhiDict[nvec[0]], t) * Derivative(P(nvec[0], t), nvec[0]).subs({nvec[0]: NoiseDict[nvec[0]]}))
        for key in rhs_dict:
            op = rhs_dict[key][0].subs({nvec[0]: NoiseDict[nvec[0]]})
            func1 = rhs_dict[key][1].subs({nvec[0]: V * PhiDict[nvec[0]] + sympy.sqrt(V) * NoiseDict[nvec[0]]})
            func2 = rhs_dict[key][2].subs({nvec[0]: NoiseDict[nvec[0]]})
            func = func1 * func2
            # if len(op.args[0].args) ==0:
            term = (op * func).subs({
                op * func: func + op.args[1] / sympy.sqrt(V) * Derivative(func, op.args[0]) + op.args[1]**2 / (2 * V) * Derivative(func, op.args[0], op.args[0])})
            # else:
            #     term = (op.args[1] * func).subs({op.args[1] * func: func + op.args[1].args[1] / sympy.sqrt(V) * Derivative(func, op.args[1].args[0])
            #                            + op.args[1].args[1]**2 / (2 * V) * Derivative(func, op.args[1].args[0], op.args[1].args[0])})
            #     term = (op.args[0] * term).subs({op.args[0] * term: term + op.args[0].args[1] / sympy.sqrt(V) * Derivative(term, op.args[0].args[0])
            #                            + op.args[0].args[1]**2 / (2 * V) * Derivative(term, op.args[0].args[0], op.args[0].args[0])})
            rhs_vKE += rhs_dict[key][3].subs(PhiConstDict) * (term.doit() - func)
    elif len(nvec) == 2:
        lhs_vKE = (Derivative(P(nvec[0], nvec[1], t), t).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]]})
                   - sympy.sqrt(V) * Derivative(PhiDict[nvec[0]], t) * Derivative(P(nvec[0], nvec[1], t), nvec[0]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]]})
                   - sympy.sqrt(V) * Derivative(PhiDict[nvec[1]], t) * Derivative(P(nvec[0], nvec[1], t), nvec[1]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]]}))

        for key in rhs_dict:
            op = rhs_dict[key][0].subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]]})
            func1 = rhs_dict[key][1].subs({nvec[0]: V * PhiDict[nvec[0]] + sympy.sqrt(V) * NoiseDict[nvec[0]], nvec[1]: V * PhiDict[nvec[1]] + sympy.sqrt(V) * NoiseDict[nvec[1]]})
            func2 = rhs_dict[key][2].subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]]})
            func = func1 * func2
            if len(op.args[0].args) == 0:
                term = (op * func).subs({op * func: func + op.args[1] / sympy.sqrt(V) * Derivative(func, op.args[0]) + op.args[1]**2 / (2 * V) * Derivative(func, op.args[0], op.args[0])})
            else:
                term = (op.args[1] * func).subs({op.args[1] * func: func + op.args[1].args[1] / sympy.sqrt(V) * Derivative(func, op.args[1].args[0])
                                                 + op.args[1].args[1]**2 / (2 * V) * Derivative(func, op.args[1].args[0], op.args[1].args[0])})
                term = (op.args[0] * term).subs({op.args[0] * term: term + op.args[0].args[1] / sympy.sqrt(V) * Derivative(term, op.args[0].args[0])
                                                 + op.args[0].args[1]**2 / (2 * V) * Derivative(term, op.args[0].args[0], op.args[0].args[0])})
            # term_num, term_denom = term.as_numer_denom()
            rhs_vKE += rhs_dict[key][3].subs(PhiConstDict) * (term.doit() - func)
    elif len(nvec) == 3:
        lhs_vKE = (Derivative(P(nvec[0], nvec[1], nvec[2], t), t).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]]})
                   - sympy.sqrt(V) * Derivative(PhiDict[nvec[0]], t) * Derivative(P(nvec[0], nvec[1], nvec[2], t), nvec[0]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]]})
                   - sympy.sqrt(V) * Derivative(PhiDict[nvec[1]], t) * Derivative(P(nvec[0], nvec[1], nvec[2], t), nvec[1]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]]})
                   - sympy.sqrt(V) * Derivative(PhiDict[nvec[2]], t) * Derivative(P(nvec[0], nvec[1], nvec[2], t), nvec[2]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]]}))
        rhs_dict, substring = rhs(stoich)
        rhs_vKE = 0
        for key in rhs_dict:
            op = rhs_dict[key][0].subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]]})
            func1 = rhs_dict[key][1].subs({nvec[0]: V * PhiDict[nvec[0]] + sympy.sqrt(V) * NoiseDict[nvec[0]], nvec[1]: V * PhiDict[nvec[1]] + sympy.sqrt(V) * NoiseDict[nvec[1]], nvec[2]: V * PhiDict[nvec[2]] + sympy.sqrt(V) * NoiseDict[nvec[2]]})
            func2 = rhs_dict[key][2].subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]]})
            func = func1 * func2
            if len(op.args[0].args) == 0:
                term = (op * func).subs({op * func: func + op.args[1] / sympy.sqrt(V) * Derivative(func, op.args[0]) + op.args[1]**2 / (2 * V) * Derivative(func, op.args[0], op.args[0])})

            elif len(op.args) == 2:
                term = (op.args[1] * func).subs({op.args[1] * func: func + op.args[1].args[1] / sympy.sqrt(V) * Derivative(func, op.args[1].args[0])
                                                 + op.args[1].args[1]**2 / (2 * V) * Derivative(func, op.args[1].args[0], op.args[1].args[0])})
                term = (op.args[0] * term).subs({op.args[0] * term: term + op.args[0].args[1] / sympy.sqrt(V) * Derivative(term, op.args[0].args[0])
                                                 + op.args[0].args[1]**2 / (2 * V) * Derivative(term, op.args[0].args[0], op.args[0].args[0])})
            elif len(op.args) == 3:
                term = (op.args[2] * func).subs({op.args[2] * func: func + op.args[2].args[1] / sympy.sqrt(V) * Derivative(func, op.args[2].args[0])
                                                 + op.args[2].args[1]**2 / (2 * V) * Derivative(func, op.args[2].args[0], op.args[2].args[0])})
                term = (op.args[1] * term).subs({op.args[1] * term: term + op.args[1].args[1] / sympy.sqrt(V) * Derivative(term, op.args[1].args[0])
                                                 + op.args[1].args[1]**2 / (2 * V) * Derivative(term, op.args[1].args[0], op.args[1].args[0])})
                term = (op.args[0] * term).subs({op.args[0] * term: term + op.args[0].args[1] / sympy.sqrt(V) * Derivative(term, op.args[0].args[0])
                                                 + op.args[0].args[1]**2 / (2 * V) * Derivative(term, op.args[0].args[0], op.args[0].args[0])})
            else:
                print('Something went wrong!')
            rhs_vKE += rhs_dict[key][3].subs(PhiConstDict) * (term.doit() - func)
    else:
        lhs_vKE = (Derivative(P(nvec[0], nvec[1], nvec[2], nvec[3], t), t).subs(
            {nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]], nvec[3]: NoiseDict[nvec[3]]})
            - sympy.sqrt(V) * Derivative(PhiDict[nvec[0]], t) * Derivative(P(nvec[0], nvec[1], nvec[2], nvec[3], t), nvec[0]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]], nvec[3]: NoiseDict[nvec[3]]})
            - sympy.sqrt(V) * Derivative(PhiDict[nvec[1]], t) * Derivative(P(nvec[0], nvec[1], nvec[2], nvec[3], t), nvec[1]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]], nvec[3]: NoiseDict[nvec[3]]})
            - sympy.sqrt(V) * Derivative(PhiDict[nvec[2]], t) * Derivative(P(nvec[0], nvec[1], nvec[2], nvec[3], t), nvec[2]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]], nvec[3]: NoiseDict[nvec[3]]})
            - sympy.sqrt(V) * Derivative(PhiDict[nvec[3]], t) * Derivative(P(nvec[0], nvec[1], nvec[2], nvec[3], t), nvec[3]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]], nvec[3]: NoiseDict[nvec[3]]}))
        rhs_dict, substring = rhs(stoich)
        rhs_vKE = 0
        for key in rhs_dict:
            op = rhs_dict[key][0].subs({nvec[0]: NoiseDict[nvec[0]],
                                        nvec[1]: NoiseDict[nvec[1]],
                                        nvec[2]: NoiseDict[nvec[2]],
                                        nvec[3]: NoiseDict[nvec[3]]})
            func1 = rhs_dict[key][1].subs({nvec[0]: V * PhiDict[nvec[0]] + sympy.sqrt(V) * NoiseDict[nvec[0]],
                                           nvec[1]: V * PhiDict[nvec[1]] + sympy.sqrt(V) * NoiseDict[nvec[1]],
                                           nvec[2]: V * PhiDict[nvec[2]] + sympy.sqrt(V) * NoiseDict[nvec[2]],
                                           nvec[3]: V * PhiDict[nvec[3]] + sympy.sqrt(V) * NoiseDict[nvec[3]]})
            func2 = rhs_dict[key][2].subs({nvec[0]: NoiseDict[nvec[0]],
                                           nvec[1]: NoiseDict[nvec[1]],
                                           nvec[2]: NoiseDict[nvec[2]],
                                           nvec[3]: NoiseDict[nvec[3]]})
            func = func1 * func2
            if len(op.args[0].args) == 0:
                term = (op * func).subs({op * func: func + op.args[1] / sympy.sqrt(V) * Derivative(func, op.args[0]) + op.args[1]**2 / (2 * V) * Derivative(func, op.args[0], op.args[0])})

            elif len(op.args) == 2:
                term = (op.args[1] * func).subs({op.args[1] * func: func + op.args[1].args[1] / sympy.sqrt(V) * Derivative(func, op.args[1].args[0])
                                                 + op.args[1].args[1]**2 / (2 * V) * Derivative(func, op.args[1].args[0], op.args[1].args[0])})
                term = (op.args[0] * term).subs({op.args[0] * term: term + op.args[0].args[1] / sympy.sqrt(V) * Derivative(term, op.args[0].args[0])
                                                 + op.args[0].args[1]**2 / (2 * V) * Derivative(term, op.args[0].args[0], op.args[0].args[0])})
            elif len(op.args) == 3:
                term = (op.args[2] * func).subs({op.args[2] * func: func + op.args[2].args[1] / sympy.sqrt(V) * Derivative(func, op.args[2].args[0])
                                                 + op.args[2].args[1]**2 / (2 * V) * Derivative(func, op.args[2].args[0], op.args[2].args[0])})
                term = (op.args[1] * term).subs({op.args[1] * term: term + op.args[1].args[1] / sympy.sqrt(V) * Derivative(term, op.args[1].args[0])
                                                 + op.args[1].args[1]**2 / (2 * V) * Derivative(term, op.args[1].args[0], op.args[1].args[0])})
                term = (op.args[0] * term).subs({op.args[0] * term: term + op.args[0].args[1] / sympy.sqrt(V) * Derivative(term, op.args[0].args[0])
                                                 + op.args[0].args[1]**2 / (2 * V) * Derivative(term, op.args[0].args[0], op.args[0].args[0])})
            elif len(op.args) == 4:
                term = (op.args[3] * func).subs({op.args[3] * func: func + op.args[3].args[1] / sympy.sqrt(V) * Derivative(func, op.args[3].args[0])
                                                 + op.args[3].args[1]**2 / (2 * V) * Derivative(func, op.args[3].args[0], op.args[3].args[0])})
                term = (op.args[2] * term).subs({op.args[2] * term: term + op.args[2].args[1] / sympy.sqrt(V) * Derivative(term, op.args[2].args[0])
                                                 + op.args[2].args[1]**2 / (2 * V) * Derivative(term, op.args[2].args[0], op.args[2].args[0])})
                term = (op.args[1] * term).subs({op.args[1] * term: term + op.args[1].args[1] / sympy.sqrt(V) * Derivative(term, op.args[1].args[0])
                                                 + op.args[1].args[1]**2 / (2 * V) * Derivative(term, op.args[1].args[0], op.args[1].args[0])})
                term = (op.args[0] * term).subs({op.args[0] * term: term + op.args[0].args[1] / sympy.sqrt(V) * Derivative(term, op.args[0].args[0])
                                                 + op.args[0].args[1]**2 / (2 * V) * Derivative(term, op.args[0].args[0], op.args[0].args[0])})
            else:
                print('Something went wrong!')
            rhs_vKE += rhs_dict[key][3].subs(PhiConstDict) * (term.doit() - func)

    return rhs_vKE.expand(), lhs_vKE, substring


def _pydstoolify(equation) -> str:
    """Utility function to mangle variable names in equations so they are accepted by PyDStool."""
    eq_str = str(equation)

    chars_to_remove = ['{', '}', '_', '\\', '^']

    for c in chars_to_remove:
        eq_str = eq_str.replace(c, '')

    return eq_str

