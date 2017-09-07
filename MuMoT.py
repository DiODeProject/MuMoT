
# coding: utf-8

# In[ ]:

# if has extension .ipynb build this notebook as an importable module using
# ipython nbconvert --to script MuMoT.ipynb

# dependencies:
#  tex distribution
#  libtool (on Mac: brew install libtool --universal; brew link libtool (http://brew.sh))
#  antlr4 (for generating LaTeX parser; http://www.antlr.org)
#  latex2sympy (no current pip installer; https://github.com/augustt198/latex2sympy; depends on antlr4 (pip install antlr4-python3-runtime)
#  graphviz (pip install graphviz; graphviz http://www.graphviz.org/Download.php)


from IPython.display import display, clear_output, Math, Latex, Javascript
import ipywidgets.widgets as widgets
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sympy import *
import PyDSTool as dst
from graphviz import Digraph
from process_latex import process_sympy
import tempfile
import os
import copy
from pyexpat import model
from idlelib.textView import view_file
from IPython.utils import io
import datetime
import warnings
from matplotlib.cbook import MatplotlibDeprecationWarning
from mpl_toolkits.mplot3d import axes3d
import networkx as nx #@UnresolvedImport
from enum import Enum
import ast
#from __builtin__ import None
#from numpy.oldnumeric.fix_default_axis import _args3
#from matplotlib.offsetbox import kwargs

get_ipython().magic('alias_magic model latex')
get_ipython().magic('matplotlib nbagg')

figureCounter = 1 # global figure counter for model views

MAX_RANDOM_SEED = 4294967295

# enum possible Network types
class NetworkType(Enum):
    FULLY_CONNECTED = 0
    ERSOS_RENYI = 1
    BARABASI_ALBERT = 2
    SPACE = 3
    DYNAMIC = 4

# class with default parameters
class MuMoTdefault:
    _initialRateValue = 2 # TODO: was 1 (choose initial values sensibly)
    _rateLimits = (0.0, 20.0) # TODO: choose limit values sensibly
    _rateStep = 0.1 # TODO: choose rate step sensibly
    @staticmethod
    def setRateDefaults(initRate=_initialRateValue, limits=_rateLimits, step=_rateStep):
        MuMoTdefault._initialRateValue = initRate
        MuMoTdefault._rateLimits = limits
        MuMoTdefault._rateStep = step
    
    _maxTime = 10
    _timeLimits = (1, 100)
    _timeStep = 1
    @staticmethod
    def setTimeDefaults(initTime=_maxTime, limits=_timeLimits, step=_timeStep):
        MuMoTdefault._maxTime = initTime
        MuMoTdefault._timeLimits = limits
        MuMoTdefault._timeStep = step
        
    _agents = 100
    _agentsLimits = (0, 1000)
    _agentsStep = 1
    @staticmethod
    def setAgentsDefaults(initAgents=_agents, limits=_agentsLimits, step=_agentsStep):
        MuMoTdefault._agents = initAgents
        MuMoTdefault._agentsLimits = limits
        MuMoTdefault._agentsStep = step
    

class MuMoTmodel: # class describing a model
    _rules = None # list of rules
    _reactants = None # set of reactants
    _systemSize = None # parameter that determines system size, set by using substitute()
    _reactantsLaTeX = None # list of LaTeX strings describing reactants (TODO: depracated?)
    _rates = None # set of rates
    _ratesLaTeX = None # dictionary of LaTeX strings describing rates
    _equations = None # dictionary of ODE righthand sides with reactant as key
    _solutions = None # set of solutions to equations
    _funcs = None # dictionary of lambdified functions for integration, plotting, etc.
    _args = None # tuple of argument symbols for lambdified functions
    _dot = None # graphviz visualisation of model
    _renderImageFormat = 'png' # image format used for rendering edge labels for model visualisation
    _tmpdirpath = '__mumot_files__' # local path for creation of temporary storage
    _tmpdir = None # temporary storage for image files, etc. used in visualising model
    _tmpfiles = None # list of temporary files created
    
    
    def substitute(self, subsString):
        # create new model with variable substitutions listed as comma separated string of assignments
        subs = []
        subsStrings = subsString.split(',')
        for subString in subsStrings:
            if '=' not in subString:
                raise SyntaxError("No '=' in assignment " + subString)
            assignment = process_sympy(subString)
            subs.append((assignment.lhs, assignment.rhs))
        newModel = MuMoTmodel()
        newModel._rules = copy.copy(self._rules)
        newModel._reactants = copy.copy(self._reactants)
        newModel._equations = copy.copy(self._equations)
        for reactant in newModel._reactants:
            for sub in subs:
                newModel._equations[reactant] = newModel._equations[reactant].subs(sub[0], sub[1])
        for rule in newModel._rules:
            for sub in subs:
                rule.rate = rule.rate.subs(sub[0], sub[1])
        for sub in subs:
            if sub[0] in newModel._reactants:
                for atom in sub[1].atoms(Symbol):
                    if atom not in newModel._reactants and atom != self._systemSize:
                        newModel._systemSize = atom
                newModel._reactants.discard(sub[0])
                del newModel._equations[sub[0]]
        if newModel._systemSize == None:
            newModel._systemSize = self._systemSize
        for reactant in newModel._equations:
            rhs = newModel._equations[reactant]
            for symbol in rhs.atoms(Symbol):
                if symbol not in newModel._reactants and symbol != newModel._systemSize:
                    newModel._rates.add(symbol)
        newModel._ratesLaTeX = {}
        rates = map(latex, list(newModel._rates))
        for (rate, latexStr) in zip(newModel._rates, rates):
            newModel._ratesLaTeX[repr(rate)] = latexStr
        # TODO: what else should be copied to new model?

        return newModel


    def visualise(self):
        # build a graphical representation of the model
        # if result cannot be plotted check for installation of libltdl - e.g. on Mac see if XQuartz requires update or do:
        #  brew install libtool --universal
        #  brew link libtool
        if self._dot == None:
            dot = Digraph(comment = "Model", engine = 'circo')
            for reactant in self._reactants:
                dot.node(str(reactant), " ", image = self._localLaTeXimageFile(reactant))
            for rule in self._rules:
                # render LaTeX representation of rule
                localfilename = self._localLaTeXimageFile(rule.rate)
                htmlLabel = r'<<TABLE BORDER="0"><TR><TD><IMG SRC="' + localfilename + r'"/></TD></TR></TABLE>>'
                if len(rule.lhsReactants) == 1:
                    dot.edge(str(rule.lhsReactants[0]), str(rule.rhsReactants[0]), label = htmlLabel)
                elif len(rule.lhsReactants) == 2:
                    # work out source and target of edge, and arrow syle
                    source = None
                    if rule.rhsReactants[0] == rule.rhsReactants[1]:
                        if rule.rhsReactants[0] == rule.lhsReactants[0] or rule.rhsReactants[0] == rule.lhsReactants[1]:
                            # 'recruited switching' motif A + B -> A + A
                            target = str(rule.rhsReactants[0])
                            head = 'normal'
                            tail = 'dot'
                            direction = 'both'
                            if rule.lhsReactants[0] != rule.rhsReactants[0]:
                                source = str(rule.lhsReactants[0])
                            elif rule.lhsReactants[1] != rule.rhsReactants[0]:
                                source = str(rule.lhsReactants[1])
                    else:
                        for i in range(0,2):
                            if rule.rhsReactants[i] != rule.lhsReactants[0] and rule.rhsReactants[i] != rule.lhsReactants[1]:
                                # one of these _reactants is not like the others... found it
                                if rule.rhsReactants[1 - i] == rule.lhsReactants[0]:
                                    # 'targeted inhibition' motif A + B -> C + A
                                    source = str(rule.lhsReactants[0])
                                    target = str(rule.lhsReactants[1])
                                elif rule.rhsReactants[1 - i] == rule.lhsReactants[1]:
                                    # 'targeted inhibition' motif A + B -> C + B
                                    source = str(rule.lhsReactants[1])
                                    target = str(rule.lhsReactants[0])
                        head = 'dot'
                        tail = 'none'

                    if source == None:
                        # 'reciprocal inhibition' motif A + B -> C + C/D
                        source = str(rule.lhsReactants[0])
                        target = str(rule.lhsReactants[1])
                        head = 'dot'
                        tail = 'dot'

                    if source != None:
                        dot.edge(source, target, label = htmlLabel, arrowhead = head, arrowtail = tail, dir = 'both')
            self._dot = dot
                
        return self._dot


    def showReactants(self):
        # show a sorted LaTeX representation of the model's reactants
        if self._reactantsLaTeX == None:
            self._reactantsLaTeX = []
            reactants = map(latex, list(self._reactants))
            for reactant in reactants:
                self._reactantsLaTeX.append(reactant)
            self._reactantsLaTeX.sort()
        for reactant in self._reactantsLaTeX:
            display(Math(reactant))



    def showRates(self):
        # show a sorted LaTeX representation of the model's rate parameters
        for rate in self._ratesLaTeX:
            display(Math(self._ratesLaTeX[rate]))
    
    
    def showODEs(self):
        # show a LaTeX representation of the model system of ODEs
        for reactant in self._reactants:
            out = "\\displaystyle \\frac{\\textrm{d}" + latex(reactant) + "}{\\textrm{d}t} := " + latex(self._equations[reactant])
            display(Math(out))
    
    
    def show(self):
        # show a LaTeX representation of the model
        # if rules have | after them update notebook (allegedly, or switch browser):
        # pip install --upgrade notebook
        for rule in self._rules:
            out = ""
            for reactant in rule.lhsReactants:
                out += latex(reactant)
                out += " + "
            out = out[0:len(out) - 2] # delete the last ' + '
            out += " \\xrightarrow{" + latex(rule.rate) + "}"
            for reactant in rule.rhsReactants:
                out += latex(reactant)
                out += " + "
            out = out[0:len(out) - 2] # delete the last ' + '
            display(Math(out))

        
    def stream(self, stateVariable1, stateVariable2, stateVariable3 = None, **kwargs):
        if self._check_state_variables(stateVariable1, stateVariable2, stateVariable3):
            if stateVariable3 != None:
                print("3d stream plots not currently supported")
                
                return None
            # construct controller
            viewController = self._controller(True, **kwargs)
            
            # construct view
            modelView = MuMoTstreamView(self, viewController, stateVariable1, stateVariable2, **kwargs)
                    
            viewController.setView(modelView)
            viewController.setReplotFunction(modelView._plot_field)         
            
            return viewController
        else:
            return None
        
    def vector(self, stateVariable1, stateVariable2, stateVariable3 = None, **kwargs):
        if self._check_state_variables(stateVariable1, stateVariable2, stateVariable3):
            # construct controller
            viewController = self._controller(True, **kwargs)
            
            # construct view
            modelView = MuMoTvectorView(self, viewController, stateVariable1, stateVariable2, stateVariable3, **kwargs)
                    
            viewController.setView(modelView)
            viewController.setReplotFunction(modelView._plot_field)         
                        
            return viewController
        else:
            return None
        

    def bifurcation(self, bifurcationParameter, stateVariable1, stateVariable2 = None, **kwargs):
        # construct interactive PyDSTool plot
   
        if self._systemSize != None:
            pass
        else:
            print('Cannot construct bifurcation plot until system size is set, using substitute()')
            return    

        paramDict = {}
        initialRateValue = 4 # TODO was 1 (choose initial values sensibly)
        rateLimits = (-100.0, 100.0) # TODO choose limit values sensibly
        rateStep = 0.1 # TODO choose rate step sensibly
        for rate in self._rates:
            paramDict[str(rate)] = initialRateValue # TODO choose initial values sensibly
        paramDict[str(self._systemSize)] = 1 # TODO choose system size sensibly

        # construct controller
        paramValues = []
        paramNames = []        
        for rate in self._rates:
            if str(rate) != bifurcationParameter:
                paramValues.append((initialRateValue, rateLimits[0], rateLimits[1], rateStep))
                paramNames.append(str(rate))
        viewController = MuMoTcontroller(paramValues, paramNames, self._ratesLaTeX, False)

        # construct view
        modelView = MuMoTbifurcationView(self, viewController, paramDict, bifurcationParameter, stateVariable1, stateVariable2, **kwargs)
        
        viewController.setView(modelView)
        viewController.setReplotFunction(modelView._replot_bifurcation)
        
        return viewController

    def multiagent(self, netType="full", initialState="Auto", maxTime="Auto", randomSeed="Auto", **kwargs):
        specialParams = {}
        if initialState=="Auto":
            #initialState = [100] + [0]*(len(reactantList)-1)
            first = True
            initialState = {}
            for reactant in self._reactants:
                if first:
                    print("Automatic Initial State sets " + str(MuMoTdefault._agents) + " agents in state " + str(reactant) )
                    initialState[reactant] = MuMoTdefault._agents
                    first = False
                else:
                    initialState[reactant] = 0
        else:
            print("TO-DO: check if the Initial State has valid length and positive values")
        print("Initial State is " + str(initialState) )
        specialParams['initialState'] = initialState
        
        # construct controller
        paramValues = []
        paramNames = [] 
        #paramValues.extend( [initialState, netType, maxTime] )
        for rate in self._rates:
            paramValues.append((MuMoTdefault._initialRateValue, MuMoTdefault._rateLimits[0], MuMoTdefault._rateLimits[1], MuMoTdefault._rateStep))
            paramNames.append(str(rate))

        if (maxTime == "Auto" or maxTime <= 0):
            maxTime = MuMoTdefault._maxTime
            timeLimitMax = max(maxTime, MuMoTdefault._timeLimits[1])
        paramNames.append("maxTime")
        paramValues.append( (maxTime, MuMoTdefault._timeLimits[0], timeLimitMax, MuMoTdefault._timeStep) )
        if (randomSeed == "Auto" or randomSeed <= 0):
            specialParams['randomSeed'] = np.random.randint(MAX_RANDOM_SEED)
            print("Automatic Random Seed set to " + str(specialParams['randomSeed']) )
        viewController = MuMoTmultiagentController(paramValues, paramNames, self._ratesLaTeX, False, specialParams)
        
        paramDict = {}
#         paramDict['reactants'] = self._reactants
#         paramDict['rules'] = self._rules
#         paramDict['maxTime'] = maxTime
        paramDict['initialState'] = initialState
        paramDict['netType'] = netType
        modelView = MuMoTmultiagentView(self, viewController, paramDict, **kwargs)
        viewController.setView(modelView)
        modelView._plot_timeEvolution()
#         viewController.setReplotFunction(modelView._plot_timeEvolution(self._reactants, self._rules))
        viewController.setReplotFunction(modelView._plot_timeEvolution)
        
        return viewController

    def SSA(self, initialState="Auto", maxTime="Auto", randomSeed="Auto", **kwargs):
        specialParams = {}
        if initialState=="Auto":
            first = True
            initialState = {}
            for reactant in self._reactants:
                if first:
                    print("Automatic Initial State sets " + str(MuMoTdefault._agents) + " agents in state " + str(reactant) )
                    initialState[reactant] = MuMoTdefault._agents
                    first = False
                else:
                    initialState[reactant] = 0
        else:
            print("TODO: check if the Initial State has valid length and positive values")
#             initialState = ast.literal_eval(initialState) # translate from string to dict
#             print(initialState)
#             print(self._reactants)
#             for state in initialState.keys():
#                 print(state)
#                 print(str(state) + " == " + str(self._reactants) + " " + str(state in self._reactants) )
#             for state in self._reactants:
#                 print(state)
#                 print(str(state) + " == " + str(initialState.keys()) + " " + str( "'"+str(state)+"'" in initialState.keys()) )
            initialState = {}
            first = True
            for reactant in self._reactants:
                if first:
                    print("Automatic Initial State sets " + str(MuMoTdefault._agents) + " agents in state " + str(reactant) )
                    initialState[reactant] = MuMoTdefault._agents
                    first = False
                else:
                    initialState[reactant] = 0
#             print(initialState)
#             print(initialState.keys())
        print("Initial State is " + str(initialState) )
        specialParams['initialState'] = initialState
        
        # construct controller
        paramValues = []
        paramNames = [] 
        #paramValues.extend( [initialState, netType, maxTime] )
        for rate in self._rates:
            paramValues.append((MuMoTdefault._initialRateValue, MuMoTdefault._rateLimits[0], MuMoTdefault._rateLimits[1], MuMoTdefault._rateStep))
            paramNames.append(str(rate))

        if (maxTime == "Auto" or maxTime <= 0):
            maxTime = MuMoTdefault._maxTime
            timeLimitMax = max(maxTime, MuMoTdefault._timeLimits[1])
        paramNames.append("maxTime")
        paramValues.append( (maxTime, MuMoTdefault._timeLimits[0], timeLimitMax, MuMoTdefault._timeStep) )
        if (randomSeed == "Auto" or randomSeed <= 0):
            specialParams['randomSeed'] = np.random.randint(MAX_RANDOM_SEED)
            print("Automatic Random Seed set to " + str(specialParams['randomSeed']) )
        viewController = MuMoTSSAController(paramValues, paramNames, self._ratesLaTeX, False, specialParams)
        
        paramDict = {}
        paramDict['initialState'] = initialState
        modelView = MuMoTSSAView(self, viewController, paramDict, **kwargs)
        viewController.setView(modelView)
        modelView._plot_timeEvolution()
        
#         viewController.setReplotFunction(modelView._plot_timeEvolution(self._reactants, self._rules))
        viewController.setReplotFunction(modelView._plot_timeEvolution)
        
        return viewController

#         if stateVariable2 == None:
#             # 2-d bifurcation diagram
#             # create widgets
# 
#             if self._systemSize != None:
#                 # TODO: shouldn't allow system size to be varied?
#                 pass
# #                self._paramValues.append(1)
# #                self._paramNames.append(str(self._systemSize))
# #                widget = widgets.FloatSlider(value = 1, min = _rateLimits[0], max = _rateLimits[1], step = _rateStep, description = str(self._systemSize), continuous_update = False)
# #                widget.on_trait_change(self._replot_bifurcation2D, 'value')
# #                self._widgets.append(widget)
# #                display(widget)
#             else:
#                 print('Cannot attempt bifurcation plot until system size is set, using substitute()')
#                 return
#             for rate in self._rates:
#                 if str(rate) != bifurcationParameter:
#                     self._paramValues.append(_initialRateValue)
#                     self._paramNames.append(str(rate))
#                     widget = widgets.FloatSlider(value = _initialRateValue, min = _rateLimits[0], max = _rateLimits[1], step = _rateStep, description = str(rate), continuous_update = False)
#                     widget.on_trait_change(self._replot_bifurcation2D, 'value')
#                     self._widgets.append(widget)
#                     display(widget)
#             widget = widgets.HTML(value = '')
#             self._errorMessage = widget                                # TODO: add to __init__()
#             display(self._errorMessage)
#             
#             # Prepare the system to start close to a steady state
#             self._bifurcationParameter = bifurcationParameter          # TODO: remove hack (bifurcation parameter for multiple possible bifurcations needs to be stored in self)
#             self._stateVariable1 = stateVariable1                      # TODO: remove hack (state variable for multiple possible bifurcations needs to be stored in self)
# #            self._pyDSode.set(pars = {bifurcationParameter: 0} )       # Lower bound of the bifurcation parameter (TODO: set dynamically)
# #            self._pyDSode.set(pars = self._pyDSmodel.pars )       # Lower bound of the bifurcation parameter (TODO: set dynamically)
# #            self._pyDSode.pars = {bifurcationParameter: 0}             # Lower bound of the bifurcation parameter (TODO: set dynamically?)
#             initconds = {stateVariable1: self._paramDict[str(self._systemSize)] / len(self._reactants)} # TODO: guess where steady states are?
#             for reactant in self._reactants:
#                 if str(reactant) != stateVariable1:
#                     initconds[str(reactant)] = self._paramDict[str(self._systemSize)] / len(self._reactants)
# #            self._pyDSmodel.ics = initconds
#             self._pyDSmodel.ics      = {'A': 0.1, 'B': 0.9 }    # TODO: replace            
# #            self._pyDSode.set(ics = initconds)
#             self._pyDSode = dst.Generator.Vode_ODEsystem(self._pyDSmodel)  # TODO: add to __init__()
#             self._pyDSode.set(pars = {bifurcationParameter: 5} )                       # TODO remove magic number
#             self._pyDScont = dst.ContClass(self._pyDSode)              # Set up continuation class (TODO: add to __init__())
#             # TODO: add self._pyDScontArgs to __init__()
#             self._pyDScontArgs = dst.args(name='EQ1', type='EP-C')     # 'EP-C' stands for Equilibrium Point Curve. The branch will be labeled 'EQ1'.
#             self._pyDScontArgs.freepars     = [bifurcationParameter]   # control parameter(s) (should be among those specified in self._pyDSmodel.pars)
#             self._pyDScontArgs.MaxNumPoints = 450                      # The following 3 parameters are set after trial-and-error TODO: how to automate this?
#             self._pyDScontArgs.MaxStepSize  = 1e-1
#             self._pyDScontArgs.MinStepSize  = 1e-5
#             self._pyDScontArgs.StepSize     = 2e-3
#             self._pyDScontArgs.LocBifPoints = ['LP', 'BP']                    # TODO WAS 'LP' (detect limit points / saddle-node bifurcations)
#             self._pyDScontArgs.SaveEigen    = True                     # to tell unstable from stable branches
# #            self._pyDScontArgs.CalcStab     = True
# 
#             plt.ion()
# #            self._bifurcation2Dfig = plt.figure(1)                     # TODO: add to __init__()
#             self._pyDScont.newCurve(self._pyDScontArgs)
#             try:
#                 try:
#                     self._pyDScont['EQ1'].backward()
#                 except:
#                     self._errorMessage.value = 'Continuation failure'
#                 try:
#                     self._pyDScont['EQ1'].forward()                                  # TODO: how to choose direction?
#                 except:
#                     self._errorMessage.value = 'Continuation failure'
#                     self._errorMessage.value = ''
#             except ZeroDivisionError:
#                 self._errorMessage.value = 'Division by zero'
# #            self._pyDScont['EQ1'].info()
#             self._pyDScont.display([bifurcationParameter, stateVariable1], stability = True, figure = 1)
#             self._pyDScont.plot.fig1.axes1.axes.set_title('Bifurcation Diagram')
#         else:
#             # 3-d bifurcation diagram
#             assert false

    def _get_solutions(self):
        if self._solutions == None:
            self._solutions = solve(iter(self._equations.values()), self._reactants, force = False, positive = False, set = False)
        return self._solutions

    def _controller(self, contRefresh, **kwargs):
        # general controller constructor with all rates as free parameters

        initialRateValue = 4 # TODO was 1 (choose initial values sensibly)
        rateLimits = (-100.0, 100.0) # TODO choose limit values sensibly
        rateStep = 0.1 # TODO choose rate step sensibly                

        # construct controller
        paramValues = []
        paramNames = []        
        for rate in self._rates:
            paramValues.append((initialRateValue, rateLimits[0], rateLimits[1], rateStep))
            paramNames.append(str(rate))
        viewController = MuMoTcontroller(paramValues, paramNames, self._ratesLaTeX, contRefresh)

        return viewController

    def _check_state_variables(self, stateVariable1, stateVariable2, stateVariable3 = None):
        if Symbol(stateVariable1) in self._reactants and Symbol(stateVariable2) in self._reactants and (stateVariable3 == None or Symbol(stateVariable3) in self._reactants):
            if stateVariable1 != stateVariable2 and stateVariable1 != stateVariable3 and stateVariable2 != stateVariable3:
                return True
            else:
                print('State variables cannot be the same')
                return False
        else:
            print('Invalid reactant provided as state variable')
            return False


    def _getFuncs(self):
        # lambdify sympy equations for numerical integration, plotting, etc.
        if self._systemSize == None:
            assert false #TODO is this necessary?
        if self._funcs == None:
            argList = []
            for reactant in self._reactants:
                argList.append(reactant)
            for rate in self._rates:
                argList.append(rate)
            argList.append(self._systemSize)
            self._args = tuple(argList)
            self._funcs = {}
            for equation in self._equations:
                f = lambdify(self._args, self._equations[equation], "math")
                self._funcs[equation] = f
            
        return self._funcs
    
    def _getArgTuple2d(self, argNames, argValues, argDict, stateVariable1, stateVariable2, X, Y):
        # get tuple to evalute functions returned by _getFuncs with, for 2d field-based plots
        argList = []
        for arg in self._args:
            if arg == stateVariable1:
                argList.append(X)
            elif arg == stateVariable2:
                argList.append(Y)
            elif arg == self._systemSize:
                argList.append(1) # TODO: system size set to 1
            else:
                argList.append(argDict[arg])
            
        return tuple(argList)

    def _getArgTuple3d(self, argNames, argValues, argDict, stateVariable1, stateVariable2, stateVariable3, X, Y, Z):
        # get tuple to evalute functions returned by _getFuncs with, for 2d field-based plots
        argList = []
        for arg in self._args:
            if arg == stateVariable1:
                argList.append(X)
            elif arg == stateVariable2:
                argList.append(Y)
            elif arg == stateVariable3:
                argList.append(Z)                
            elif arg == self._systemSize:
                argList.append(1) # TODO: system size set to 1
            else:
                argList.append(argDict[arg])
            
        return tuple(argList)

    def _getArgTuple(self, argNames, argValues, argDict, reactants, reactantValues):
        # get tuple to evalute functions returned by _getFuncs with
        assert false # need to work this out
        argList = []
        for arg in self._args:
            if arg == stateVariable1:
                argList.append(X)
            elif arg == stateVariable2:
                argList.append(Y)
            elif arg == stateVariable3:
                argList.append(Z)                
            elif arg == self._systemSize:
                argList.append(1) # TODO: system size set to 1
            else:
                argList.append(argDict[arg])
            
        return tuple(argList)


                                 
    def _localLaTeXimageFile(self, source):
        # render LaTeX source to local image file
        tmpfile = tempfile.NamedTemporaryFile(dir = self._tmpdir.name, suffix = '.' + self._renderImageFormat, delete = False)
        self._tmpfiles.append(tmpfile)
        preview(source, euler = False, output = self._renderImageFormat, viewer = 'file', filename = tmpfile.name)

        return tmpfile.name[tmpfile.name.find(self._tmpdirpath):]  

#     def _toHTML(self, label):
#         # DEPRECATED
#         errorLabel = label
#         htmlLabel = r'<<I>'
#         parts = label.split('_')
#         label = parts[0]
#         if label[0] == '\\':
#             label = label.replace('\\','&')
#             label += ';'
# 
#         if len(parts) > 1:
#             subscript = parts[1]
#             if len(parts) > 2:
#                 raise SyntaxError("Nested subscripts not currently supported: " + subscript + " in " + errorLabel)
#                 return
#             if len(subscript) > 1:
#                 if  subscript[0] == '{' and subscript[len(subscript) - 1] == '}':
#                     subscript = subscript[1:-1]
#                     if '{' in subscript or '}' in subscript:
#                         raise SyntaxError("Malformed subscript: {" + subscript + "} in " + errorLabel)
#                         return
#                 else:
#                     raise SyntaxError("Non single-character subscript not {} delimited: " + subscript + " in " + errorLabel)
#                     return
#             htmlLabel += label + "<SUB>" + subscript + "</SUB>" + r'</I>>'
#         else:
#             htmlLabel += label + r'</I>>'
#         
#         return htmlLabel
    

    def __init__(self):
        self._rules = []
        self._reactants = set()
        self._systemSize = None
        self._reactantsLaTeX = None
        self._rates = set()
        self._ratesLaTeX= None
        self._equations = {}
        self._pyDSmodel = None
        self._dot = None
        # TODO the following not currently working on OS X
#        if not os.path.idsir(self._tmpdirpath):
#            os.mkdir(self._tmpdirpath)
#            os.system('chmod' + self._tmpdirpath + 'u+rwx')
#            os.system('chmod' + self._tmpdirpath + 'g-rwx')
#            os.system('chmod' + self._tmpdirpath + 'o+rwx')
        self._tmpdir = tempfile.TemporaryDirectory(dir = self._tmpdirpath)
        self._tmpfiles = []
        
    def __del__(self):
        # TODO: check when this is invoked
        for tmpfile in self._tmpfiles:
            del tmpfile
        del self._tmpdir

class _Rule: # class describing a single reaction rule
    lhsReactants = []
    rhsReactants = []
    rate = ""
    def __init__(self):
        self.lhsReactants = []
        self.rhsReactants = []
        self.rate = ""

class MuMoTcontroller: # class describing a controller for a model view
    _view = None
    _paramValues = None
    _paramNames = None
    _paramLabelDict = None
    _widgets = None
    _widgetDict = None
    _errorMessage = None
    _progressBar = None
    _progressBar_multi = None # used two progress bars, otherwise the previous cell bar (where controller is created) does not react anymore 

    def __init__(self, paramValues, paramNames, paramLabelDict, continuousReplot):
        self._paramValues = []
        self._paramNames = []
        self._paramLabelDict = paramLabelDict
        self._widgets = []
        
#         self._ratesDict = {}
        self._widgetDict = {}
        for pair in zip(paramNames, paramValues):
            widget = widgets.FloatSlider(value = pair[1][0], min = pair[1][1], 
                                         max = pair[1][2], step = pair[1][3], 
                                         description = r'\(' + self._paramLabelDict.get(pair[0],pair[0]) + r'\)', 
                                         continuous_update = continuousReplot)
            self._widgetDict[pair[0]] = widget
            # widget.on_trait_change(replotFunction, 'value')
            self._widgets.append(widget)
            display(widget)
#             self._ratesDict[pair[0]] = pair[1][0]
        widget = widgets.HTML(value = '')
        self._errorMessage = widget
        display(self._errorMessage)
        self._paramNames = paramNames
        for triple in paramValues:
            self._paramValues.append(triple[0])
            
    def setReplotFunction(self, replotFunction):
        for widget in self._widgets:
            widget.on_trait_change(replotFunction, 'value')

    def setView(self, view):
        self._view = view

    def showLogs(self):
        self._view.showLogs()
        
    def multirun(self, iterations, randomSeeds="Auto", plotType="evo", downloadData=False):
        # Creating the progress bar (useful to give user progress status for long executions)
        self._progressBar_multi = widgets.FloatProgress(
            value=0,
            min=0,
            max=iterations,
            step=1,
            description='Loading:',
            bar_style='success', # 'success', 'info', 'warning', 'danger' or ''
            orientation='horizontal'
        )
        display(self._progressBar_multi)
        self._view._plotType = plotType
        
        # setting the "Auto" value to the random seeds:
        if randomSeeds == "Auto":
            randomSeeds = []
            for _ in range(iterations):
                randomSeeds.append(np.random.randint(MAX_RANDOM_SEED))
            print("Automatic Random Seeds set to " + str(randomSeeds) )
        else: # checking if the length of the randomSeeds list is the same of iterations
            if not len(randomSeeds) == iterations:
                print("ERROR! Invalid randomSeeds value. The randomSeeds must be a integer list with the length = iterations")
                return
            if sum(x < 0 or x > MAX_RANDOM_SEED or not isinstance( x, int ) for x in randomSeeds) > 0:
                print("ERROR! Invalid randomSeeds value. The randomSeeds must be integers between 0 and " + str(MAX_RANDOM_SEED))
                return
        
        dataRes = self._view._multirun(iterations, randomSeeds)
        
        self._progressBar_multi.value = self._progressBar_multi.max
        self._progressBar_multi.description = "Completed 100%:"
        
        if downloadData:
            return self._downloadFile(dataRes)

    def _update_params_from_widgets(self):
        for i in np.arange(0, len(self._paramValues)):
            # UGLY!
            self._paramValues[i] = self._widgets[i].value
            
    def _downloadFile(self, data_to_download):
        js_download = """
        var csv = '%s';
        
        var filename = 'results.csv';
        var blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        if (navigator.msSaveBlob) { // IE 10+
            navigator.msSaveBlob(blob, filename);
        } else {
            var link = document.createElement("a");
            if (link.download !== undefined) { // feature detection
                // Browsers that support HTML5 download attribute
                var url = URL.createObjectURL(blob);
                link.setAttribute("href", url);
                link.setAttribute("download", filename);
                link.style.visibility = 'hidden';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
        }
        """ % str(data_to_download)
        #str(data_to_download) 
        #data_to_download.to_csv(index=False).replace('\n','\\n').replace("'","\'")
        
        return Javascript(js_download)

class MuMoTSSAController(MuMoTcontroller): # class describing a controller for multiagent views
    _fileToDownload = None
    _initialState = None

    
    def __init__(self, paramValues, paramNames, paramLabelDict, continuousReplot, specialParams):
        MuMoTcontroller.__init__(self, paramValues, paramNames, paramLabelDict, continuousReplot)
        advancedWidgets = []
        
        self._initialState = specialParams['initialState']
        for state,pop in self._initialState.items():
            widget = widgets.IntSlider(value = pop,
                                         min = min(pop, MuMoTdefault._agentsLimits[0]), 
                                         max = max(pop, MuMoTdefault._agentsLimits[1]),
                                         step = MuMoTdefault._agentsStep,
                                         description = "State " + str(state), 
                                         continuous_update = continuousReplot)
            self._widgetDict['init'+str(state)] = widget
            self._widgets.append(widget)
            advancedWidgets.append(widget)
            
        ## Random seed input field
        widget = widgets.IntText(
            value=specialParams['randomSeed'],
            description='Random seed:',
            disabled=False
        )
        self._widgetDict['randomSeed'] = widget
        self._widgets.append(widget)
        advancedWidgets.append(widget)
        
        ## Toggle buttons for plotting style 
        plotToggle = widgets.ToggleButtons(
            options=[('Temporal evolution', 'evo'), ('Final distribution', 'final')],
            description='Plot:',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Population change over time', 'Population distribution in each state at final timestep'],
        #     icons=['check'] * 3
        )
        self._widgetDict['plotType'] = plotToggle
        self._widgets.append(plotToggle)
        advancedWidgets.append(plotToggle)
        
        advancedPage = widgets.Box(children=advancedWidgets)
        advancedOpts = widgets.Accordion(children=[advancedPage])
        advancedOpts.set_title(0, 'Advanced options')
        display(advancedOpts)
        
        # Loading bar (useful to give user progress status for long executions)
        self._progressBar = widgets.FloatProgress(
            value=0,
            min=0,
            max=self._widgetDict['maxTime'].value,
            #step=1,
            description='Loading:',
            bar_style='success', # 'success', 'info', 'warning', 'danger' or ''
            orientation='horizontal'
        )
        display(self._progressBar)
        
    
        
    def _update_params_from_widgets(self): # overwritten parent's method. Old method is not used becuase substituted by widgetDict
        for state in self._initialState.keys():
            self._initialState[state] = self._widgetDict['init'+str(state)].value
        #numNodes = sum(self._initialState.values())
        np.random.seed(self._widgetDict['randomSeed'].value)

class MuMoTmultiagentController(MuMoTcontroller): # class describing a controller for multiagent views
    _fileToDownload = None
    _initialState = None

    
    def __init__(self, paramValues, paramNames, paramLabelDict, continuousReplot, specialParams):
        MuMoTcontroller.__init__(self, paramValues, paramNames, paramLabelDict, continuousReplot)
        advancedWidgets = []
        
        self._initialState = specialParams['initialState']
        for state,pop in self._initialState.items():
            widget = widgets.IntSlider(value = pop,
                                         min = min(pop, MuMoTdefault._agentsLimits[0]), 
                                         max = max(pop, MuMoTdefault._agentsLimits[1]),
                                         step = MuMoTdefault._agentsStep,
                                         description = "State " + str(state), 
                                         continuous_update = continuousReplot)
            self._widgetDict['init'+str(state)] = widget
            self._widgets.append(widget)
            advancedWidgets.append(widget)
            
        ## Network type dropdown selector
        netDropdown = widgets.Dropdown( 
            options=[('Full graph', NetworkType.FULLY_CONNECTED), 
                     ('Erdos-Renyi', NetworkType.ERSOS_RENYI),
                     ('Barabasi-Albert', NetworkType.BARABASI_ALBERT),
                     # TODO: add network topology generated by random points in space 
                     ('Moving particles', NetworkType.DYNAMIC)
                     ],
            description='Network topology:',
            value = NetworkType.FULLY_CONNECTED, 
            disabled=False
        )
        self._widgetDict['netType'] = netDropdown
        #self._widgets.append(netDropdown)
        netDropdown.on_trait_change(self._update_net_params, 'value')
        advancedWidgets.append(netDropdown)
        
        ## Network connectivity slider
        widget = widgets.FloatSlider(value = 0,
                                    min = 0, 
                                    max = 1,
                            description = 'Network connectivity parameter', 
                            continuous_update = continuousReplot,
                            disabled=True
        )
        self._widgetDict['netParam'] = widget
        self._widgets.append(widget)
        advancedWidgets.append(widget)
        
        ## Agent speed
        widget = widgets.FloatSlider(value = 0.01, min = 0, max = 1, step=0.01,
                            description = 'Particle speed', 
                            continuous_update = continuousReplot,
                            disabled=True
        )
        self._widgetDict['speed'] = widget
        self._widgets.append(widget)
        advancedWidgets.append(widget)
        
        ## Random walk correlatedness
        widget = widgets.FloatSlider(value = 0.5, min = 0, max = 1, step=0.01,
                            description = 'Correlatedness of the random walk', 
                            continuous_update = continuousReplot,
                            disabled=True
        )
        self._widgetDict['correlated'] = widget
        self._widgets.append(widget)
        advancedWidgets.append(widget)
        
        ## Random seed input field
        widget = widgets.IntText(
            value=specialParams['randomSeed'],
            description='Random seed:',
            disabled=False
        )
        self._widgetDict['randomSeed'] = widget
        self._widgets.append(widget)
        advancedWidgets.append(widget)
        #display(widget)
        
        ## Time scaling slider
        widget =  widgets.FloatSlider(value = 1,
                                    min = 0.001, 
                                    max = 1,
                            description = 'Time scaling', 
                            continuous_update = continuousReplot
        )
        self._widgetDict['scaling'] = widget
        self._widgets.append(widget)
        advancedWidgets.append(widget)
        
        ## Toggle buttons for plotting style 
        plotToggle = widgets.ToggleButtons(
            #options={'Temporal evolution' : 'evo', 'Network' : 'graph', 'Final distribution' : 'final'},
            options=[('Temporal evolution','evo'), ('Network','graph'), ('Final distribution', 'final')],
            description='Plot:',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Population change over time', 'Network topology', 'Number of agents in each state at final timestep'],
        #     icons=['check'] * 3
        )
        self._widgetDict['plotType'] = plotToggle
        self._widgets.append(plotToggle)
        advancedWidgets.append(plotToggle)
        
        ## Particle display checkboxes
        widget = widgets.Checkbox(
            value=True,
            description='Show particle trace',
            disabled = not (self._widgetDict['netType'].value == NetworkType.DYNAMIC)
        )
        self._widgetDict['trace'] = widget
        self._widgets.append(widget)
        advancedWidgets.append(widget)
        widget = widgets.Checkbox(
            value=True,
            description='Show communication links',
            disabled = not (self._widgetDict['netType'].value == NetworkType.DYNAMIC)
        )
        self._widgetDict['interactions'] = widget
        self._widgets.append(widget)
        advancedWidgets.append(widget)
        
        self._update_net_params()
        
        advancedPage = widgets.Box(children=advancedWidgets)
        advancedOpts = widgets.Accordion(children=[advancedPage])
        advancedOpts.set_title(0, 'Advanced options')
        display(advancedOpts)        
        
        # Loading bar (useful to give user progress status for long executions)
        self._progressBar = widgets.IntProgress(
            value=0,
            min=0,
            max=self._widgetDict['maxTime'].value,
            step=1,
            description='Loading:',
            bar_style='success', # 'success', 'info', 'warning', 'danger' or ''
            orientation='horizontal'
        )
        display(self._progressBar)
    
    def _update_params_from_widgets(self): # overwritten parent's method. Old method is not used becuase substituted by widgetDict
        for state in self._initialState.keys():
            self._initialState[state] = self._widgetDict['init'+str(state)].value
        numNodes = sum(self._initialState.values())
        np.random.seed(self._widgetDict['randomSeed'].value)
        netParam = self._widgetDict['netParam'].value
        self._view._initGraph(graphType=self._widgetDict['netType'].value, numNodes=numNodes, netParam=netParam)
    
    def _update_net_params(self):
        # oder of assignment is important (firt, update the min and max, later, the value)
        # TODO: the update of value happens two times (changing min-max and value) and therefore the calculatio are done two times
        if (self._widgetDict['netType'].value == NetworkType.FULLY_CONNECTED):
            self._widgetDict['netParam'].min = 0
            self._widgetDict['netParam'].max = 1
            self._widgetDict['netParam'].step = 1
            self._widgetDict['netParam'].value = 0
            self._widgetDict['netParam'].disabled = True
            self._widgetDict['netParam'].description = "None"
        elif (self._widgetDict['netType'].value == NetworkType.ERSOS_RENYI):
            self._widgetDict['netParam'].disabled = False
            self._widgetDict['netParam'].min = 0.1
            self._widgetDict['netParam'].max = 1
            self._widgetDict['netParam'].step = 0.1
            self._widgetDict['netParam'].value = 0.5
            self._widgetDict['netParam'].description = "Network connectivity parameter (link probability)"
        elif (self._widgetDict['netType'].value == NetworkType.BARABASI_ALBERT):
            self._widgetDict['netParam'].disabled = False
            maxVal = sum(self._initialState.values())-1
            self._widgetDict['netParam'].min = 1
            self._widgetDict['netParam'].max = maxVal
            self._widgetDict['netParam'].step = 1
            self._widgetDict['netParam'].value = min(maxVal, 3)
            self._widgetDict['netParam'].description = "Network connectivity parameter (new edges)"            
        elif (self._widgetDict['netType'].value == NetworkType.SPACE):
            self._widgetDict['netParam'].value = -1
        
        if (self._widgetDict['netType'].value == NetworkType.DYNAMIC):
            self._widgetDict['netParam'].disabled = False
            self._widgetDict['netParam'].min = 0.0
            self._widgetDict['netParam'].max = 1
            self._widgetDict['netParam'].step = 0.05
            self._widgetDict['netParam'].value = 0.1
            self._widgetDict['netParam'].description = "Interaction range"
            self._widgetDict['speed'].disabled = False
            self._widgetDict['correlated'].disabled = False
            self._widgetDict['trace'].disabled = False
            self._widgetDict['interactions'].disabled = False
        else:
            self._widgetDict['speed'].disabled = True
            self._widgetDict['correlated'].disabled = True
            self._widgetDict['trace'].disabled = True
            self._widgetDict['interactions'].disabled = True
    
    def _update_scaling_widget(self, scaling):
        if (self._widgetDict['scaling'].value > scaling):
            self._widgetDict['scaling'].value = scaling  
        if (self._widgetDict['scaling'].max > scaling):
            self._widgetDict['scaling'].max = scaling
            self._widgetDict['scaling'].min = scaling/100
            self._widgetDict['scaling'].step = scaling/100 
    
    def downloadTimeEvolution(self):
        return self._downloadFile(self._fileToDownload)
    



class MuMoTview: # class describing a view on a model
    _mumotModel = None
    _figure = None
    _figureNum = None
    _widgets = None
    _controller = None
    _logs = None
    _plotType = None
    
    def __init__(self, model, controller):
        global figureCounter
        self._figureNum = figureCounter
        figureCounter += 1
        self._mumotModel = model
        self._controller = controller
        self._logs = []
        
        plt.ion()
        with warnings.catch_warnings(): # ignore warnings when plt.hold has been deprecated in installed libraries - still need to try plt.hold(True) in case older libraries in use
            warnings.filterwarnings("ignore",category=MatplotlibDeprecationWarning)
            warnings.filterwarnings("ignore",category=UserWarning)
            plt.hold(True)  
        self._figure = plt.figure(self._figureNum) 

    def _log(self, analysis):
        print("Starting", analysis, "with parameters ", end='')
        for i in zip(self._controller._paramNames, self._controller._paramValues):
            print('(' + i[0] + '=' + repr(i[1]) + '), ', end='')
        print("at", datetime.datetime.now())
        
                        
    def showLogs(self):
        for log in self._logs:
            log.show()
            
    def _multirun(self, iterations, randomSeeds):
        colors = cm.rainbow(np.linspace(0, 1, len( self._mumotModel._reactants ) ))  # @UndefinedVariable
        colorMap = {}
        i = 0
        for state in sorted(self._mumotModel._reactants, key=str):
            colorMap[state] = colors[i] 
            i += 1
        
        allEvos = []
        for i in range(iterations):
    #         print("Iteration n." + str(i+1) )
            self._controller._progressBar_multi.value = i
            self._controller._progressBar_multi.description = "Loading " + str(round(i/iterations*100)) + "%:"
            
            logs = self._singleRun(randomSeeds[i])
            if not logs: return # multirun is not supported for this controller
            allEvos.append(logs[1])
            
        ## Plot
        global figureCounter
        plt.figure(figureCounter)
        plt.clf()
        figureCounter += 1
        if (self._plotType == "evo"):
            if not hasattr(self._controller, '_initialState'):
                print("ERROR! in multirun arguments. The specified controller does not have the attribute _initialState which is required for plotType 'evo'.")
                return
            systemSize = sum(self._controller._initialState.values())
            maxTime = self._controller._widgetDict['maxTime'].value
            plt.axis([0, maxTime, 0, systemSize])
            
            for evo in allEvos:
                for state,pop in evo.items():
                    if (state == 'time'): continue
                    plt.plot(evo['time'], pop, color=colorMap[state]) #label=state,
                    
        markers = [plt.Line2D([0,0],[0,0],color=color, marker='', linestyle='-') for color in colorMap.values()]
        plt.legend(markers, colorMap.keys(), bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)
        
        return allEvos
    
    ## overwrite this method for views that allow the 'multirun' command
    def _singleRun(self, randomSeed):
        print("ERROR! The command multirun is not supported for this view.")
        return None

class MuMoTfieldView(MuMoTview): # field view on model (specialised by MuMoTvectorView and MuMoTstreamView)
    _stateVariable1 = None # 1st state variable (x-dimension)
    _stateVariable2 = None # 2nd state variable (y-dimension)
    _stateVariable3 = None # 3rd state variable (z-dimension)
    _X = None # X ordinates array
    _Y = None # Y ordinates array
    _Z = None # Z ordinates array
    _X = None # X derivatives array
    _Y = None # Y derivatives array
    _Z = None # Z derivatives array
    _speed = None # speed array
    _mask = {} # class-global dictionary of memoised masks with (mesh size, dimension) as key
    
    def __init__(self, model, controller, stateVariable1, stateVariable2, stateVariable3 = None, **kwargs):
        super().__init__(model, controller)

        self._stateVariable1 = Symbol(stateVariable1)
        self._stateVariable2 = Symbol(stateVariable2)
        if stateVariable3 != None:
            self._stateVariable3 = Symbol(stateVariable3)
        _mask = {}

        self._plot_field()
            

    def _plot_field(self):
        plt.figure(self._figureNum)
        plt.clf()
    
    def _get_field2d(self, kind, meshPoints):
        self._controller._update_params_from_widgets()
        with io.capture_output() as log:
            self._log(kind)
            funcs = self._mumotModel._getFuncs()
            argNamesSymb = list(map(Symbol, self._controller._paramNames))
            argDict = dict(zip(argNamesSymb, self._controller._paramValues))
            self._Y, self._X = np.mgrid[0:1:complex(0, meshPoints), 0:1:complex(0, meshPoints)] # TODO system size defined to be one
            mask = self._mask.get((meshPoints, 2))
            if mask is None:
                mask = np.zeros(self._X.shape, dtype=bool)
                upperright = np.triu_indices(meshPoints) # TODO: allow user to set mesh points with keyword 
                mask[upperright] = True
                np.fill_diagonal(mask, False)
                mask = np.flipud(mask)
                self._mask[(meshPoints, 2)] = mask
            self._Xdot = funcs[self._stateVariable1](*self._mumotModel._getArgTuple2d(self._controller._paramNames, self._controller._paramValues, argDict, self._stateVariable1, self._stateVariable2, self._X, self._Y))
            self._Ydot = funcs[self._stateVariable2](*self._mumotModel._getArgTuple2d(self._controller._paramNames, self._controller._paramValues, argDict, self._stateVariable1, self._stateVariable2, self._X, self._Y))
            self._speed = np.log(np.sqrt(self._Xdot ** 2 + self._Ydot ** 2))
            self._Xdot = np.ma.array(self._Xdot, mask=mask)
            self._Ydot = np.ma.array(self._Ydot, mask=mask)        
        self._logs.append(log)
        
    def _get_field3d(self, kind, meshPoints):
        self._controller._update_params_from_widgets()
        with io.capture_output() as log:
            self._log(kind)
            funcs = self._mumotModel._getFuncs()
            argNamesSymb = list(map(Symbol, self._controller._paramNames))
            argDict = dict(zip(argNamesSymb, self._controller._paramValues))
            self._Z, self._Y, self._X = np.mgrid[0:1:complex(0, meshPoints), 0:1:complex(0, meshPoints), 0:1:complex(0, meshPoints)] # TODO system size defined to be one
            mask = self._mask.get((meshPoints, 3))
            if mask is None:
#                mask = np.zeros(self._X.shape, dtype=bool)
#                 upperright = np.triu_indices(meshPoints) # TODO: allow user to set mesh points with keyword 
#                 mask[upperright] = True
#                 np.fill_diagonal(mask, False)
#                 mask = np.flipud(mask)
                mask = self._X + self._Y + self._Z >= 1
#                mask = mask.astype(int)
                self._mask[(meshPoints, 3)] = mask
            self._Xdot = funcs[self._stateVariable1](*self._mumotModel._getArgTuple3d(self._controller._paramNames, self._controller._paramValues, argDict, self._stateVariable1, self._stateVariable2, self._stateVariable3, self._X, self._Y, self._Z))
            self._Ydot = funcs[self._stateVariable2](*self._mumotModel._getArgTuple3d(self._controller._paramNames, self._controller._paramValues, argDict, self._stateVariable1, self._stateVariable2, self._stateVariable3, self._X, self._Y, self._Z))
            self._Zdot = funcs[self._stateVariable3](*self._mumotModel._getArgTuple3d(self._controller._paramNames, self._controller._paramValues, argDict, self._stateVariable1, self._stateVariable2, self._stateVariable3, self._X, self._Y, self._Z))
            self._speed = np.log(np.sqrt(self._Xdot ** 2 + self._Ydot ** 2 + self._Zdot ** 2))
#            self._Xdot = self._Xdot * mask
#            self._Ydot = self._Ydot * mask
#            self._Zdot = self._Zdot * mask
            self._Xdot = np.ma.array(self._Xdot, mask=mask)
            self._Ydot = np.ma.array(self._Ydot, mask=mask)        
            self._Zdot = np.ma.array(self._Zdot, mask=mask)
        self._logs.append(log)

class MuMoTstreamView(MuMoTfieldView):
    def _plot_field(self):
        super()._plot_field()
        self._get_field2d("2d stream plot", 100) # TODO: allow user to set mesh points with keyword
        plt.streamplot(self._X, self._Y, self._Xdot, self._Ydot, color = self._speed, cmap = 'gray') # TODO: define colormap by user keyword
#        plt.set_aspect('equal') # TODO


class MuMoTvectorView(MuMoTfieldView):
    def _plot_field(self):
        super()._plot_field()
        if self._stateVariable3 == None:
            self._get_field2d("2d vector plot", 10) # TODO: allow user to set mesh points with keyword
            plt.quiver(self._X, self._Y, self._Xdot, self._Ydot, units='width', color = 'black') # TODO: define colormap by user keyword
    #        plt.set_aspect('equal') # TODO
        else:
            self._get_field3d("3d vector plot", 10)
            ax = self._figure.gca(projection = '3d')
            ax.quiver(self._X, self._Y, self._Z, self._Xdot, self._Ydot, self._Zdot, length = 0.01, color = 'black') # TODO: define colormap by user keyword; normalise off maximum value in self._speed, and meshpoints?
#           plt.set_aspect('equal') # TODO
        
class MuMoTbifurcationView(MuMoTview): # bifurcation view on model 
    _pyDSmodel = None
    _bifurcationParameter = None
    _stateVariable1 = None
    _stateVariable2 = None

    def __init__(self, model, controller, paramDict, bifurcationParameter, stateVariable1, stateVariable2, **kwargs):
        super().__init__(model, controller)

        with io.capture_output() as log:      
            name = 'MuMoT Model' + str(id(self))
            self._pyDSmodel = dst.args(name = name)
            self._pyDSmodel.pars = paramDict
            varspecs = {}
            for reactant in self._mumotModel._reactants:
                varspecs[str(reactant)] = str(self._mumotModel._equations[reactant])
            self._pyDSmodel.varspecs = varspecs
    
            if model._systemSize != None:
                # TODO: shouldn't allow system size to be varied?
                pass
    #                self._paramValues.append(1)
    #                self._paramNames.append(str(self._systemSize))
    #                widget = widgets.FloatSlider(value = 1, min = _rateLimits[0], max = _rateLimits[1], step = _rateStep, description = str(self._systemSize), continuous_update = False)
    #                widget.on_trait_change(self._replot_bifurcation2D, 'value')
    #                self._widgets.append(widget)
    #                display(widget)
            else:
                print('Cannot attempt bifurcation plot until system size is set, using substitute()')
                return
            
            if stateVariable2 == None:
                # 2-d bifurcation diagram
                pass
            else:
                # 3-d bifurcation diagram (TODO: currently unsupported)
                self._stateVariable2 = stateVariable2
                assert false
                
            # Prepare the system to start close to a steady state
            self._bifurcationParameter = bifurcationParameter
            self._stateVariable1 = stateVariable1
            self._stateVariable2= stateVariable2
    #            self._pyDSode.set(pars = {bifurcationParameter: 0} )       # Lower bound of the bifurcation parameter (TODO: set dynamically)
    #            self._pyDSode.set(pars = self._pyDSmodel.pars )       # Lower bound of the bifurcation parameter (TODO: set dynamically)
    #            self._pyDSode.pars = {bifurcationParameter: 0}             # Lower bound of the bifurcation parameter (TODO: set dynamically?)
            initconds = {stateVariable1: self._pyDSmodel.pars[str(self._mumotModel._systemSize)] / len(self._mumotModel._reactants)} # TODO: guess where steady states are?
            for reactant in self._mumotModel._reactants:
                if str(reactant) != stateVariable1:
                    initconds[str(reactant)] = self._pyDSmodel.pars[str(self._mumotModel._systemSize)] / len(self._mumotModel._reactants)
                self._pyDSmodel.ics = initconds
            self._pyDSmodel.ics      = {'A': 0.1, 'B': 0.9 }           
    #            self._pyDSode.set(ics = initconds)
            self._pyDSode = dst.Generator.Vode_ODEsystem(self._pyDSmodel)  
            self._pyDSode.set(pars = {bifurcationParameter: 5} )                       # TODO remove magic number
            self._pyDScont = dst.ContClass(self._pyDSode)              # Set up continuation class 
            # TODO: add self._pyDScontArgs to __init__()
            self._pyDScontArgs = dst.args(name='EQ1', type='EP-C')     # 'EP-C' stands for Equilibrium Point Curve. The branch will be labeled 'EQ1'.
            self._pyDScontArgs.freepars     = [bifurcationParameter]   # control parameter(s) (should be among those specified in self._pyDSmodel.pars)
            self._pyDScontArgs.MaxNumPoints = 450                      # The following 3 parameters are set after trial-and-error TODO: how to automate this?
            self._pyDScontArgs.MaxStepSize  = 1e-1
            self._pyDScontArgs.MinStepSize  = 1e-5
            self._pyDScontArgs.StepSize     = 2e-3
            self._pyDScontArgs.LocBifPoints = ['LP', 'BP']                    # TODO WAS 'LP' (detect limit points / saddle-node bifurcations)
            self._pyDScontArgs.SaveEigen    = True                     # to tell unstable from stable branches
#            self._pyDScontArgs.CalcStab     = True
        self._logs.append(log)


#            self._bifurcation2Dfig = plt.figure(1)                    

        if kwargs != None:
            self._plotType = kwargs.get('plotType', 'pyDS')
        else:
            self._plotType = 'pyDS'

        self._plot_bifurcation()
            

    def _plot_bifurcation(self):
        self._controller._errorMessage.value= ''
        with io.capture_output() as log:
            self._log("bifurcation analysis")
            self._pyDScont.newCurve(self._pyDScontArgs)
            try:
                try:
                    self._pyDScont['EQ1'].backward()
                except:
                    self._controller._errorMessage.value = self._controller._errorMessage.value + 'Continuation failure (backward)<br>'
                try:
                    self._pyDScont['EQ1'].forward()                                  # TODO: how to choose direction?
                except:
                    self._controller._errorMessage.value = self._controller._errorMessage.value + 'Continuation failure (forward)<br>'
            except ZeroDivisionError:
                self._controller._errorMessage.value = self._controller._errorMessage.value + 'Division by zero<br>'
    #            self._pyDScont['EQ1'].info()
        if self._plotType.lower() == 'mumot':
            # use internal plotting routines (TODO: not yet supported)
            assert false
        else:
            if self._plotType.lower() != 'pyds':
                self._controller._errorMessage.value = self._controller._errorMessage.value + 'Unknown plotType argument: using default pyDS tool plotting<br>' 
            if self._stateVariable2 == None:
                # 2-d bifurcation diagram
                self._pyDScont.display([self._bifurcationParameter, self._stateVariable1], stability = True, figure = self._figureNum)
#                self._pyDScont.plot.fig1.axes1.axes.set_title('Bifurcation Diagram')
            else:
                pass
        self._logs.append(log)

    def _replot_bifurcation(self):
        self._controller._update_params_from_widgets()
        for name, value in zip(self._controller._paramNames, self._controller._paramValues):
            self._pyDSmodel.pars[name] = value
        self._pyDScont.plot.clearall()
#        self._pyDSmodel.ics      = {'A': 0.1, 'B': 0.9 }    # TODO: replace           
        self._pyDSode = dst.Generator.Vode_ODEsystem(self._pyDSmodel)  # TODO: add to __init__()
        self._pyDSode.set(pars = {self._bifurcationParameter: 5} )                       # TODO remove magic number
        self._pyDScont = dst.ContClass(self._pyDSode)              # Set up continuation class (TODO: add to __init__())
##        self._pyDScont.newCurve(self._pyDScontArgs)
#        self._pyDScont['EQ1'].reset(self._pyDSmodel.pars)
#        self._pyDSode.set(pars = self._pyDSmodel.pars)
#        self._pyDScont['EQ1'].reset()
#        self._pyDScont.update(self._pyDScontArgs)                         # TODO: what does this do?
        self._plot_bifurcation()

class MuMoTmultiagentView(MuMoTview): # agent on networks view on model 
    _colors = None
    _probabilities = None
    _graph = None
    _agents = None
    _positions = None
    _arena_width = 1
    _arena_height = 1
    _plot = None

    def __init__(self, model, controller, paramDict, **kwargs):
        super().__init__(model, controller)

        with io.capture_output() as log:
            colors = cm.rainbow(np.linspace(0, 1, len(self._mumotModel._reactants) ))  # @UndefinedVariable
            self._colors = {}
            i = 0
            for state in self._mumotModel._reactants:
                self._colors[state] = colors[i] 
                i += 1
            
        self._logs.append(log)

        if kwargs != None: # TODO: Currently not used (updated through _widgetDict). To rethink plot type in a better way
            self._plotType = kwargs.get('plotType', 'plain')
        else:
            self._plotType = 'plain'
            
    
    def _log(self, analysis):
        print("Starting", analysis, "with parameters ", end='')
        for w in self._controller._widgetDict.values():
            print('(' + w.description + '=' + str(w.value) + '), ', end='')
        print("at", datetime.datetime.now())
    
    def _plot_timeEvolution(self):
        with io.capture_output() as log:
            self._controller._update_params_from_widgets()
            self._plotType = self._controller._widgetDict['plotType'].value
            print("Plot Type: " + str(self._plotType) ) 
            
#                 self._controller._ratesDict[self._controller._paramNames[i]] = self._controller._widgets[i].value 
            self._log("Multiagent simulation")
            self._convertRatesIntoProbabilities(self._mumotModel._reactants, self._mumotModel._rules)

            # Clearing the plot
            self._plot = self._figure.add_subplot(111)
            self._plot.clear()

            maxTime = self._controller._widgetDict['maxTime'].value
            if (self._plotType == 'evo'):
                plt.axes().set_aspect('auto')
                # create the frame
                totAgents = sum(self._controller._initialState.values())
                self._plot.axis([0, maxTime, 0, totAgents])
                self._figure.show()
            elif self._controller._widgetDict['netType'].value == NetworkType.DYNAMIC and self._plotType == "graph":
                plt.axes().set_aspect('equal')
                
            # plot legend
            markers = [plt.Line2D([0,0],[0,0],color=color, marker='', linestyle='-') for color in self._colors.values()]
            self._plot.legend(markers, self._colors.keys(), bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)
            # show canvas
            self._figure.canvas.draw()
            
            logs = self._runMultiagent(self._controller._initialState, maxTime)
            # replot legend at the end
            self._plot.legend(markers, self._colors.keys(), bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.) # TODO: display legend every timeframe in 'graph' plots
            self._controller._fileToDownload = logs[1]
#             plt.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)
            
#             for state,pop in logs[1].items():
#                 print("Plotting:"+str(pop))
#                 plt.plot(pop, label=state)
            
        self._logs.append(log)
        
    def _runMultiagent(self, initialState, maxTime):
        # init the controller variables
        self._initMultiagent()
        
        # init logging structs
        historyState = []
        historyState.append(initialState)
        evo = {}
        for state,pop in initialState.items():
            evo[state] = []
            evo[state].append(pop)
        if self._controller._widgetDict['trace'].value:
            positionHistory = []
            for _ in np.arange(sum(initialState.values())):
                positionHistory.append( [] )
        
        # init progress bar
        self._controller._progressBar.max = maxTime
        
        dynamic = self._controller._widgetDict['netType'].value == NetworkType.DYNAMIC
        if (not dynamic) and self._plotType == "graph":
            pos_layout = nx.circular_layout(self._graph)
        
        for i in np.arange(1, maxTime+1):
            #print("Time: " + str(i))
            self._controller._progressBar.value = i
            self._controller._progressBar.description = "Loading " + str(round(i/maxTime*100)) + "%:"
            #print("Agents: " + str(self._controller._agents))
            if dynamic and self._controller._widgetDict['trace'].value:
                for idx, a in enumerate(self._agents):
                    positionHistory[idx].append( self._positions[idx] )
            
            currentState = self._stepMultiagent()
                    
            historyState.append(currentState)
            for state,pop in currentState.items():
                evo[state].append(pop)
            
            ## Plotting
            if (self._plotType == "evo"):
                for state,pop in evo.items():
                    #self._plot.plot(pop, color=self._colors[state]) #label=state,
                    # Plot only the last timestep rather than overlay all
                    self._plot.plot([i-1,i], pop[len(pop)-2:len(pop)], color=self._colors[state])

            elif (self._plotType == "graph"):
                self._plot.clear()
                if dynamic:
                    self._plot.axis([0, 1, 0, 1])
#                     xs = [p[0] for p in positions]
#                     ys = [p[1] for p in positions]
#                     plt.plot(xs, ys, 'o' )
                    xs = {}
                    ys = {}
                    for state in initialState.keys():
                        xs[state] = []
                        ys[state] = []
                    for a in np.arange(len(self._positions)):
                        xs[self._agents[a]].append( self._positions[a][0] )
                        ys[self._agents[a]].append( self._positions[a][1] )
                        
                        if self._controller._widgetDict['interactions'].value:
                            for n in self._getNeighbours(a, self._positions, self._controller._widgetDict['netParam'].value): 
                                self._plot.plot((self._positions[a][0], self._positions[n][0]),(self._positions[a][1], self._positions[n][1]), '-', c='y')
                        
                        if self._controller._widgetDict['trace'].value:
                            trace_xs = [p[0] for p in positionHistory[a] ]
                            trace_ys = [p[1] for p in positionHistory[a] ]
                            self._plot.plot( trace_xs, trace_ys, '-', c='0.6') 
                    for state in currentState.keys():
                        self._plot.plot(xs.get(state,[]), ys.get(state,[]), 'o', c=self._colors[state] )
#                     plt.axes().set_aspect('equal')
                else:
                    stateColors=[]
                    for n in self._graph.nodes():
                        stateColors.append( self._colors.get( self._agents[n], 'w') ) 
                    nx.draw(self._graph, pos_layout, node_color=stateColors, with_labels=True)
            # dinamically update the plot each timestep
            self._figure.canvas.draw()
                
        self._controller._progressBar.description = "Completed 100%:"
        print("State distribution each timestep: " + str(historyState))
        print("Temporal evolution per state: " + str(evo))
        return (historyState,evo)
    
    def _singleRun(self, randomSeed):
        # set random seed
        np.random.seed(randomSeed)
        
        # init the controller variables
        self._initMultiagent()
        
        # Create logging structs
        historyState = []
        evo = {}
        evo['time'] = [0]
        for state,pop in self._controller._initialState.items():
            evo[state] = []
            evo[state].append(pop)
         
        t = 0
        currentState = []
        for reactant in sorted(self._mumotModel._reactants, key=str):
            currentState.append(self._controller._initialState[reactant])
        historyState.append( [t] + currentState )
        maxTime = self._controller._widgetDict['maxTime'].value
 
        for t in np.arange(1, maxTime+1):
            newState = self._stepMultiagent()
            currentState = []
            for reactant in sorted(self._mumotModel._reactants, key=str):
                currentState.append(newState[reactant])

            # log step
            historyState.append( [t] + currentState )
            for state,pop in zip(sorted(self._mumotModel._reactants, key=str), currentState):
                evo[state].append(pop)
            evo['time'].append(t)
                      
#         print("State distribution each timestep: " + str(historyState))
#         print("Temporal evolution per state: " + str(evo))
        return (historyState,evo) 
    
    def _convertRatesIntoProbabilities(self, reactants, rules):
        self._initProbabilitiesMap(reactants, rules)
        #print(self._probabilities)
        self._computeScalingFactor()
        self._applyScalingFactor()
        #print(self._probabilities)
    
    # deriving the transition probabilities map from reaction rules
    def _initProbabilitiesMap(self, reactants, rules):
        self._probabilities = {}
        assignedDestReactants = {}
        for reactant in reactants:
            probSets = {}
            probSets['void'] = []
            for rule in rules:
                if not len(rule.lhsReactants) == len(rule.rhsReactants):
                    print('Raction with varying number of reactacts is not currently supported in multiagent/SSA simulations.' +
                          ' Please, keep the same number of reactants on the left and right handside of each reaction rule.')
                    return 1
                for react in rule.lhsReactants:
                    if react == reactant:
                        numReagents = len(rule.lhsReactants)
                        # if individual transition (i.e. no interaction needed)
                        if numReagents == 1:
                            probSets['void'].append( [rule.rate, self._controller._widgetDict[str(rule.rate)].value, rule.rhsReactants[0]] )
                        
                        # if interaction transition
                        elif numReagents == 2:
                            # checking if the considered reactant is active or passive in the interaction (i.e. change state afterwards)
                            if reactant not in rule.rhsReactants: # add entry only if the reactant is passive (i.e. change state afterwards)
                                # determining the otherReactant, which is NOT the considered one
                                if rule.lhsReactants[0] == reactant:
                                    otherReact = rule.lhsReactants[1]
                                else:
                                    otherReact = rule.lhsReactants[0]
                                # determining the destReactant
                                if rule.rhsReactants[0] in assignedDestReactants.get(rule, []) or rule.rhsReactants[0] == otherReact :
                                    destReact = rule.rhsReactants[1]
                                else:
                                    destReact = rule.rhsReactants[0]
                                # this is necessary to keep track of the assigned reactant when both reactants change on the right-handside
                                if assignedDestReactants.get(rule) == None:
                                    assignedDestReactants[rule] = []
                                assignedDestReactants[rule].append(destReact)
                                
                                if probSets.get(otherReact) == None:
                                    probSets[otherReact] = []
                                    
                                probSets[otherReact].append( [rule.rate, self._controller._widgetDict[str(rule.rate)].value, destReact] )
                            else:
                                if rule.lhsReactants.count(reactant) == 2: 
                                    # TODO: treat in a special way the 'self' interaction!
                                    warningMsg = "WARNING!! Reactant " + str(reactant) + " has a self-reaction " + str(rule.rate) + " which is not currently properly handled."
                                    self._controller._errorMessage.value += warningMsg + "<br>"
                                    print(warningMsg)
                            
                        elif numReagents > 2:
                            print('More than two reagents in one rule. Unhandled situation, please use at max two reagents per reaction rule')
                            return 1
                        
            self._probabilities[reactant] = probSets
            print("React " + str(reactant))
            print(probSets)

    def _computeScalingFactor(self):
        # Determining the minimum speed of the process (thus the max-scaling factor)
        maxRatesAll = 0
        for probSets in self._probabilities.values():
            voidRates = 0
            maxRates = 0
            for react, probSet in probSets.items():
                tmpRates = 0
                for prob in probSet:
                    #print("adding P=" + str(prob[1]))
                    if react == 'void':
                        voidRates += prob[1]
                    else:
                        tmpRates += prob[1]
                if tmpRates > maxRates:
                    maxRates = tmpRates
            #print("max Rates=" + str(maxRates) + " void Rates=" + str(voidRates))
            if (maxRates + voidRates) > maxRatesAll:
                maxRatesAll = maxRates + voidRates
        #self._scaling = 1/maxRatesAll
        if maxRatesAll>0: scaling = 1/maxRatesAll 
        else: scaling = 1
        self._controller._update_scaling_widget(scaling)
        print("Scaling factor s=" + str(self._controller._widgetDict['scaling'].value))
        
    def _applyScalingFactor(self):
        # Multiply all rates by the scaling factor
        for probSets in self._probabilities.values():
            for probSet in probSets.values():
                for prob in probSet:
                    prob[1] *= self._controller._widgetDict['scaling'].value
                    
    def _initGraph(self, graphType, numNodes, netParam=None):
        if (graphType == NetworkType.FULLY_CONNECTED):
            print("Generating full graph")
            self._graph = nx.complete_graph(numNodes) #np.repeat(0, self.numNodes)
        elif (graphType == NetworkType.ERSOS_RENYI):
            print("Generating Erdos-Renyi graph (connected)")
            if netParam is not None and netParam > 0 and netParam <= 1: 
                self._graph = nx.erdos_renyi_graph(numNodes, netParam, np.random.randint(MAX_RANDOM_SEED))
                i = 0
                while ( not nx.is_connected( self._graph ) ):
                    print("Graph was not connected; Resampling!")
                    i = i+1
                    self._graph = nx.erdos_renyi_graph(numNodes, netParam, np.random.randint(MAX_RANDOM_SEED)*i*2211)
            else:
                print ("ERROR! Invalid network parameter (link probability) for E-R networks. It must be between 0 and 1; input is " + str(netParam) )
                return
        elif (graphType == NetworkType.BARABASI_ALBERT):
            print("Generating Barabasi-Albert graph")
            netParam = int(netParam)
            if netParam is not None and netParam > 0 and netParam <= numNodes: 
                self._graph = nx.barabasi_albert_graph(numNodes, netParam, np.random.randint(MAX_RANDOM_SEED))
            else:
                print ("ERROR! Invalid network parameter (number of edges per new node) for B-A networks. It must be an integer between 1 and " + str(numNodes) + "; input is " + str(netParam))
                return
        elif (graphType == NetworkType.SPACE):
            ## TODO: implement network generate by placing points (with local communication range) randomly in 2D space
            print("ERROR: Graphs of type SPACE are not implemented yet.")
            return
        elif (graphType == NetworkType.DYNAMIC):
            self._positions = []
            for _ in range(numNodes):
                x = np.random.rand()
                y = np.random.rand()
                o = np.random.rand() * np.pi * 2.0
                self._positions.append( (x,y,o) )
            return

    def _initMultiagent(self):
        # init the agents list
        self._agents = []
        for state, pop in self._controller._initialState.items():
            self._agents.extend( [state]*pop )
        self._agents = np.random.permutation(self._agents).tolist() # random shuffling of elements (useful to avoid initial clusters in networks)
            
        
    def _stepMultiagent(self):
        currentState = {}
        for state in self._controller._initialState.keys():
            currentState[state] = 0
        tmp_agents = copy.deepcopy(self._agents)
        dynamic = self._controller._widgetDict['netType'].value == NetworkType.DYNAMIC
        if dynamic:
            tmp_positions = copy.deepcopy(self._positions)
            communication_range = self._controller._widgetDict['netParam'].value
            speed = self._controller._widgetDict['speed'].value
            correlatedness = self._controller._widgetDict['correlated'].value
        for idx, a in enumerate(self._agents):
            if dynamic:
                neighNodes = self._getNeighbours(idx, tmp_positions, communication_range)
#                 print("Agent " + str(idx) + " moved from " + str(self._positions[idx]) )
                self._positions[idx] = self._updatePosition( self._positions[idx][0], self._positions[idx][1], self._positions[idx][2], speed, correlatedness)
#                 print("to position " + str(self._positions[idx]) )
            else:
                neighNodes = list(nx.all_neighbors(self._graph, idx))
            neighAgents = [tmp_agents[x] for x in neighNodes]
#                 print("Neighs of agent " + str(idx) + " are " + str(neighNodes) + " with states " + str(neighAgents) )
            self._agents[idx] = self._stepOneAgent(a, neighAgents)
            currentState[ self._agents[idx] ] = currentState.get(self._agents[idx],0) + 1
        
        return currentState

    # one timestep for one agent
    def _stepOneAgent(self, agent, neighs):
        #probSets = copy.deepcopy(self._probabilities[agent])
        rnd = np.random.rand()
        lastVal = 0
        probSets = self._probabilities[agent]
        # counting how many neighbours for each state (to be uses for the interaction probabilities)
        neighCount = {x:neighs.count(x) for x in probSets.keys()}
#         print("Agent " + str(agent) + " with probSet=" + str(probSets))
#         print("nc:"+str(neighCount))
        for react, probSet in probSets.items():
            for prob in probSet:
                if react == 'void':
                    popScaling = 1
                else:
                    popScaling = neighCount[react]/len(neighs) if len(neighs) > 0 else 0
                val = popScaling * prob[1]
                if (rnd < val + lastVal):
                    # A state change happened!
                    #print("Reaction: " + str(prob[0]) + " by agent " + str(agent) + " that becomes " + str(prob[2]) )
                    return prob[2]
                else:
                    lastVal += val
        # No state change happened
        return agent
    
    def _updatePosition(self, x, y, o, speed, correlatedness):
        # random component
        rand_o = np.random.rand() * np.pi * 2.0
        rand_x = speed * np.cos(rand_o) * (1-correlatedness)
        rand_y = speed * np.sin(rand_o) * (1-correlatedness)
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
        #### CODE FOR A BOUNDED ARENA
        # if a.position.x < 0: a.position.x = 0
        # elif a.position.x > self.dimensions.x: a.position.x = self.dimensions.x 
        # if a.position.y < 0: a.position.y = 0
        # elif a.position.y > self.dimensions.y: a.position.x = self.dimensions.x 
        return (x,y,o)

    # return the (index) list of neighbours of 'agent'
    def _getNeighbours(self, agent, positions, distance_range):
        neighbour_list = []
        for neigh in np.arange(len(positions)):
            if (not neigh == agent) and (self._distance_on_torus(positions[agent][0], positions[agent][1], positions[neigh][0], positions[neigh][1]) < distance_range):
                neighbour_list.append(neigh)
        return neighbour_list
    
    # returns the minimum distance calucalted on the torus given by periodic boundary conditions
    def _distance_on_torus( self, x_1, y_1, x_2, y_2 ):
        return np.sqrt(min(abs(x_1 - x_2), self._arena_width - abs(x_1 - x_2))**2 + 
                    min(abs(y_1 - y_2), self._arena_height - abs(y_1 - y_2))**2)
    

class MuMoTSSAView(MuMoTview): # agent on networks view on model 
    _colors = None
    _plot = None
    #_reactantsList = None # storing a reactant list to keep a fixed order
    _reactantsMatrix = None # a matrix form of the left-handside of the rules
    _ruleChanges = None # the effect of each rule

    def __init__(self, model, controller, paramDict, **kwargs):
        super().__init__(model, controller)

        with io.capture_output() as log:
            colors = cm.rainbow(np.linspace(0, 1, len(self._mumotModel._reactants) ))  # @UndefinedVariable
            self._colors = {}
            i = 0
            for state in self._mumotModel._reactants:
                self._colors[state] = colors[i] 
                i += 1
        
        self._logs.append(log)

        if kwargs != None: # TODO: Currently not used (updated through _widgetDict). To rethink plot type in a better way
            self._plotType = kwargs.get('plotType', 'plain')
        else:
            self._plotType = 'plain'
    
    def _log(self, analysis):
        print("Starting", analysis, "with parameters ", end='')
        for w in self._controller._widgetDict.values():
            print('(' + w.description + '=' + str(w.value) + '), ', end='')
        print("at", datetime.datetime.now())
    
    def _plot_timeEvolution(self):
        with io.capture_output() as log:
            self._controller._update_params_from_widgets()
            self._plotType = self._controller._widgetDict['plotType'].value
            print("Plot Type: " + str(self._plotType) ) 
            
#                 self._controller._ratesDict[self._controller._paramNames[i]] = self._controller._widgets[i].value 
            self._log("Stochastic Simulation Algorithm (SSA)")
            self._createSSAmatrix(self._mumotModel._reactants, self._mumotModel._rules)
            
            # Clearing the plot
            self._plot = self._figure.add_subplot(111)
            self._plot.clear()
            # Creating main figure frame
            maxTime = self._controller._widgetDict['maxTime'].value
            if (self._plotType == 'evo'):
                totAgents = sum(self._controller._initialState.values())
                self._plot.axis([0, maxTime, 0, totAgents])
                self._figure.show()
                # make legend
                markers = [plt.Line2D([0,0],[0,0],color=color, marker='', linestyle='-') for color in self._colors.values()]
                self._plot.legend(markers, self._colors.keys(), bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)
                # show canvas
                self._figure.canvas.draw()
           
            logs = self._runSSA(self._controller._initialState, maxTime)
            self._controller._fileToDownload = logs[1]
           
        self._logs.append(log)
        
    def _runSSA(self, initialState, maxTime): 
        # Create logging structs
        historyState = []
        evo = {}
        evo['time'] = [0]
        for state,pop in initialState.items():
            evo[state] = []
            evo[state].append(pop)
         
        self._controller._progressBar.max = maxTime
         
        t = 0
        currentState = []
        for reactant in sorted(self._mumotModel._reactants, key=str):
            currentState.append(initialState[reactant])
        historyState.append( [t] + currentState )
 
        while t < maxTime:
            # update progress bar
            self._controller._progressBar.value = t
            self._controller._progressBar.description = "Loading " + str(round(t/maxTime*100)) + "%:"
#             print("Time: " + str(t))
             
            timeInterval,currentState = self._stepSSA(currentState)
            # increment time
            t += timeInterval
             
            # log step
            historyState.append( [t] + currentState )
            for state,pop in zip(sorted(self._mumotModel._reactants, key=str), currentState):
                evo[state].append(pop)
            evo['time'].append(t)
             
            ## Plot
            if (self._plotType == "evo"):
                for state,pop in evo.items():
                    if (state == 'time'): continue
                    #self._plot.plot(evo['time'], pop, color=self._colors[state]) #label=state,
                    # Plot only the last timestep rather than overlay all
                    self._plot.plot(evo['time'][len(pop)-2:len(pop)], pop[len(pop)-2:len(pop)], color=self._colors[state]) #label=state,
            elif (self._plotType == "final"):
                print("TODO: Missing final distribution visualisation.")
            # dinamically update the plot each timestep
            self._figure.canvas.draw()
         
        self._controller._progressBar.value = self._controller._progressBar.max
        self._controller._progressBar.description = "Completed 100%:"
        print("State distribution each timestep: " + str(historyState))
        print("Temporal evolution per state: " + str(evo))
        return (historyState,evo)
    
    def _createSSAmatrix(self, reactants, rules):
#         self._reactantsList = []
#         for reactant in reactants:
#             self._reactantsList.append(reactant)
        self._reactantsMatrix = []
        self._ruleChanges = []
        for rule in rules:
            lineR = []
            lineC = []
            for reactant in sorted(self._mumotModel._reactants, key=str):
                before = rule.lhsReactants.count(reactant)
                after = rule.rhsReactants.count(reactant)
                lineR.append( before * self._controller._widgetDict[str(rule.rate)].value )
                lineC.append( after - before )
            self._reactantsMatrix.append(lineR)
            self._ruleChanges.append(lineC)  
            
    def _stepSSA(self, currentState): 
        # update transition probabilities accounting for the current state
        probabilitiesOfChange = []
        for rule in self._reactantsMatrix:
            prob = sum([a*b for a,b in zip(rule,currentState)])
            numReagents = sum(x > 0 for x in rule)
            if numReagents > 1:
                prob /= sum(currentState)**( numReagents -1 ) 
            probabilitiesOfChange.append(prob)
        probSum = sum(probabilitiesOfChange)
        
        # computing when is happening next reaction
        timeInterval = np.random.exponential( 1/probSum );
        
        # Selecting the occurred reaction at random, with probability proportional to each reaction probabilities
        bottom = 0.0
        # Get a random between [0,1) (but we don't want 0!)
        reaction = 0.0
        while (reaction == 0.0):
            reaction = np.random.random_sample()
        # Normalising probOfChange in the range [0,1]
        probabilitiesOfChange = [pc/probSum for pc in probabilitiesOfChange]
        index = -1
        for i, prob in enumerate(probabilitiesOfChange):
            if ( reaction >= bottom and reaction < (bottom + prob)):
                index = i
                break
            bottom += prob
        
        if (index == -1):
            print("ERROR! Transition not found. Error in the algorithm execution.")
            sys.exit()
        #print(currentState)
        #print(self._ruleChanges[index])
        # apply the change
        currentState = list(np.array(currentState) + np.array(self._ruleChanges[index]) )
        #print(currentState)
                
        return (timeInterval, currentState)
    
    def _singleRun(self, randomSeed):
        # set random seed
        np.random.seed(randomSeed)
        # Create logging structs
        historyState = []
        evo = {}
        evo['time'] = [0]
        for state,pop in self._controller._initialState.items():
            evo[state] = []
            evo[state].append(pop)
         
        t = 0
        currentState = []
        for reactant in sorted(self._mumotModel._reactants, key=str):
            currentState.append(self._controller._initialState[reactant])
        historyState.append( [t] + currentState )
        maxTime = self._controller._widgetDict['maxTime'].value
 
        while t < maxTime:
            timeInterval,currentState = self._stepSSA(currentState)
            # increment time
            t += timeInterval

            # log step
            historyState.append( [t] + currentState )
            for state,pop in zip(sorted(self._mumotModel._reactants, key=str), currentState):
                evo[state].append(pop)
            evo['time'].append(t)
                      
#         print("State distribution each timestep: " + str(historyState))
#         print("Temporal evolution per state: " + str(evo))
        return (historyState,evo)
    

def parseModel(modelDescription):
    # TODO: add system size to model description
    if "get_ipython" in modelDescription:
        # hack to extract model description from input cell tagged with %%model magic
        modelDescr = modelDescription.split("\"")[0].split("'")[5]
    elif "->" in modelDescription:
        # model description provided as string
        modelDescr = modelDescription
    else:
        # assume input describes filename and attempt to load
        assert false
    
    # strip out any basic LaTeX equation formatting user has input
    modelDescr = modelDescr.replace('$','')
    modelDescr = modelDescr.replace(r'\\\\','')
    # split _rules line-by-line, one rule per line
    modelRules = modelDescr.split('\\n')
    # parse and construct the model
    reactants = set()
    rates = set()
    rules = []
    model = MuMoTmodel()

    for rule in modelRules:
        if (len(rule) > 0):
            tokens = rule.split()
            reactantCount = 0
            rate = ""
            state = 'A'
            newRule = _Rule()
            for token in tokens:
                # simulate a simple pushdown automaton to recognise valid _rules, storing representation during processing
                # state A: expecting a reactant (transition to state B)
                # state B: expecting a '+' (transition to state A) or a '->' (transition to state C) (increment reactant count on entering)
                # state C: expecting a reactant (transition to state D)
                # state D: expecting a '+' or a ':' (decrement reactant count on entering)
                # state E: expecting a rate equation until end of rule
                token = token.replace("\\\\",'\\')

                if state == 'A':
                    if token != "+" and token != "->" and token != ":":
                        state = 'B'
                        if '^' in token:
                            raise SyntaxError("Reactants cannot contain '^' :" + token + " in " + rule)
                        reactantCount += 1
                        expr = process_sympy(token)
                        reactantAtoms = expr.atoms()
                        if len(reactantAtoms) != 1:
                            raise SyntaxError("Non-singleton symbol set in token " + token +" in rule " + rule)
                        for reactant in reactantAtoms:
                            pass # this loop extracts element from singleton set
                        if reactant not in reactants:
                            reactants.add(reactant)
                        newRule.lhsReactants.append(reactant)
                    else:
                        _raiseModelError("reactant", token, rule)
                        return
                elif state == 'B':
                    if token == "->":
                        state = 'C'
                    elif token == '+':
                        state = 'A'
                    else:
                        _raiseModelError("'->' or '+'", token, rule)
                        return
                elif state == 'C':
                    if token != "+" and token != "->" and token != ":":
                        state = 'D'
                        if '^' in token:
                            raise SyntaxError("Reactants cannot contain '^' :" + token + " in " + rule)
                        reactantCount -= 1
                        expr = process_sympy(token)
                        reactantAtoms = expr.atoms()
                        if len(reactantAtoms) != 1:
                            raise SyntaxError("Non-singleton symbol set in token " + token +" in rule " + rule)
                        for reactant in reactantAtoms:
                            pass # this loop extracts element from singleton set
                        if reactant not in reactants:
                            reactants.add(reactant)
                        newRule.rhsReactants.append(reactant)                        
                    else:
                        _raiseModelError("reactant", token, rule)
                        return
                elif state == 'D':
                    if token == ":":
                        state = 'E'
                    elif token == '+':
                        state = 'C'
                    else:
                        _raiseModelError("':' or '+'", token, rule)
                        return
                elif state == 'E':
                    rate += token
                    # state = 'F'
                else:
                    # should never reach here
                    assert False

            newRule.rate = process_sympy(rate) 
            rateAtoms = newRule.rate.atoms(Symbol)
            for atom in rateAtoms:
                if atom not in rates:
                    rates.add(atom)
                    
            if reactantCount == 0:
                rules.append(newRule)
            else:
                raise SyntaxError("Unequal number of reactants on lhs and rhs of rule " + rule)

            
    model._rules = rules
    model._reactants = reactants
    model._rates = rates
    model._equations = _deriveODEsFromRules(model._reactants, model._rules)
    model._ratesLaTeX = {}
    rates = map(latex, list(model._rates))
    for (rate, latexStr) in zip(model._rates, rates):
        model._ratesLaTeX[repr(rate)] = latexStr
                    
    return model

def _deriveODEsFromRules(reactants, rules):
    # TODO: replace with principled derivation via Master Equation and van Kampen expansion
    equations = {}
    terms = []
    for rule in rules:
        term = None
        for reactant in rule.lhsReactants:
            if term == None:
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
                if rhs == None:
                    rhs = factor * term
                else:
                    rhs = rhs + factor * term
        equations[reactant] = rhs

    return equations

    
def _raiseModelError(expected, read, rule):
    raise SyntaxError("Expected " + expected + " but read '" + read + "' in rule: " + rule)


