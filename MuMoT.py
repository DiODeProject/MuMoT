
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


from IPython.display import display, Math, Latex
import ipywidgets.widgets as widgets
from matplotlib import pyplot as plt
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
#from numpy.oldnumeric.fix_default_axis import _args3
#from matplotlib.offsetbox import kwargs

get_ipython().magic('alias_magic model latex')
get_ipython().magic('matplotlib nbagg')

figureCounter = 1 # global figure counter for model views

# class with default parameters
class MuMoTdefault:
    initialRateValue = 2 # TODO: was 1 (choose initial values sensibly)
    rateLimits = (0.0, 20.0) # TODO: choose limit values sensibly
    rateStep = 0.1 # TODO: choose rate step sensibly
    @staticmethod
    def setRateDefaults(self, initRate=initialRateValue, limits=rateLimits, step=rateStep):
        self.initialRateValue = initRate
        self.rateLimits = limits
        self.rateStep = step
    
    maxTime = 10
    timeLimits = (1, 100)
    timeStep = 1
    @staticmethod
    def setTimeDefaults(self, initTime=maxTime, limits=timeLimits, step=timeStep):
        self.maxTime = initTime
        self.timeLimits = limits
        self.timeStep = step
    
    

class MuMoTmodel: # class describing a model
    _rules = None # list of rules
    _reactants = None # set of reactants
    _systemSize = None # parameter that determines system size, set by using substitute()
    _reactantsLaTeX = None # list of LaTeX strings describing reactants (TODO: depracated?)
    _rates = None # set of rates
    _ratesLaTeX = None # dictionary of LaTeX strings describing rates
    _equations = None # dictionary of ODE righthand sides with reactant as key
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

        
    def stream(self, stateVariable1, stateVariable2, **kwargs):
        if self._systemSize != None:
            pass
        else:
            print('Cannot construct streamplot until system size is set, using substitute()')
            return    

        # construct controller
        viewController = self._field2d_controller(stateVariable1, stateVariable2, True, **kwargs)
        
        # construct view
        modelView = MuMoTstreamView(self, viewController, stateVariable1, stateVariable2, **kwargs)
                
        viewController.setView(modelView)
        viewController.setReplotFunction(modelView._plot_field)         
        
        return viewController
        
    def vector(self, stateVariable1, stateVariable2, stateVariable3 = None, **kwargs):
        # TODO resolve valid 2d or 3d plot invocation
        
        if stateVariable3 == None:
            if self._systemSize != None:
                pass
            else:
                print('Cannot construct streamplot until system size is set, using substitute()')
                return    
    
            # construct controller
            viewController = self._field2d_controller(stateVariable1, stateVariable2, True, **kwargs)
            
            # construct view
            modelView = MuMoTvectorView(self, viewController, stateVariable1, stateVariable2, **kwargs)
                    
            viewController.setView(modelView)
            viewController.setReplotFunction(modelView._plot_field)         
            
        else:
            assert false
        
        return viewController
            
        

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

    def networkAgents(self, netType="full", initialState="Auto", maxTime="Auto", **kwargs):
        if initialState=="Auto":
            #initialState = [100] + [0]*(len(reactantList)-1)
            first = True
            initialState = {}
            for reactant in self._reactants:
                if first:
                    print("Automatic Initial State sets 100 agents in state " + str(reactant) )
                    initialState[reactant] = 100
                    first = False
                else:
                    initialState[reactant] = 0
        else:
            print("TO-DO: check if the Initial State has valid length and positive values")
        print("Initial State is " + str(initialState) )
        
        # construct controller
        paramValues = []
        paramNames = [] 
        #paramValues.extend( [initialState, netType, maxTime] )
        for rate in self._rates:
            paramValues.append((MuMoTdefault.initialRateValue, MuMoTdefault.rateLimits[0], MuMoTdefault.rateLimits[1], MuMoTdefault.rateStep))
            paramNames.append(str(rate))

        if (maxTime == "Auto" or maxTime <= 0):
            maxTime = MuMoTdefault.maxTime
#             timeLimitMax = max(maxTime, MuMoTdefault.timeLimits[1])
#         paramNames.append("maxTime")
#         paramValues.append( (maxTime, MuMoTdefault.timeLimits[0], timeLimitMax, MuMoTdefault.timeStep) )
        
        viewController = MuMoTcontroller(paramValues, paramNames, self._ratesLaTeX, False)
        
        paramDict = {}
#         paramDict['reactants'] = self._reactants
#         paramDict['rules'] = self._rules
        paramDict['maxTime'] = maxTime
        paramDict['initialState'] = initialState
        paramDict['netType'] = netType
        modelView = MuMoTnetworkView(self, viewController, paramDict, **kwargs)
        
        viewController.setView(modelView)
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
# #                widget = widgets.FloatSlider(value = 1, min = rateLimits[0], max = rateLimits[1], step = rateStep, description = str(self._systemSize), continuous_update = False)
# #                widget.on_trait_change(self._replot_bifurcation2D, 'value')
# #                self._widgets.append(widget)
# #                display(widget)
#             else:
#                 print('Cannot attempt bifurcation plot until system size is set, using substitute()')
#                 return
#             for rate in self._rates:
#                 if str(rate) != bifurcationParameter:
#                     self._paramValues.append(initialRateValue)
#                     self._paramNames.append(str(rate))
#                     widget = widgets.FloatSlider(value = initialRateValue, min = rateLimits[0], max = rateLimits[1], step = rateStep, description = str(rate), continuous_update = False)
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

    def _field2d_controller(self, stateVariable1, stateVariable2, contRefresh, **kwargs):
        # controller for 2d field-based plot (stream or 2d vector)
                
        if Symbol(stateVariable1) in self._reactants and Symbol(stateVariable2) in self._reactants:
            if stateVariable1 != stateVariable2:
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
                
            else:
                print('State variables cannot be the same')
                return
        else:
            print('Invalid reactant provided as state variable')
            return

    def _getFuncs(self):
        # lambdify sympy equations for numerical integration, plotting, etc.
        if self._systemSize == None:
            assert false
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
    
    def _getArgTuple(self, argNames, argValues, argDict, stateVariable1 = None, stateVariable2 = None, X = None, Y = None):
        # get tuple to evalute functions returned by _getFuncs with
        # TODO how to return a tuple usable for numerical integration, for which stateVariable1 == stateVariable2 == None?
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
    _ratesDict = None

    def __init__(self, paramValues, paramNames, paramLabelDict, continuousReplot):
        self._paramValues = []
        self._paramNames = []
        self._paramLabelDict = paramLabelDict
        self._widgets = []
        self._ratesDict = {}
        for pair in zip(paramNames, paramValues):
            widget = widgets.FloatSlider(value = pair[1][0], min = pair[1][1], 
                                         max = pair[1][2], step = pair[1][3], 
                                         description = r'\(' + self._paramLabelDict[pair[0]] + r'\)', 
                                         continuous_update = continuousReplot)
            # widget.on_trait_change(replotFunction, 'value')
            self._widgets.append(widget)
            display(widget)
            self._ratesDict[pair[0]] = pair[1][0]
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

# class MuMoTmultiagentController(MuMoTcontroller): # class describing a controller for multiagent views
#     _probabilities = None
#     _scaling = 0
#     
#     def __init__(self, paramValues, paramNames, paramLabelDict, continuousReplot, reactants, rules):
#         MuMoTcontroller.__init__(self, paramValues, paramNames, paramLabelDict, continuousReplot)
        
    

class MuMoTview: # class describing a view on a model
    _mumotModel = None
    _figure = None
    _figureNum = None
    _widgets = None
    _controller = None
    _logs = None
    
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

class MuMoTfieldView(MuMoTview): # field view on model (specialised by MuMoTvectorView and MuMoTstreamView)
    _stateVariable1 = None # 1st state variable (x-dimension)
    _stateVariable2 = None # 2nd state variable (y-dimension)
    _X = None # X ordinates array
    _Y = None # Y ordinates array
    _X = None # X derivatives array
    _Y = None # Y derivatives array
    _speed = None # speed array
    
    def __init__(self, model, controller, stateVariable1, stateVariable2, **kwargs):
        super().__init__(model, controller)

        self._stateVariable1 = Symbol(stateVariable1)
        self._stateVariable2 = Symbol(stateVariable2)

        self._plot_field()
    
    def _plot_field(self):
        plt.figure(self._figureNum)
        plt.clf()
    
    def _get_field2d(self, kind, meshPoints):
        for i in np.arange(0, len(self._controller._paramValues)):
            # UGLY!
            self._controller._paramValues[i] = self._controller._widgets[i].value
        with io.capture_output() as log:
            self._log(kind)
            funcs = self._mumotModel._getFuncs()
            argNamesSymb = list(map(Symbol, self._controller._paramNames))
            argDict = dict(zip(argNamesSymb, self._controller._paramValues))
            self._Y, self._X = np.mgrid[0:1:complex(0, meshPoints), 0:1:complex(0, meshPoints)] # TODO system size defined to be one
            self._Xdot = funcs[self._stateVariable1](*self._mumotModel._getArgTuple(self._controller._paramNames, self._controller._paramValues, argDict, self._stateVariable1, self._stateVariable2, self._X, self._Y))
            self._Ydot = funcs[self._stateVariable2](*self._mumotModel._getArgTuple(self._controller._paramNames, self._controller._paramValues, argDict, self._stateVariable1, self._stateVariable2, self._X, self._Y))
            self._speed = np.log(np.sqrt(self._Xdot ** 2 + self._Ydot ** 2))
        self._logs.append(log)

class MuMoTstreamView(MuMoTfieldView):
    def _plot_field(self):
        super()._plot_field()
        self._get_field2d("stream plot", 100) # TODO: allow user to set mesh points with keyword
        plt.streamplot(self._X, self._Y, self._Xdot, self._Ydot, color = self._speed, cmap = 'gray') # TODO: define colormap by user keyword
#        plt.set_aspect('equal') # TODO


class MuMoTvectorView(MuMoTfieldView):
    def _plot_field(self):
        super()._plot_field()
        self._get_field2d("vector plot", 10) # TODO: allow user to set mesh points with keyword
        plt.quiver(self._X, self._Y, self._Xdot, self._Ydot, units='width', color = 'black') # TODO: define colormap by user keyword
#        plt.set_aspect('equal') # TODO

        
class MuMoTbifurcationView(MuMoTview): # bifurcation view on model 
    _pyDSmodel = None
    _bifurcationParameter = None
    _stateVariable1 = None
    _stateVariable2 = None
    _plotType = None

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
    #                widget = widgets.FloatSlider(value = 1, min = rateLimits[0], max = rateLimits[1], step = rateStep, description = str(self._systemSize), continuous_update = False)
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
        for i in np.arange(0, len(self._controller._paramValues)):
            # UGLY!
            self._controller._paramValues[i] = self._controller._widgets[i].value
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

class MuMoTnetworkView(MuMoTview): # agent on networks view on model 
    _probabilities = None
#     _reactants = None
#     _rules = None
    _scaling = 0
    _agentsState = None
    _graph = None
    _maxTime = None
    _plotType = None

    def __init__(self, model, controller, paramDict, **kwargs):
        super().__init__(model, controller)

        with io.capture_output() as log:      
#             name = 'MuMoT Model' + str(id(self))
#             self._reactants = paramDict['reactants']
#             self._rules = paramDict['rules']
            self._maxTime = paramDict['maxTime']
            self._initialState = paramDict['initialState']
            
        self._logs.append(log)

        if kwargs != None:
            self._plotType = kwargs.get('plotType', 'plain')
        else:
            self._plotType = 'plain'

        self._plot_timeEvolution()
    
    def _plot_timeEvolution(self):
        with io.capture_output() as log:
            self._log("Networked multiagent")
            for i in np.arange(0, len(self._controller._paramValues)):
                self._controller._paramValues[i] = self._controller._widgets[i].value
                self._controller._ratesDict[self._controller._paramNames[i]] = self._controller._widgets[i].value 
    #         print(self._controller._paramValues)
            self.generateProbabilitiesMap(self._mumotModel._reactants, self._mumotModel._rules)
            print(self._probabilities)
            self.computeScalingFactor()
            self.applyScalingFactor()
            print(self._probabilities)
            logs = self.iterateAgentStep(self._initialState, self._maxTime)
            # Plotting evo figure
            plt.figure(self._figureNum)
            plt.clf()
            for state,pop in logs[1].items():
                print("Plotting:"+str(pop))
                plt.plot(pop, label=state)
            plt.figure(self._figureNum)
            plt.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)
        self._logs.append(log)
        
    def iterateAgentStep(self, initialState, maxTime):
        # Create logging structs
        historyState = []
        historyState.append(initialState)
        evo = {}
        for state,pop in initialState.items():
            evo[state] = []
            evo[state].append(pop)
        # init the agents list
        agents = []
        for state, pop in initialState.items():
            agents.extend( [state]*pop )
        for i in np.arange(0, maxTime):
            print("Time: " + str(i))
            #print("Agents: " + str(agents))
            currentState = {}
            for state in initialState.keys():
                currentState[state] = 0
            tmp_agents = copy.deepcopy(agents)
            for idx, a in enumerate(agents):
                neighs = tmp_agents # TODO: compute the neighs
                agents[idx] = self.oneStep(a, neighs)
                currentState[ agents[idx] ] = currentState.get(agents[idx],0) + 1
            historyState.append(currentState)
            for state,pop in currentState.items():
                evo[state].append(pop)
        print(historyState)
        print(evo)
        return (historyState,evo)
    
    # one timestep for one agent
    def oneStep(self, agent, neighs):
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
                    popScaling = neighCount[react]/len(neighs) 
                val = popScaling * prob[1]
                if (rnd < val + lastVal):
                    # A state change happened!
                    #print("Reaction: " + str(prob[0]) + " by agent " + str(agent) + " that becomes " + str(prob[2]) )
                    return prob[2]
                else:
                    lastVal += val
        # No state change happened
        return agent
    
    def generateProbabilitiesMap(self, reactants, rules):
        # deriving the transition probabilities map from reaction rules
        self._probabilities = {}
        for reactant in reactants:
            probSets = {}
            probSets['void'] = []
            for rule in rules:
                assignedDestReactants = []
                for react in rule.lhsReactants:
                    if react == reactant:
                        numReagents = len(rule.lhsReactants)
                        # if individual transition (i.e. no interaction needed)
                        if numReagents == 1:
                            probSets['void'].append( [rule.rate, self._controller._ratesDict[str(rule.rate)], rule.rhsReactants[0]] )
                        
                        # if interaction transition
                        elif numReagents == 2:
                            # checking if the considered reactant is active or passive in the interaction (i.e. change state afterwards)
                            if reactant not in rule.rhsReactants:
                                # determining the otherReactant, which is NOT the considered one
                                if rule.lhsReactants[0] == reactant:
                                    otherReact = rule.lhsReactants[1]
                                else:
                                    otherReact = rule.lhsReactants[0]
                                # determining the destReactant
                                if rule.rhsReactants[0] in assignedDestReactants or rule.rhsReactants[0] == otherReact :
                                    destReact = rule.rhsReactants[1]
                                else:
                                    destReact = rule.rhsReactants[0]
                                assignedDestReactants.append(destReact)
                                
                                if probSets.get(otherReact) == None:
                                    probSets[otherReact] = []
                                    
                                probSets[otherReact].append( [rule.rate, self._controller._ratesDict[str(rule.rate)], destReact] )
                            #else:
                                # TO-DO treat in a special way the 'self' interaction!
                                #print("Reactant " + str(reactant) + " has active role in reaction " + str(rule.rate))
                                
                            
                        elif numReagents > 2:
                            print('More than two reagents in one rule. Unhandled situation, please use at max two reagents per reaction rule')
                            return 1
                        
            self._probabilities[reactant] = probSets
#             print("React " + str(reactant))
#             print(probSets)
        
    def computeScalingFactor(self):
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
        self._scaling = 1/maxRatesAll
        print("Scaling factor s=" + str(self._scaling))

    def applyScalingFactor(self):
        # Multiply all rates by the scaling factor
        for probSets in self._probabilities.values():
            for probSet in probSets.values():
                for prob in probSet:
                    prob[1] *= self._scaling

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

