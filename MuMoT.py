
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

get_ipython().magic('alias_magic model latex')
get_ipython().magic('matplotlib nbagg')


class MuMoTmodel: # class describing a model
    _rules = [] # list of rules
    _reactants = set() # set of reactants
    _systemSize = None # parameter that determines system size, set by using substitute()
    _reactantsLaTeX = [] # list of LaTeX strings describing reactants (TODO: depracated?)
    _rates = set() # set of rates
    _ratesLaTeX = [] # list of LaTeX strings describing rates (TODO: depracated?)
    _equations = {} # dictionary of ODE righthand sides with reactant as key
    _pyDSmodel = None
    _dot = None # graphviz visualisation of model
    _renderImageFormat = 'png' # image format used for rendering edge labels for model visualisation
    _tmpdirpath = '__mumot_files__' # local path for creation of temporary storage
    _tmpdir = None # temporary storage for image files, etc. used in visualising model
    _tmpfiles = [] # list of temporary files created
    
    
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
        if self._ratesLaTeX == None:
            self._ratesLaTeX = []
            rates = map(latex, list(self._rates))
            for rate in rates:
                self._ratesLaTeX.append(rate)
            self._ratesLaTeX.sort()
        for rate in self._ratesLaTeX:
            display(Math(rate))
    
    
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


    def bifurcation(self, bifurcationParameter, stateVariable1, stateVariable2 = None):
        # experimental: start constructing interactive PyDSTool model from equations, based on PyDSTool tutorials
        name = 'MuMoT Model' + str(id(self))
        self._pyDSmodel = dst.args(name = name)
        self._paramDict = {} # TODO make local?
        initialRateValue = 0.1
        rateLimits = (-10.0, 10.0)
        rateStep = 0.1
        for rate in self._rates:
            self._paramDict[str(rate)] = initialRateValue # TODO choose initial values sensibly
        self._paramDict[str(self._systemSize)] = 1 # TODO choose initial values sensibly
        self._pyDSmodel.pars = self._paramDict
        varspecs = {}
        for reactant in self._reactants:
            varspecs[str(reactant)] = str(self._equations[reactant])
        self._pyDSmodel.varspecs = varspecs

        if stateVariable2 == None:
            # 2-d bifurcation diagram
            # create widgets
            self._paramValues = []
            self._paramNames = []
            self._widgets = []
            if self._systemSize != None:
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
            for rate in self._rates:
                if str(rate) != bifurcationParameter:
                    self._paramValues.append(initialRateValue)
                    self._paramNames.append(str(rate))
                    widget = widgets.FloatSlider(value = initialRateValue, min = rateLimits[0], max = rateLimits[1], step = rateStep, description = str(rate), continuous_update = False)
                    widget.on_trait_change(self._replot_bifurcation2D, 'value')
                    self._widgets.append(widget)
                    display(widget)
            widget = widgets.HTML(value = '')
            self._errorMessage = widget                                # TODO: add to __init__()
            display(self._errorMessage)
            # Prepare the system to start close to a steady state
            self._bifurcationParameter = bifurcationParameter          # TODO: remove hack (bifurcation parameter for multiple possible bifurcations needs to be stored in self)
            self._stateVariable1 = stateVariable1                      # TODO: remove hack (state variable for multiple possible bifurcations needs to be stored in self)
#            self._pyDSode.set(pars = {bifurcationParameter: 0} )       # Lower bound of the bifurcation parameter (TODO: set dynamically)
#            self._pyDSode.set(pars = self._pyDSmodel.pars )       # Lower bound of the bifurcation parameter (TODO: set dynamically)
#            self._pyDSode.pars = {bifurcationParameter: 0}             # Lower bound of the bifurcation parameter (TODO: set dynamically?)
            initconds = {stateVariable1: self._paramDict[str(self._systemSize)] / len(self._reactants)} # TODO: guess where steady states are?
            for reactant in self._reactants:
                if str(reactant) != stateVariable1:
                    initconds[str(reactant)] = self._paramDict[str(self._systemSize)] / len(self._reactants)
            self._pyDSmodel.ics = initconds                            
#            self._pyDSode.set(ics = initconds)
            self._pyDSode = dst.Generator.Vode_ODEsystem(self._pyDSmodel)  # TODO: add to __init__()
            self._pyDScont = dst.ContClass(self._pyDSode)              # Set up continuation class (TODO: add to __init__())
            # TODO: add self._pyDScontArgs to __init__()
            self._pyDScontArgs = dst.args(name='EQ1', type='EP-C')     # 'EP-C' stands for Equilibrium Point Curve. The branch will be labeled 'EQ1'.
            self._pyDScontArgs.freepars     = [bifurcationParameter]   # control parameter(s) (should be among those specified in self._pyDSmodel.pars)
            self._pyDScontArgs.MaxNumPoints = 450                      # The following 3 parameters are set after trial-and-error TODO: how to automate this?
            self._pyDScontArgs.MaxStepSize  = 2
            self._pyDScontArgs.MinStepSize  = 1e-5
            self._pyDScontArgs.StepSize     = 2e-2
            self._pyDScontArgs.LocBifPoints = 'LP'                     # detect limit points / saddle-node bifurcations
            self._pyDScontArgs.SaveEigen    = True                     # to tell unstable from stable branches

            plt.ion()
#            self._bifurcation2Dfig = plt.figure(1)                     # TODO: add to __init__()
            self._pyDScont.newCurve(self._pyDScontArgs)
            try:
                self._pyDScont['EQ1'].backward()                            # TODO: how to choose direction?
                self._errorMessage.value = ''
            except ZeroDivisionError:
                self._errorMessage.value = 'Division by zero'
#            self._pyDScont['EQ1'].info()
            self._pyDScont.display([bifurcationParameter, stateVariable1], stability = True, figure = 1)
            self._pyDScont.plot.fig1.axes1.axes.set_title('Bifurcation Diagram')
        else:
            # 3-d bifurcation diagram
            assert false


    def _replot_bifurcation2D(self):
        for i in np.arange(0, len(self._paramValues)):
            # UGLY!
            self._paramValues[i] = self._widgets[i].value
        for name, value in zip(self._paramNames, self._paramValues):
            self._pyDSmodel.pars[name] = value
        self._pyDScont.plot.clearall()
        self._pyDSode = dst.Generator.Vode_ODEsystem(self._pyDSmodel)  # TODO: add to __init__()
        self._pyDScont = dst.ContClass(self._pyDSode)              # Set up continuation class (TODO: add to __init__())
        self._pyDScont.newCurve(self._pyDScontArgs)
#        self._pyDScont['EQ1'].reset(self._pyDSmodel.pars)
#        self._pyDSode.set(pars = self._pyDSmodel.pars)
#        self._pyDScont['EQ1'].reset()
#        self._pyDScont.update(self._pyDScontArgs)                         # TODO: what does this do?
        try:
            self._pyDScont['EQ1'].backward()                                  # TODO: how to choose direction?
            self._errorMessage.value = ''
        except ZeroDivisionError:
            self._errorMessage.value = 'Division by zero'
#        self._pyDScont['EQ1'].info()
        self._pyDScont.display([self._bifurcationParameter, self._stateVariable1], stability = True, figure = 1)
        self._pyDScont.plot.fig1.axes1.axes.set_title('Bifurcation Diagram')
                                 
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
