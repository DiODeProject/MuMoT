""" @package MuMoT
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
"""

from IPython.display import display, clear_output, Math, Latex, Javascript
import ipywidgets.widgets as widgets
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatch
import numpy as np
from scipy.integrate import odeint
from sympy import *
import math
import PyDSTool as dst
from graphviz import Digraph
from process_latex.process_latex import process_sympy # was `from process_latex import process_sympy` before packaging for pip
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
import json
import sys

import matplotlib.ticker as ticker
from math import log10, floor
import ast
from matplotlib.pyplot import plot

#from matplotlib.offsetbox import kwargs
#from __builtin__ import None
#from numpy.oldnumeric.fix_default_axis import _args3
#from matplotlib.offsetbox import kwargs



get_ipython().magic('alias_magic model latex')
get_ipython().magic('matplotlib nbagg')

figureCounter = 1 # global figure counter for model views

MAX_RANDOM_SEED = 4294967295
INITIAL_RATE_VALUE = 0.5
RATE_BOUND = 10.0
RATE_STEP = 0.1
MULTIPLOT_COLUMNS = 2
EMPTYSET_SYMBOL = process_sympy('1')

INITIAL_COND_INIT_VAL = 0.0
INITIAL_COND_INIT_BOUND = 1.0


line_color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'grey', 'orange', 'k']

# enum possible Network types
class NetworkType(Enum):
    FULLY_CONNECTED = 0
    ERSOS_RENYI = 1
    BARABASI_ALBERT = 2
    SPACE = 3
    DYNAMIC = 4

# class with default parameters
class MuMoTdefault:
    _initialRateValue = 2 ## @todo: was 1 (choose initial values sensibly)
    _rateLimits = (0.0, 20.0) ## @todo: choose limit values sensibly
    _rateStep = 0.1 ## @todo: choose rate step sensibly
    @staticmethod
    def setRateDefaults(initRate=_initialRateValue, limits=_rateLimits, step=_rateStep):
        MuMoTdefault._initialRateValue = initRate
        MuMoTdefault._rateLimits = limits
        MuMoTdefault._rateStep = step
    
    _maxTime = 3
    _timeLimits = (0, 10)
    _timeStep = 0.1
    @staticmethod
    def setTimeDefaults(initTime=_maxTime, limits=_timeLimits, step=_timeStep):
        MuMoTdefault._maxTime = initTime
        MuMoTdefault._timeLimits = limits
        MuMoTdefault._timeStep = step
        
    _agents = 1.0
    _agentsLimits = (0.0, 1.0)
    _agentsStep = 0.01
    @staticmethod
    def setAgentsDefaults(initAgents=_agents, limits=_agentsLimits, step=_agentsStep):
        MuMoTdefault._agents = initAgents
        MuMoTdefault._agentsLimits = limits
        MuMoTdefault._agentsStep = step
    

## class describing a model
class MuMoTmodel:
    ## list of rules
    _rules = None 
    ## set of reactants
    _reactants = None
    ## set of fixed-concentration reactants (boundary conditions)
    _constantReactants = None 
    ## parameter that determines system size, set by using substitute()
    _systemSize = None
    ## is system size constant or not?
    _constantSystemSize = None
    ## list of LaTeX strings describing reactants (@todo: depracated?)
    _reactantsLaTeX = None
    ## set of rates
    _rates = None 
    ## dictionary of LaTeX strings describing rates and constant reactants (@todo: rename)
    _ratesLaTeX = None 
    ## dictionary of ODE righthand sides with reactant as key
    _equations = None
    ## set of solutions to equations
    _solutions = None 
    ## summary of stoichiometry as nested dictionaries
    _stoichiometry = None
    ## dictionary (reagents as keys) with reaction-rate and relative effects of each reaction-rule for each reagent (structure necessary for multiagent simulations)
    _agentProbabilities = None
    ## dictionary of lambdified functions for integration, plotting, etc.
    _funcs = None
    ## tuple of argument symbols for lambdified functions
    _args = None 
    ## graphviz visualisation of model
    _dot = None
    ## image format used for rendering edge labels for model visualisation
    _renderImageFormat = 'png'
    ## local path for creation of temporary storage
    _tmpdirpath = '__mumot_files__'
    ## temporary storage for image files, etc. used in visualising model
    _tmpdir = None 
    ## list of temporary files created
    _tmpfiles = None 
    
    ## create new model with variable substitutions listed as comma separated string of assignments
    def substitute(self, subsString):
        subs = []
        subsStrings = subsString.split(',')
        for subString in subsStrings:
            if '=' not in subString:
                raise SyntaxError("No '=' in assignment " + subString)
            assignment = process_sympy(subString)
            subs.append((assignment.lhs, assignment.rhs))
        newModel = MuMoTmodel()
        newModel._rules = copy.deepcopy(self._rules)
        newModel._reactants = copy.deepcopy(self._reactants)
        newModel._constantReactants = copy.deepcopy(self._constantReactants)
        newModel._equations = copy.deepcopy(self._equations)
        newModel._stoichiometry = copy.deepcopy(self._stoichiometry)
        for sub in subs:
            if sub[0] in newModel._reactants and len(sub[1].atoms(Symbol)) == 1:
                raise SyntaxError("Using substitute to rename reactants not supported: " + str(sub[0]) + " = " + str(sub[1]))
        for reaction in newModel._stoichiometry:
            for sub in subs:
                newModel._stoichiometry[reaction]['rate'] = newModel._stoichiometry[reaction]['rate'].subs(sub[0], sub[1])
                for reactant in newModel._stoichiometry[reaction]:
                    if not reactant == 'rate':
                        if reactant == sub[0]:
                            if '+' not in str(sub[1]) and '-' not in str(sub[1]):
                                #replace keys according to: dictionary[new_key] = dictionary.pop(old_key)
                                newModel._stoichiometry[reaction][sub[1]] = newModel._stoichiometry[reaction].pop(reactant)
                            else:
                                newModel._stoichiometry[reaction][reactant].append({reactant: sub[1]})
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
                        if newModel._systemSize == None:
                            newModel._systemSize = atom
                        else:
                            raise SyntaxError("More than one unknown reactant encountered when trying to set system size: " + str(sub[0]) + " = " + str(sub[1]))
                if newModel._systemSize == None:
                    raise SyntaxError("Expected to find system size parameter but failed: " + str(sub[0]) + " = " + str(sub[1]))
                ## @todo: more thorough error checking for valid system size expression
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
        constantReactants = map(latex, list(newModel._constantReactants))
        for (reactant, latexStr) in zip(newModel._constantReactants, constantReactants):
            newModel._ratesLaTeX[repr(reactant)] = '(' + latexStr + ')'

        ## @todo: what else should be copied to new model?

        return newModel


    ## build a graphical representation of the model
    # if result cannot be plotted check for installation of libltdl - eg on Mac see if XQuartz requires update or do:<br>
    #  `brew install libtool --universal` <br>
    #  `brew link libtool`
    def visualise(self):
        errorShown = False
        if self._dot == None:
            dot = Digraph(comment = "Model", engine = 'circo')
            if not self._constantSystemSize:
                dot.node(str('1'), " ", image = self._localLaTeXimageFile(Symbol('\\emptyset'))) ## @todo: only display if used: for now, guess it is used if system size is non-constant                
            for reactant in self._reactants:
                dot.node(str(reactant), " ", image = self._localLaTeXimageFile(reactant))
            for reactant in self._constantReactants:
                dot.node(str(reactant), " ", image = self._localLaTeXimageFile(Symbol(self._ratesLaTeX[repr(reactant)])))                
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
                else:
                    if not errorShown:
                        errorShown = True
                        print("Model contains rules with three or more reactants; only displaying unary and binary rules")
            self._dot = dot
                
        return self._dot

    ## show a sorted LaTeX representation of the model's constant reactants
    def showConstantReactants(self):
        for reactant in self._constantReactants:
            display(Math(self._ratesLaTeX[repr(reactant)]))
#         if self._constantReactantsLaTeX == None:
#             self._constantReactantsLaTeX = []
#             reactants = map(latex, list(self._constantReactants))
#             for reactant in reactants:
#                 self._constantReactantsLaTeX.append(reactant)
#             self._constantReactantsLaTeX.sort()
#         for reactant in self._constantReactantsLaTeX:
#             display(Math(reactant))
        #reactant_list = []
        #for reaction in self._stoichiometry:
        #    for reactant in self._stoichiometry[reaction]:
        #        if not reactant in reactant_list:
        #            if not reactant == 'rate':
        #                display(Math(latex(reactant)))
        #                reactant_list.append(reactant)


    ## show a sorted LaTeX representation of the model's reactants
    def showReactants(self):
        if self._reactantsLaTeX == None:
            self._reactantsLaTeX = []
            reactants = map(latex, list(self._reactants))
            for reactant in reactants:
                self._reactantsLaTeX.append(reactant)
            self._reactantsLaTeX.sort()
        for reactant in self._reactantsLaTeX:
            display(Math(reactant))
        #reactant_list = []
        #for reaction in self._stoichiometry:
        #    for reactant in self._stoichiometry[reaction]:
        #        if not reactant in reactant_list:
        #            if not reactant == 'rate':
        #                display(Math(latex(reactant)))
        #                reactant_list.append(reactant)

    ## show a sorted LaTeX representation of the model's rate parameters
    def showRates(self):
        for reaction in self._stoichiometry:
            out = latex(self._stoichiometry[reaction]['rate']) + "\; (" + latex(reaction) + ")"
            display(Math(out))        
    
    def showRatesOLD(self):
        for rate in self._ratesLaTeX:
            display(Math(self._ratesLaTeX[rate]))

    def showSingleAgentRules(self):
        for agent,probs in self._agentProbabilities.items():
            if agent == EMPTYSET_SYMBOL:
                print("Spontaneous birth from EMPTYSET", end=' ' )
            else:
                print("Agent " + str(agent), end=' ')
            if probs:
                print("reacts" )
                for prob in probs:
                    print ("  at rate " + str(prob[1]), end=' ')
                    if prob[0]:
                        print ("when encounters " + str(prob[0]), end=' ' )
                    else:
                        print ("alone", end=' ') 
                    print("and becomes " + str(prob[2]), end=', ')
                    if prob[0]:
                        print("while", end=' ')
                        for i in np.arange(len(prob[0])):
                            print("reagent " + str(prob[0][i]) + " becomes " + str(prob[3][i]), end=' ')
                    print("")
            else: 
                print("does not initiate any reaction." ) 
    
    ## show a LaTeX representation of the model system of ODEs
    def showODEs(self):
        for reactant in self._reactants:
            out = "\\displaystyle \\frac{\\textrm{d}" + latex(reactant) + "}{\\textrm{d}t} := " + latex(self._equations[reactant])
            display(Math(out))
    
    ## displays stoichiometry as a dictionary with keys ReactionNr,
    # ReactionNr represents another dictionary with reaction rate, reactants and their stoichiometry
    def showStoichiometry(self):
        out = latex(self._stoichiometry)
        display(Math(out))
    
    ## displays Master equation expressed with ladder operators
    def showMasterEquation(self):
        P, t = symbols('P t')
        out_rhs=""
        stoich = self._stoichiometry
        nvec = []
        for key1 in stoich:
            for key2 in stoich[key1]:
                if not key2 == 'rate':
                    if not key2 in nvec:
                        nvec.append(key2)
        nvec = sorted(nvec, key=default_sort_key)
        assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'
        rhs_dict, substring = _deriveMasterEquation(stoich)
        #rhs_ME = 0
        term_count = 0
        for key in rhs_dict:
            #rhs_ME += terms_gcd(key*(rhs_dict[key][0]-1)*rhs_dict[key][1]*rhs_dict[key][2], deep=True)
            if term_count == 0:
                rhs_plus = ""
            else:
                rhs_plus = " + "
            out_rhs += rhs_plus + latex(rhs_dict[key][3]) + " ( " + latex((rhs_dict[key][0]-1)) + " ) " +  latex(rhs_dict[key][1]) + latex(rhs_dict[key][2])
            term_count += 1
        if len(nvec)==2:
            lhs_ME = Derivative(P(nvec[0], nvec[1],t),t)
        elif len(nvec)==3:
            lhs_ME = Derivative(P(nvec[0], nvec[1], nvec[2], t), t)
        else:
            lhs_ME = Derivative(P(nvec[0], nvec[1], nvec[2], nvec[3], t), t)     
        
        #return {lhs_ME: rhs_ME}
        out = latex(lhs_ME) + ":= " + out_rhs
        display(Math(out))
        if not substring == None:
            for sub in substring:
                display(Math("With \; substitution:\;" + latex(sub) + ":= " + latex(substring[sub])))
        
    ## shows van Kampen expansion when the operators are expanded up to second order
    def showVanKampenExpansion(self):
        rhs_vke, lhs_vke, substring = _doVanKampenExpansion(_deriveMasterEquation, self._stoichiometry)
        out = latex(lhs_vke) + " := \n" + latex(rhs_vke)
        display(Math(out))
        if not substring == None:
            for sub in substring:
                display(Math("With \; substitution:\;" + latex(sub) + ":= " + latex(substring[sub])))
    
    ## shows ODEs derived from the leading term in van Kampen expansion
    def showODEs_vKE(self):
        ODEdict = _getODEs_vKE(_get_orderedLists_vKE, self._stoichiometry)
        for ode in ODEdict:
            out = latex(ode) + " := " + latex(ODEdict[ode])
            display(Math(out))
        
    ## shows Fokker-Planck equation derived from term ~ O(1) in van Kampen expansion
    # this is the linear noise approximation
    def showFokkerPlanckEquation(self):
        FPEdict, substring = _getFokkerPlanckEquation(_get_orderedLists_vKE, self._stoichiometry)
        for fpe in FPEdict:
            out = latex(fpe) + " := " + latex(FPEdict[fpe])
            display(Math(out))
            if not substring == None:
                for sub in substring:
                    display(Math("With \; substitution:\;" + latex(sub) + ":= " + latex(substring[sub])))
    
    ## displays equations of motion of first and second order moments of noise                
    def showNoiseEOM(self):
        EQsys1stOrdMom, EOM_1stOrderMom, NoiseSubs1stOrder, EQsys2ndOrdMom, EOM_2ndOrderMom, NoiseSubs2ndOrder= _getNoiseEOM(_getFokkerPlanckEquation, _get_orderedLists_vKE, self._stoichiometry)
        for eom1 in EOM_1stOrderMom:
            out = "\\displaystyle \\frac{\\textrm{d}" + latex(eom1.subs(NoiseSubs1stOrder)) + "}{\\textrm{d}t} := " + latex(EOM_1stOrderMom[eom1].subs(NoiseSubs1stOrder))
            display(Math(out))
        for eom2 in EOM_2ndOrderMom:
            out = "\\displaystyle \\frac{\\textrm{d}" + latex(eom2.subs(NoiseSubs2ndOrder)) + "}{\\textrm{d}t} := " + latex(EOM_2ndOrderMom[eom2].subs(NoiseSubs2ndOrder))
            display(Math(out))
    
    
    ## displays noise in the stationary state
    def showNoiseStationarySol(self):
        SOL_1stOrderMom, NoiseSubs1stOrder, SOL_2ndOrdMomDict, NoiseSubs2ndOrder = _getNoiseStationarySol(_getNoiseEOM, _getFokkerPlanckEquation, _get_orderedLists_vKE, self._stoichiometry)
        print('Stationary solutions of first and second order moments of noise:')
        if SOL_1stOrderMom == None:
            print('Noise 1st-order moments could not be calculated analytically.')
            return None
        else:
            for sol1 in SOL_1stOrderMom:
                out = latex(sol1.subs(NoiseSubs1stOrder)) + latex(r'(t \to \infty)') + ":= " + latex(SOL_1stOrderMom[sol1].subs(NoiseSubs1stOrder))
                display(Math(out))
        if SOL_2ndOrdMomDict == None:
            print('Noise 2nd-order moments could not be calculated analytically.')
            return None
        else:
            for sol2 in SOL_2ndOrdMomDict:
                out = latex(sol2.subs(NoiseSubs2ndOrder)) + latex(r'(t \to \infty)') + " := " + latex(SOL_2ndOrdMomDict[sol2].subs(NoiseSubs2ndOrder))
                display(Math(out)) 
#         for sol1 in SOL_1stOrderMom:
#             out = latex(sol1.subs(NoiseSubs1stOrder)) + latex(r'(t \to \infty)') + ":= " + latex(SOL_1stOrderMom[sol1].subs(NoiseSubs1stOrder))
#             display(Math(out))
#         for sol2 in SOL_2ndOrdMomDict:
#             out = latex(sol2.subs(NoiseSubs2ndOrder)) + latex(r'(t \to \infty)') + " := " + latex(SOL_2ndOrdMomDict[sol2].subs(NoiseSubs2ndOrder))
#             display(Math(out))     
        
        
        
    
    # show a LaTeX representation of the model <br>
    # if rules have | after them update notebook (allegedly, or switch browser): <br>
    # `pip install --upgrade notebook`
    def show(self):
        for rule in self._rules:
            out = ""
            for reactant in rule.lhsReactants:
                if type(reactant) is numbers.One:
                    reactant = Symbol('\emptyset')
                if reactant in self._constantReactants:
                    out += "(" 
                out += latex(reactant)
                if reactant in self._constantReactants:
                    out += ")"                 
                out += " + "
            out = out[0:len(out) - 2] # delete the last ' + '
            out += " \\xrightarrow{" + latex(rule.rate) + "}"
            for reactant in rule.rhsReactants:
                if type(reactant) is numbers.One:
                    reactant = Symbol('\emptyset')
                if reactant in self._constantReactants:
                    out += "(" 
                out += latex(reactant)
                if reactant in self._constantReactants:
                    out += ")"                 
                out += " + "
            out = out[0:len(out) - 2] # delete the last ' + '
            display(Math(out))


    ## construct interactive time evolution plot for state variables      
    def numSimStateVar(self, showStateVars = None, params = None, **kwargs):
    #def numSimStateVar(self, stateVariable1, stateVariable2, stateVariable3 = None, stateVariable4 = None, params = None, **kwargs):
        try:
            kwargs['showInitSV'] = True
            
            # construct controller
            viewController = self._controller(False, params = params, **kwargs)
 
            # construct view
            modelView = MuMoTtimeEvoStateVarView(self, viewController, showStateVars, params = params, **kwargs)
             
            viewController.setView(modelView)
            viewController._setReplotFunction(modelView._plot_NumSolODE)         
            
            return viewController
        
        except:
            return None
    
    ## construct interactive time evolution plot for noise around fixed points         
    #def numSimNoiseCorrelations(self, showStateVars = None, params = None, **kwargs):
    def numSimNoiseCorrelations(self, params = None, **kwargs):
        try:
            kwargs['showInitSV'] = True
            kwargs['showNoise'] = True
            
            EQsys1stOrdMom, EOM_1stOrderMom, NoiseSubs1stOrder, EQsys2ndOrdMom, EOM_2ndOrderMom, NoiseSubs2ndOrder= _getNoiseEOM(_getFokkerPlanckEquation, _get_orderedLists_vKE, self._stoichiometry)
            #SOL_1stOrderMom, NoiseSubs1stOrder, SOL_2ndOrdMomDict, NoiseSubs2ndOrder = _getNoiseStationarySol(_getNoiseEOM, _getFokkerPlanckEquation, _get_orderedLists_vKE, self._stoichiometry)
            
            
            # construct controller
            viewController = self._controller(False, params = params, **kwargs)
            
            # construct view
            modelView = MuMoTtimeEvoNoiseCorrView(self, viewController, EOM_1stOrderMom, EOM_2ndOrderMom, params = params, **kwargs)
                  
            viewController.setView(modelView)
            viewController._setReplotFunction(modelView._plot_NumSolODE)         
            
            return viewController
        
        except:
            return None

    
#     ## construct interactive plot of noise around fixed points        
#     def fixedPointNoise(self, stateVariable1, stateVariable2, stateVariable3=None, params = None, **kwargs):
#         if self._check_state_variables(stateVariable1, stateVariable2, stateVariable3):
#             
#             kwargs['showNoise'] = True
#             n1, n2, n3, n4 = _getNoiseStationarySol(_getNoiseEOM, _getFokkerPlanckEquation, _get_orderedLists_vKE, self._stoichiometry)
#             # n3 is second order solution and will be used by MuMoTnoiseView
#             
#             
#             # get stationary solution of moments of noise variables <eta_i> and <eta_i eta_j>
#             #noiseStatSol = (n1,n2,n3,n4)#_getNoiseStationarySol(_getNoiseEOM, _getFokkerPlanckEquation, _get_orderedLists_vKE, self._stoichiometry)
#             
#             # construct controller
#             viewController = self._controller(False, plotLimitsSlider = not(self._constantSystemSize), params = params, **kwargs)
#             #viewController = self._controller(True, plotLimitsSlider = True, **kwargs)
#             
#             # construct view
#             modelView = MuMoTnoiseView(self, viewController, n3, stateVariable1, stateVariable2, params = params, **kwargs)
#                     
#             viewController.setView(modelView)
#             viewController._setReplotFunction(modelView._plot_field)         
#             
#             return viewController
#         else:
#             return None
# 
# 
# 
#     ## construct interactive stream plot        
#     def stream(self, stateVariable1, stateVariable2, stateVariable3 = None, params = None, **kwargs):
#         if self._check_state_variables(stateVariable1, stateVariable2, stateVariable3):
#             if stateVariable3 != None:
#                 print("3d stream plots not currently supported")
#                 
#                 return None
#             # construct controller
#             viewController = self._controller(True, plotLimitsSlider = not(self._constantSystemSize), params = params, **kwargs)
#             
#             # construct view
#             modelView = MuMoTstreamView(self, viewController, stateVariable1, stateVariable2, params = params, **kwargs)
#                     
#             viewController.setView(modelView)
#             viewController._setReplotFunction(modelView._plot_field)         
#             
#             return viewController
#         else:
#             return None
    
    ## construct interactive stream plot with the option to show noise around fixed points
    def stream(self, stateVariable1, stateVariable2, stateVariable3 = None, params = None, **kwargs):
        if self._check_state_variables(stateVariable1, stateVariable2, stateVariable3):
            if stateVariable3 != None:
                print("3d stream plots not currently supported")
                return None
            
            if kwargs.get('showNoise', False) == True:
                #check for substitutions in conserved systems for which stream plot only works WITHOUT noise in case of reducing number of state variables from 3 to 2
                substitutions = False
                for reaction in self._stoichiometry:
                    for key in self._stoichiometry[reaction]:
                        if key != 'rate':
                            if self._stoichiometry[reaction][key] != 'const':
                                if len(self._stoichiometry[reaction][key]) > 2:
                                    substitutions = True
                
                if substitutions == False:
                    SOL_1stOrderMomDict, NoiseSubs1stOrder, SOL_2ndOrdMomDict, NoiseSubs2ndOrder  = _getNoiseStationarySol(_getNoiseEOM, _getFokkerPlanckEquation, _get_orderedLists_vKE, self._stoichiometry)
                    # SOL_2ndOrdMomDict is second order solution and will be used by MuMoTnoiseView
                else:
                    SOL_2ndOrdMomDict = None
                
                if SOL_2ndOrdMomDict == None:
                    if substitutions == True:
                        print('Noise in stream plots is only available for systems with exactly two time dependent reactants. Stream plot only works WITHOUT noise in case of reducing number of state variables from 3 to 2 via substitute() - method.')
                    print('Warning: Noise in the system could not be calculated: \'showNoise\' automatically disabled.')
                    kwargs['showNoise'] = False
            else:
                SOL_2ndOrdMomDict = None
            
            continuous_update = not (kwargs.get('showNoise', False) or kwargs.get('showFixedPoints', False))
            # construct controller
            viewController = self._controller(continuous_update, plotLimitsSlider = not(self._constantSystemSize), params = params, **kwargs)
            
            # construct view
            modelView = MuMoTstreamView(self, viewController, SOL_2ndOrdMomDict, stateVariable1, stateVariable2, params = params, **kwargs)
                    
            viewController.setView(modelView)
            viewController._setReplotFunction(modelView._plot_field)         
            
            return viewController
        else:
            return None
    
    
    
    ## construct interactive vector plot        
    def vector(self, stateVariable1, stateVariable2, stateVariable3 = None, params = None, **kwargs):
        if self._check_state_variables(stateVariable1, stateVariable2, stateVariable3):

            continuous_update = not (kwargs.get('showNoise', False) or kwargs.get('showFixedPoints', False))

            # construct controller
            viewController = self._controller(continuous_update, plotLimitsSlider = not(self._constantSystemSize), params = params, **kwargs)
            
            # construct view
            modelView = MuMoTvectorView(self, viewController, stateVariable1, stateVariable2, stateVariable3, params = params, **kwargs)
                    
            viewController.setView(modelView)
            viewController._setReplotFunction(modelView._plot_field)         
                        
            return viewController
        else:
            return None
        
    ## construct interactive PyDSTool plot
    def bifurcation(self, bifurcationParameter, stateVariable1, stateVariable2 = None, params=None, **kwargs):
        try:
            kwargs['showInitSV'] = True
            
            if bifurcationParameter[0]=='\\':
                bifPar = bifurcationParameter[1:]
            else:
                bifPar = bifurcationParameter
            
            kwargs['chooseBifParam'] = bifPar
            
    
            # construct controller
            viewController = self._controller(False, params = params, **kwargs)
           
            # construct view
            modelView = MuMoTbifurcationView(self, viewController, bifurcationParameter, stateVariable1, stateVariable2, params = params, **kwargs)
            
            viewController.setView(modelView)
            viewController._setReplotFunction(modelView._plot_bifurcation)
        
            return viewController
        except:
            return None
    
#     
#     def bifurcation(self, bifurcationParameter, stateVariable1, stateVariable2 = None, params = None, **kwargs):
#         if self._systemSize != None:
#             pass
#         else:
#             print('Cannot construct bifurcation plot until system size is set, using substitute()')
#             return    
# 
#         initialRateValue = INITIAL_RATE_VALUE ## @todo was 1 (choose initial values sensibly)
#         rateLimits = (0, RATE_BOUND) ## @todo choose limit values sensibly
#         rateStep = RATE_STEP ## @todo choose rate step sensibly                
# 
# 
#         # construct controller
#         paramValues = []
#         paramNames = []        
#         for rate in self._rates:
#             if str(rate) != bifurcationParameter:
#                 paramValues.append((initialRateValue, rateLimits[0], rateLimits[1], rateStep))
#                 paramNames.append(str(rate))
#         viewController = MuMoTcontroller(paramValues, paramNames, self._ratesLaTeX, False, params = params, **kwargs)
# 
#         # construct view
#         modelView = MuMoTbifurcationView(self, viewController, bifurcationParameter, stateVariable1, stateVariable2, params = params, **kwargs)
#         
#         viewController.setView(modelView)
#         viewController._setReplotFunction(modelView._replot_bifurcation)
#         
#         return viewController
#     
    
    
    
    
    
    

    def multiagent(self, netType="full", initialState="Auto", maxTime="Auto", randomSeed="Auto", **kwargs):
        MAParams = {}
        if initialState=="Auto":
            first = True
            initialState = {}
            (allReactants, allConstantReactants) = self._getAllReactants()
            for reactant in allReactants | allConstantReactants: #self._reactants|self._constantReactants:
                if first:
#                     print("Automatic Initial State sets " + str(MuMoTdefault._agents) + " agents in state " + str(reactant) )
                    initialState[reactant] = MuMoTdefault._agents
                    first = False
                else:
                    initialState[reactant] = 0
        else:
            ## @todo check if the Initial State has valid length and positive values
#             print("TODO: check if the Initial State has valid length and positive values")
#             initialState_str = ast.literal_eval(initialState) # translate string into dict
            initialState_str = initialState
            initialState = {}
            for state,pop in initialState_str.items():
                initialState[process_sympy(state)] = pop # convert string into SymPy symbol
        #print("Initial State is " + str(initialState) )
        MAParams['initialState'] = initialState
        
        # init the max-time
        if (maxTime == "Auto" or maxTime <= 0):
            maxTime = MuMoTdefault._maxTime
        timeLimitMax = max(maxTime, MuMoTdefault._timeLimits[1])
        MAParams["maxTime"] = (maxTime, MuMoTdefault._timeLimits[0], timeLimitMax, MuMoTdefault._timeStep)
        # init the random seed 
        if (randomSeed == "Auto" or randomSeed <= 0 or randomSeed > MAX_RANDOM_SEED):
            randomSeed = np.random.randint(MAX_RANDOM_SEED)
#             print("Automatic Random Seed set to " + str(randomSeed) )
        MAParams['randomSeed'] = randomSeed
        # check validity of the network type
        if _decodeNetworkTypeFromString(netType) == None: return # terminating the process if the input argument is wrong
        MAParams['netType'] = netType
#         print("Network type set to " + str(MAParams['netType']) ) 

        # Setting some default values 
        ## @todo add possibility to customise these values from input line
        MAParams['netParam'] = kwargs.get('netParam', (0, 1.0, 1.0, 1.0)) # these value will be overwritten by _update_net_params()
        MAParams['motionCorrelatedness'] = kwargs.get('motionCorrelatedness', (0.5, 0.0, 1.0, 0.05))
        MAParams['particleSpeed'] = kwargs.get('particleSpeed', (0.01, 0.0, 0.1, 0.005))
        MAParams['visualisationType'] = kwargs.get('visualisationType', 'evo') # default view is time-evolution
        MAParams['plotProportions'] = kwargs.get('plotProportions', False)
        MAParams['timestepSize'] = kwargs.get('timestepSize', (1, 0.01, 1, 0.01) )
        MAParams['showTrace'] = kwargs.get('showTrace', netType == 'dynamic')
        MAParams['showInteractions'] = kwargs.get('showInteractions', False)
        ## realtimePlot flag (TRUE = the plot is updated each timestep of the simulation; FALSE = it is updated once at the end of the simulation)
        MAParams['realtimePlot'] = kwargs.get('realtimePlot', False)
        
        # construct controller
        paramValues = []
        paramNames = [] 
        for rate in self._rates:
            paramValues.append((MuMoTdefault._initialRateValue, MuMoTdefault._rateLimits[0], MuMoTdefault._rateLimits[1], MuMoTdefault._rateStep))
            paramNames.append(str(rate))

        viewController = MuMoTmultiagentController(paramValues, paramNames, self._ratesLaTeX, False, MAParams, **kwargs)
        # Get the default network values assigned from the controller
        #MAParams['netParam'] = viewController._widgetsExtraParams['netParam'].value ## @todo use defaults without relying on the controller
        
        modelView = MuMoTmultiagentView(self, viewController, MAParams, **kwargs)
        viewController.setView(modelView)
#         viewController._setReplotFunction(modelView._plot_timeEvolution(self._reactants, self._rules))
        viewController._setReplotFunction(modelView._plot_timeEvolution, modelView._redrawOnly)
        #viewController._widgetsExtraParams['netType'].value.observe(modelView._update_net_params, 'value') #netType is special

        return viewController

    def SSA(self, initialState="Auto", maxTime="Auto", randomSeed="Auto", **kwargs):
        ssaParams = {}
        if initialState=="Auto":
            first = True
            initialState = {}
            (allReactants, allConstantReactants) = self._getAllReactants()
            for reactant in allReactants | allConstantReactants: #self._reactants|self._constantReactants:
                if first:
#                     print("Automatic Initial State sets " + str(MuMoTdefault._agents) + " agents in state " + str(reactant) )
                    initialState[reactant] = MuMoTdefault._agents
                    first = False
                else:
                    initialState[reactant] = 0
        else:
            ## @todo check if the Initial State has valid length and positive values
#             print("TODO: check if the Initial State has valid length and positive values")
#             initialState_str = ast.literal_eval(initialState) # translate string into dict
            initialState_str = initialState
            initialState = {}
            for state,pop in initialState_str.items():
                initialState[process_sympy(state)] = pop # convert string into SymPy symbol
        #print("Initial State is " + str(initialState) )
        ssaParams['initialState'] = initialState
        
        if (maxTime == "Auto" or maxTime <= 0):
            maxTime = MuMoTdefault._maxTime
        timeLimitMax = max(maxTime, MuMoTdefault._timeLimits[1])
        ssaParams["maxTime"] = (maxTime, MuMoTdefault._timeLimits[0], timeLimitMax, MuMoTdefault._timeStep)
        if (randomSeed == "Auto" or randomSeed <= 0 or randomSeed > MAX_RANDOM_SEED):
            randomSeed = np.random.randint(MAX_RANDOM_SEED)
#             print("Automatic Random Seed set to " + str(randomSeed) )
        ssaParams['randomSeed'] = randomSeed
        ssaParams['visualisationType'] = 'evo'
        ssaParams['plotProportions'] = kwargs.get('plotProportions', False)
        ssaParams['realtimePlot'] = kwargs.get('realtimePlot', False)
            
        # construct controller
        paramValues = []
        paramNames = [] 
        #paramValues.extend( [initialState, netType, maxTime] )
        for rate in self._rates:
            paramValues.append((MuMoTdefault._initialRateValue, MuMoTdefault._rateLimits[0], MuMoTdefault._rateLimits[1], MuMoTdefault._rateStep))
            paramNames.append(str(rate))

        viewController = MuMoTSSAController(paramValues=paramValues, paramNames=paramNames, paramLabelDict=self._ratesLaTeX, continuousReplot=False, ssaParams=ssaParams, **kwargs)
        #paramDict = {}
        #paramDict['initialState'] = initialState
        modelView = MuMoTSSAView(self, viewController, ssaParams, **kwargs)
        viewController.setView(modelView)
        #modelView._plot_timeEvolution()
        
#         viewController._setReplotFunction(modelView._plot_timeEvolution(self._reactants, self._rules))
        viewController._setReplotFunction(modelView._plot_timeEvolution, modelView._redrawOnly)
        
        return viewController

#         if stateVariable2 == None:
#             # 2-d bifurcation diagram
#             # create widgets
# 
#             if self._systemSize != None:
#                 ## @todo: shouldn't allow system size to be varied?
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
#             self._errorMessage = widget                                ## @todo: add to __init__()
#             display(self._errorMessage)
#             
#             # Prepare the system to start close to a steady state
#             self._bifurcationParameter = bifurcationParameter          ## @todo: remove hack (bifurcation parameter for multiple possible bifurcations needs to be stored in self)
#             self._stateVariable1 = stateVariable1                      ## @todo: remove hack (state variable for multiple possible bifurcations needs to be stored in self)
# #            self._pyDSode.set(pars = {bifurcationParameter: 0} )       # Lower bound of the bifurcation parameter (@todo: set dynamically)
# #            self._pyDSode.set(pars = self._pyDSmodel.pars )       # Lower bound of the bifurcation parameter (@todo: set dynamically)
# #            self._pyDSode.pars = {bifurcationParameter: 0}             # Lower bound of the bifurcation parameter (@todo: set dynamically?)
#             initconds = {stateVariable1: self._paramDict[str(self._systemSize)] / len(self._reactants)} ## @todo: guess where steady states are?
#             for reactant in self._reactants:
#                 if str(reactant) != stateVariable1:
#                     initconds[str(reactant)] = self._paramDict[str(self._systemSize)] / len(self._reactants)
# #            self._pyDSmodel.ics = initconds
#             self._pyDSmodel.ics      = {'A': 0.1, 'B': 0.9 }    ## @todo: replace            
# #            self._pyDSode.set(ics = initconds)
#             self._pyDSode = dst.Generator.Vode_ODEsystem(self._pyDSmodel)  ## @todo: add to __init__()
#             self._pyDSode.set(pars = {bifurcationParameter: 5} )                       ## @todo remove magic number
#             self._pyDScont = dst.ContClass(pyDSode)              # Set up continuation class (@todo: add to __init__())
#             ## @todo: add self._pyDScontArgs to __init__()
#             self._pyDScontArgs = dst.args(name='EQ1', type='EP-C')     # 'EP-C' stands for Equilibrium Point Curve. The branch will be labeled 'EQ1'.
#             self._pyDScontArgs.freepars     = [bifurcationParameter]   # control parameter(s) (should be among those specified in self._pyDSmodel.pars)
#             self._pyDScontArgs.MaxNumPoints = 450                      # The following 3 parameters are set after trial-and-error @todo: how to automate this?
#             self._pyDScontArgs.MaxStepSize  = 1e-1
#             self._pyDScontArgs.MinStepSize  = 1e-5
#             self._pyDScontArgs.StepSize     = 2e-3
#             self._pyDScontArgs.LocBifPoints = ['LP', 'BP']                    ## @todo WAS 'LP' (detect limit points / saddle-node bifurcations)
#             self._pyDScontArgs.SaveEigen    = True                     # to tell unstable from stable branches
# #            self._pyDScontArgs.CalcStab     = True
# 
#             plt.ion()
# #            self._bifurcation2Dfig = plt.figure(1)                     ## @todo: add to __init__()
#             self._pyDScont.newCurve(self._pyDScontArgs)
#             try:
#                 try:
#                     self._pyDScont['EQ1'].backward()
#                 except:
#                     self._errorMessage.value = 'Continuation failure'
#                 try:
#                     self._pyDScont['EQ1'].forward()                                  ## @todo: how to choose direction?
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

    ## get the pair of set (reactants, constantReactants). This method is necessary to have all reactants also after a substitution (to set the system size) has occurred
    def _getAllReactants(self):
        allReactants = set()
        allConstantReactants = set()
        for reaction in self._stoichiometry.values():
            for reactant,info in reaction.items():
                if (not reactant == 'rate') and (reactant not in allReactants) and (reactant not in allConstantReactants):
                    if info == 'const':
                        allConstantReactants.add(reactant)
                    else:
                        allReactants.add(reactant)
        return (allReactants, allConstantReactants)

    def _get_solutions(self):
        if self._solutions == None:
            self._solutions = solve(iter(self._equations.values()), self._reactants, force = False, positive = False, set = False)
        return self._solutions

    ## general controller constructor with all rates as free parameters
    def _controller(self, contRefresh, displayController = True, plotLimitsSlider = False, params = None, **kwargs):
        initialRateValue = INITIAL_RATE_VALUE ## @todo was 1 (choose initial values sensibly)
        rateLimits = (0, RATE_BOUND) ## @todo choose limit values sensibly
        rateStep = RATE_STEP ## @todo choose rate step sensibly                

        # construct controller
        paramValues = []
        paramNames = []  
        bifParam = kwargs.get('chooseBifParam', False)    
        bifParInitVal = kwargs.get('BifParInit', None) 
        
        for rate in self._rates:
            if str(rate) != bifParam:
                paramValues.append((initialRateValue, rateLimits[0], rateLimits[1], rateStep))
                paramNames.append(str(rate))
            else:
                if bifParInitVal:
                    paramValues.append((bifParInitVal, rateLimits[0], rateLimits[1], rateStep))
                else:
                    paramValues.append((initialRateValue, rateLimits[0], rateLimits[1], rateStep))
                #paramNames.append(str(rate)+'_{init}')
                paramNames.append(latex(Symbol(str(rate)+'_init')))
        for reactant in self._constantReactants:
            paramValues.append((initialRateValue, rateLimits[0], rateLimits[1], rateStep))
#            paramNames.append('(' + latex(reactant) + ')')
            paramNames.append(str(reactant))
#         if kwargs.get('showInitSV', False) == True:
#             initialInitCondValue = INITIAL_COND_INIT_VAL
#             initialCondLimits = (0, INITIAL_COND_INIT_BOUND)
#             for reactant in self._reactants:
#                 if reactant not in self._constantReactants:
#                     paramNames.append(latex(Symbol('Phi^0_'+str(reactant))))
#                     paramValues.append((initialInitCondValue, initialCondLimits[0], initialCondLimits[1], rateStep))            

        if kwargs.get('showInitSV', False) == True:
            initialCondLimits = (0, INITIAL_COND_INIT_BOUND)
            initCondsSV = kwargs.get('initCondsSV', False)
            for reactant in self._reactants:
                if reactant not in self._constantReactants:
                    paramNames.append(latex(Symbol('Phi^0_'+str(reactant))))
                    if initCondsSV != False:
                        # check input of initCondsSV
                        if str(reactant) not in initCondsSV:
                            initialInitCondValue = INITIAL_COND_INIT_VAL
                        else:
                            initialInitCondValue = initCondsSV[str(reactant)]
                    else:
                        initialInitCondValue = INITIAL_COND_INIT_VAL   
                    paramValues.append((initialInitCondValue, initialCondLimits[0], initialCondLimits[1], rateStep))            
                        
        systemSizeSlider = kwargs.get('showNoise', False)
        viewController = MuMoTcontroller(paramValues, paramNames, self._ratesLaTeX, contRefresh, plotLimitsSlider, systemSizeSlider, params, **kwargs)

        return viewController

    ## derive the single-agent rules from reaction rules
    def _getSingleAgentRules(self):
        # create the empty structure
        self._agentProbabilities = {}
        (allReactants, allConstantReactants) = self._getAllReactants()
        for reactant in allReactants | allConstantReactants | {EMPTYSET_SYMBOL} :
            self._agentProbabilities[reactant] = []
            
        # populate the created structure
        for rule in self._rules:
            targetReact = []
            # check if constant reactants are not changing state
            # WARNING! if the checks hereafter are changed, it might be necessary to modify the way in which network-simulations are disabled (atm only on EMPTYSET_SYMBOL presence because (A) -> B are not allowed)
            for idx, reactant in enumerate(rule.lhsReactants):
                if reactant in allConstantReactants: #self._constantReactants:
                    if not (rule.rhsReactants[idx] == reactant or rule.rhsReactants[idx] == EMPTYSET_SYMBOL):
                        errorMsg = 'In rule ' + str(rule.lhsReactants) + ' -> '  + str(rule.rhsReactants) + ' constant reactant are not properly used. ' \
                                    'Constant reactants must either match the same constant reactant or the EMPTYSET on the right-handside. \n' \
                                    'NOTE THAT ORDER MATTERS: MuMoT assumes that first reactant on left-handside becomes first reactant on right-handside and so on for sencond and third...'
                        print(errorMsg)
                        raise ValueError(errorMsg)
                elif rule.rhsReactants[idx] in allConstantReactants:
                    errorMsg = 'In rule ' + str(rule.lhsReactants) + ' -> '  + str(rule.rhsReactants) + ' constant reactant are not properly used.' \
                                    'Constant reactants appears on the right-handside and not on the left-handside. \n' \
                                    'NOTE THAT ORDER MATTERS: MuMoT assumes that first reactant on left-handside becomes first reactant on right-handside and so on for sencond and third...'
                    print(errorMsg)
                    raise ValueError(errorMsg)
                
                if reactant == EMPTYSET_SYMBOL:
                    targetReact.append(rule.rhsReactants[idx])
            
            # creating a rule for the first non-empty element (on the lhs) of the rule (if all empty, it uses an empty)
            idx = 0
            while idx < len(rule.lhsReactants)-1:
                reactant = rule.lhsReactants[idx]
                if not reactant == EMPTYSET_SYMBOL:
                    break
                else:
                    idx += 1
                
            # create list of other reactants on the left-handside that reacts with the considered reactant
            otherReact = []
            otherTargets = []
            for idx2, react2 in enumerate(rule.lhsReactants):
                if idx == idx2 or react2 == EMPTYSET_SYMBOL: continue
                otherReact.append(react2)
                if react2 in allConstantReactants:
                    otherTargets.append(react2)
                else:
                    otherTargets.append(rule.rhsReactants[idx2])
            
            # the target reactant for the first non-empty item is the same index item plus all the reactants coming from empty-sets
            if reactant in allConstantReactants:
                targetReact.append(reactant)
            elif not reactant == EMPTYSET_SYMBOL:
                targetReact.append(rule.rhsReactants[idx])
            
            # create a new entry
            self._agentProbabilities[reactant].append( [otherReact, rule.rate, targetReact, otherTargets] )
        #print( self._agentProbabilities )

    def _check_state_variables(self, stateVariable1, stateVariable2, stateVariable3 = None):
        if process_sympy(stateVariable1) in self._reactants and process_sympy(stateVariable2) in self._reactants and (stateVariable3 == None or process_sympy(stateVariable3) in self._reactants):
            if stateVariable1 != stateVariable2 and stateVariable1 != stateVariable3 and stateVariable2 != stateVariable3:
                return True
            else:
                print('State variables cannot be the same')
                return False
        else:
            print('Invalid reactant provided as state variable')
            return False


    ## lambdify sympy equations for numerical integration, plotting, etc.
    def _getFuncs(self):
#         if self._systemSize == None:
#             assert false ## @todo is this necessary?
        if self._funcs == None:
            argList = []
            for reactant in self._reactants:
                argList.append(reactant)
            for reactant in self._constantReactants:
                argList.append(reactant)
            for rate in self._rates:
                argList.append(rate)
            if self._systemSize != None:
                argList.append(self._systemSize)
            self._args = tuple(argList)
            self._funcs = {}
            for equation in self._equations:
                f = lambdify(self._args, self._equations[equation], "math")
                self._funcs[equation] = f
            
        return self._funcs
    
    ## get tuple to evalute functions returned by _getFuncs with, for 2d field-based plots
    def _getArgTuple2d(self, argNames, argValues, argDict, stateVariable1, stateVariable2, X, Y):
        argList = []
        for arg in self._args:
            if arg == stateVariable1:
                argList.append(X)
            elif arg == stateVariable2:
                argList.append(Y)
            elif arg == self._systemSize:
                argList.append(1) ## @todo: system size set to 1
            else:
                argList.append(argDict[arg])
            
        return tuple(argList)

    ## get tuple to evalute functions returned by _getFuncs with, for 2d field-based plots
    def _getArgTuple3d(self, argNames, argValues, argDict, stateVariable1, stateVariable2, stateVariable3, X, Y, Z):
        argList = []
        for arg in self._args:
            if arg == stateVariable1:
                argList.append(X)
            elif arg == stateVariable2:
                argList.append(Y)
            elif arg == stateVariable3:
                argList.append(Z)                
            elif arg == self._systemSize:
                argList.append(1) ## @todo: system size set to 1
            else:
                argList.append(argDict[arg])
            
        return tuple(argList)

    ## get tuple to evalute functions returned by _getFuncs with
    def _getArgTuple(self, argNames, argValues, argDict, reactants, reactantValues):
        assert false # need to work this out
        argList = []
#         for arg in self._args:
#             if arg == stateVariable1:
#                 argList.append(X)
#             elif arg == stateVariable2:
#                 argList.append(Y)
#             elif arg == stateVariable3:
#                 argList.append(Z)                
#             elif arg == self._systemSize:
#                 argList.append(1) ## @todo: system size set to 1
#             else:
#                 argList.append(argDict[arg])
            
        return tuple(argList)


    ## render LaTeX source to local image file                                 
    def _localLaTeXimageFile(self, source):
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
        self._constantSystemSize = True
        self._reactantsLaTeX = None
        self._rates = set()
        self._ratesLaTeX= None
        self._equations = {}
        self._stoichiometry = {}
        self._pyDSmodel = None
        self._dot = None
        if not os.path.isdir(self._tmpdirpath):
            os.mkdir(self._tmpdirpath)
            os.system('chmod' + self._tmpdirpath + 'u+rwx')
            os.system('chmod' + self._tmpdirpath + 'g-rwx')
            os.system('chmod' + self._tmpdirpath + 'o+rwx')
        self._tmpdir = tempfile.TemporaryDirectory(dir = self._tmpdirpath)
        self._tmpfiles = []
        
    def __del__(self):
        ## @todo: check when this is invoked
        for tmpfile in self._tmpfiles:
            del tmpfile
        del self._tmpdir

## class describing a single reaction rule
class _Rule:
    lhsReactants = []
    rhsReactants = []
    rate = ""
    def __init__(self):
        self.lhsReactants = []
        self.rhsReactants = []
        self.rate = ""

## class describing a controller for a model view
class MuMoTcontroller:
    _view = None
    ## dictionary of LaTeX labels for widgets, with parameter name as key
    _paramLabelDict = None
    ## dictionary of controller widgets only for the free parameters of the model, with parameter name as key
    _widgetsFreeParams = None
    ## dictionary of controller widgets for the special parameters (e.g., simulation length, initial state), with parameter name as key
    _widgetsExtraParams = None
    ## dictionary of controller widgets, with parameter that influence only the plotting and not the computation
    _widgetsPlotOnly = None
    ## list keeping the order of the extra-widgets (_widgetsExtraParams and _widgetsPlotOnly)
    _extraWidgetsOrder = None
    ## replot function widgets have been assigned (for use by MuMoTmultiController)
    _replotFunction = None
    ## widget for simple error messages to be displayed to user during interaction
    _errorMessage = None
    ## plot limits slider widget
    _plotLimitsWidget = None ## @todo: is it correct that this is a variable of the general MuMoTcontroller?? it might be simply added in the _widgetsPlotOnly dictionary
    ## system size slider widget
    _systemSizeWidget = None    
    ## bookmark button widget
    _bookmarkWidget = None
    ## progress bar @todo: is this best put in base class when it is not always used?
    _progressBar = None
    ## used two progress bars, otherwise the previous cell bar (where controller is created) does not react anymore  @todo: is this best put in base class when it is not always used?
    _progressBar_multi = None 

    def __init__(self, paramValues, paramNames, paramLabelDict = {}, continuousReplot = False, plotLimits = False, systemSize = False, params = None, **kwargs):
        silent = kwargs.get('silent', False)
        self._silent = silent
        self._paramLabelDict = paramLabelDict
        self._widgetsFreeParams = {}
        self._widgetsExtraParams = {}
        self._widgetsPlotOnly = {}
        self._extraWidgetsOrder = []
        unsortedPairs = zip(paramNames, paramValues)
        fixedParams = None
 #       fixedParamsDecoded = None
        if params is not None:
            fixedParams, foo = zip(*params)
            fixedParamsDecoded = []
            for fixedParam in fixedParams:
                if fixedParam == 'plotLimits' or fixedParam == 'systemSize':
                    pass
                else:
                    expr = process_sympy(fixedParam.replace('\\\\','\\'))
                    atoms = expr.atoms()
                    if len(atoms) > 1:
                        raise SyntaxError("Non-singleton parameter name in parameter " + fixedParam)
                    for atom in atoms:
                        # parameter name should contain a single atom
                        pass
                    fixedParamsDecoded.append(str(atom))    
## @todo: refactor the above to use _process_params (below didn't work when first tried)                                
#        fixedParamsDecoded = None
#        if params is not None:
#            (fixedParamsDecoded, foo) = _process_params(params)
        for pair in sorted(unsortedPairs):
            if fixedParams is None or pair[0] not in fixedParamsDecoded: 
                widget = widgets.FloatSlider(value = pair[1][0], min = pair[1][1], 
                                             max = pair[1][2], step = pair[1][3], 
                                             description = r'\(' + self._paramLabelDict.get(pair[0],pair[0]) + r'\)', 
                                             continuous_update = continuousReplot)
                self._widgetsFreeParams[pair[0]] = widget
                if not(self._silent):
                    display(widget)
        if plotLimits:
            if fixedParams is None or 'plotLimits' not in fixedParams:             
                ## @todo: remove hard coded values and limits
                self._plotLimitsWidget = widgets.FloatSlider(value = 1.0, min = 1.0, 
                                             max = 10.0, step = 0.5, 
                                             description = "Plot limits", 
                                             continuous_update = False)
                if not silent:
                    display(self._plotLimitsWidget)
                
        if systemSize:
            if fixedParams is None or 'systemSize' not in fixedParams:
                ## @todo: remove hard coded values and limits
                self._systemSizeWidget = widgets.IntSlider(value = 5, min = 5, 
                                             max = 100, step = 1, 
                                             description = "System size", 
                                             continuous_update = False)
                if not silent:  
                    display(self._systemSizeWidget)
                        
        self._bookmarkWidget = widgets.Button(description='', disabled=False, button_style='', tooltip='Paste bookmark to log', icon='fa-bookmark')
        self._bookmarkWidget.on_click(self._print_standalone_view_cmd)
        bookmark = kwargs.get('bookmark', True)
        if not silent and bookmark:
            display(self._bookmarkWidget)

        widget = widgets.HTML()
        widget.value = ''
        self._errorMessage = widget
        if not silent and bookmark:
            display(self._errorMessage)

    def _print_standalone_view_cmd(self, foo):
        self._errorMessage.value = "Pasted bookmark to log - view with showLogs(tail = True)"
        self._view._print_standalone_view_cmd()

    
    ## set the functions that must be triggered when the widgets are changed.
    ## @param[in]    recomputeFunction    The function to be called when recomputing is necessary 
    ## @param[in]    redrawFunction    The function to be called when only redrawing (relying on previous computation) is sufficient 
    def _setReplotFunction(self, recomputeFunction, redrawFunction=None):
        self._replotFunction = recomputeFunction
        for widget in self._widgetsFreeParams.values():
            #widget.on_trait_change(recomputeFunction, 'value')
            widget.observe(recomputeFunction, 'value')
        for widget in self._widgetsExtraParams.values():
            widget.observe(recomputeFunction, 'value')
        if self._plotLimitsWidget != None:
            self._plotLimitsWidget.observe(recomputeFunction, 'value')
        if self._systemSizeWidget != None:
            self._systemSizeWidget.observe(recomputeFunction, 'value')
        if redrawFunction != None:
            for widget in self._widgetsPlotOnly.values():
                widget.observe(redrawFunction, 'value')

    ## create and display the "Advanced options" tab
    def _displayAdvancedOptionsTab(self):
        advancedWidgets = []
        for widgetName in self._extraWidgetsOrder:
            if self._widgetsExtraParams.get(widgetName):
                advancedWidgets.append(self._widgetsExtraParams[widgetName])
            elif self._widgetsPlotOnly.get(widgetName):
                advancedWidgets.append(self._widgetsPlotOnly[widgetName])
            else:
                print("WARNING! In the _extraWidgetsOrder is listed the widget " + widgetName + " which is although not found in _widgetsExtraParams or _widgetsPlotOnly")
#                 for widget in self._widgetsExtraParams.values():
#                     advancedWidgets.append(widget)
#                 for widget in self._widgetsPlotOnly.values():
#                     advancedWidgets.append(widget)
        advancedPage = widgets.Box(children=advancedWidgets)
        advancedPage.layout.flex_flow = 'column'
        advancedOpts = widgets.Accordion(children=[advancedPage]) #, selected_index=-1)
        advancedOpts.set_title(0, 'Advanced options')
        advancedOpts.selected_index = None
        display(advancedOpts)


    def setView(self, view):
        self._view = view

    def showLogs(self, tail = False):
        self._view.showLogs(tail)
        
    def multirun(self, iterations, randomSeeds="Auto", visualisationType="evo", downloadData=False):
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
        self._view._visualisationType = visualisationType
        
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


    def _setErrorWidget(self, errorWidget):
        self._errorMessage = errorWidget

            
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

## class describing a controller for multiagent views
class MuMoTSSAController(MuMoTcontroller):
    
    def __init__(self, paramValues, paramNames, paramLabelDict, continuousReplot, ssaParams, **kwargs):
        MuMoTcontroller.__init__(self, paramValues, paramNames, paramLabelDict, continuousReplot, systemSize=True, **kwargs)
        
        initialState = ssaParams['initialState']
        for state,pop in initialState.items():
            widget = widgets.FloatSlider(value = pop,
                                         min = min(pop, MuMoTdefault._agentsLimits[0]), 
                                         max = max(pop, MuMoTdefault._agentsLimits[1]),
                                         step = MuMoTdefault._agentsStep,
                                         description = "State " + str(state),
                                         style = {'description_width': 'initial'},
                                         continuous_update = continuousReplot)
            self._widgetsExtraParams['init'+str(state)] = widget
            #advancedWidgets.append(widget)
            
        # Max time slider
        maxTime = ssaParams['maxTime']
        widget = widgets.FloatSlider(value = maxTime[0], min = maxTime[1], 
                                         max = maxTime[2], step = maxTime[3], 
                                         description = 'Simulation time:',
                                         style = {'description_width': 'initial'},
                                         #layout=widgets.Layout(width='50%'),
                                         disabled=False,
                                         continuous_update = continuousReplot) 
        self._widgetsExtraParams['maxTime'] = widget
        #advancedWidgets.append(widget)
        
        # Random seed input field
        widget = widgets.IntText(
            value=ssaParams['randomSeed'],
            description='Random seed:',
            style = {'description_width': 'initial'},
            disabled=False
        )
        self._widgetsExtraParams['randomSeed'] = widget
        #advancedWidgets.append(widget)
        
        ## Toggle buttons for plotting style 
        plotToggle = widgets.ToggleButtons(
            options=[('Temporal evolution', 'evo'), ('Final distribution', 'final')],
            value = ssaParams['visualisationType'],
            description='Plot:',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Population change over time', 'Population distribution in each state at final timestep'],
        #     icons=['check'] * 3
        )
        self._widgetsPlotOnly['visualisationType'] = plotToggle
        #advancedWidgets.append(plotToggle)
        
        ## Checkbox for proportions or full populations plot 
        widget = widgets.Checkbox(
            value = ssaParams.get('plotProportions',False),
            description='Plot population proportions',
            disabled = False
        )
        self._widgetsPlotOnly['plotProportions'] = widget
        
        ## Checkbox for realtime plot update
        widget = widgets.Checkbox(
            value = ssaParams.get('realtimePlot',False),
            description='Runtime plot update',
            disabled = False
        )
        self._widgetsExtraParams['realtimePlot'] = widget
        #advancedWidgets.append(widget)
        
        # define the widget order
        for state,pop in initialState.items():
            self._extraWidgetsOrder.append('init'+str(state))
        self._extraWidgetsOrder.append('maxTime')
        self._extraWidgetsOrder.append('randomSeed')
        self._extraWidgetsOrder.append('visualisationType')
        self._extraWidgetsOrder.append('plotProportions')
        self._extraWidgetsOrder.append('realtimePlot')
        
        # add widgets to the Advanced options tab
        if not self._silent:
            self._displayAdvancedOptionsTab()
        
        # Loading bar (useful to give user progress status for long executions)
        self._progressBar = widgets.FloatProgress(
            value=0,
            min=0,
            max=self._widgetsExtraParams['maxTime'].value,
            #step=1,
            description='Loading:',
            bar_style='success', # 'success', 'info', 'warning', 'danger' or ''
            orientation='horizontal'
        )
        if not self._silent:
            display(self._progressBar)
        

## class describing a controller for multiagent views
class MuMoTmultiagentController(MuMoTcontroller):

    def __init__(self, paramValues, paramNames, paramLabelDict, continuousReplot, MAParams, **kwargs):
        MuMoTcontroller.__init__(self, paramValues, paramNames, paramLabelDict, continuousReplot, systemSize=True, **kwargs)
        
        initialState = MAParams['initialState']
        for state,pop in initialState.items():
            widget = widgets.FloatSlider(value = pop,
                                         min = min(pop, MuMoTdefault._agentsLimits[0]), 
                                         max = max(pop, MuMoTdefault._agentsLimits[1]),
                                         step = MuMoTdefault._agentsStep,
                                         description = "State " + str(state), 
                                         style = {'description_width': 'initial'},
                                         continuous_update = continuousReplot)
            self._widgetsExtraParams['init'+str(state)] = widget
            #advancedWidgets.append(widget)
        
        # Max time slider
        maxTime = MAParams['maxTime']
        widget = widgets.FloatSlider(value = maxTime[0], min = maxTime[1], 
                                         max = maxTime[2], step = maxTime[3], 
                                         description = 'Simulation time:',
                                         disabled=False,
                                         style = {'description_width': 'initial'},
                                         layout=widgets.Layout(width='70%'),
                                         continuous_update = continuousReplot) 
        self._widgetsExtraParams['maxTime'] = widget
        #advancedWidgets.append(widget)
        
        ## Network type dropdown selector
        netDropdown = widgets.Dropdown( 
            options=[('Full graph', NetworkType.FULLY_CONNECTED), 
                     ('Erdos-Renyi', NetworkType.ERSOS_RENYI),
                     ('Barabasi-Albert', NetworkType.BARABASI_ALBERT),
                     ## @todo: add network topology generated by random points in space
                     ('Moving particles', NetworkType.DYNAMIC)
                     ],
            description='Network topology:',
            value = _decodeNetworkTypeFromString(MAParams['netType']), 
            style = {'description_width': 'initial'},
            disabled=False
        )
        netDropdown.observe(self._update_net_params, 'value')
        self._widgetsExtraParams['netType'] = netDropdown
        #advancedWidgets.append(netDropdown)
        
        # Network connectivity slider
        netParam = MAParams['netParam']
        widget = widgets.FloatSlider(value = netParam[0],
                                    min = netParam[1], 
                                    max = netParam[2],
                                    step = netParam[3],
                            description = 'Network connectivity parameter', 
                            style = {'description_width': 'initial'},
                            layout=widgets.Layout(width='50%'),
                            continuous_update = continuousReplot,
                            disabled=False
        )
        self._widgetsExtraParams['netParam'] = widget
        #advancedWidgets.append(widget)
        
        # Agent speed
        particleSpeed = MAParams['particleSpeed']
        widget = widgets.FloatSlider(value = particleSpeed[0],
                                     min = particleSpeed[1], max = particleSpeed[2], step=particleSpeed[3],
                            description = 'Particle speed', 
                            style = {'description_width': 'initial'},
                            layout=widgets.Layout(width='50%'),
                            continuous_update = continuousReplot,
                            disabled=False
        )
        self._widgetsExtraParams['particleSpeed'] = widget
        #advancedWidgets.append(widget)
        
        # Random walk correlatedness
        motionCorrelatedness = MAParams['motionCorrelatedness']
        widget = widgets.FloatSlider(value = motionCorrelatedness[0],
                                     min = motionCorrelatedness[1],
                                     max = motionCorrelatedness[2],
                                     step=motionCorrelatedness[3],
                            description = 'Correlatedness of the random walk',
                            layout=widgets.Layout(width='50%'),
                            style = {'description_width': 'initial'},
                            continuous_update = continuousReplot,
                            disabled=False
        )
        self._widgetsExtraParams['motionCorrelatedness'] = widget
        #advancedWidgets.append(widget)
        
        # Random seed input field
        widget = widgets.IntText(
            value=MAParams['randomSeed'],
            description='Random seed:',
            style = {'description_width': 'initial'},
            disabled=False
        )
        self._widgetsExtraParams['randomSeed'] = widget
        #advancedWidgets.append(widget)
        #display(widget)
        
        # Time scaling slider
        timestepSize = MAParams['timestepSize']
        widget =  widgets.FloatSlider(value = timestepSize[0],
                                    min = timestepSize[1], 
                                    max = timestepSize[2],
                                    step = timestepSize[3],
                            description = 'Timestep size', 
                            style = {'description_width': 'initial'},
                            layout=widgets.Layout(width='50%'),
                            continuous_update = continuousReplot
        )
        self._widgetsExtraParams['timestepSize'] = widget
        #advancedWidgets.append(widget)
        
        ## Toggle buttons for plotting style 
        plotToggle = widgets.ToggleButtons(
            #options={'Temporal evolution' : 'evo', 'Network' : 'graph', 'Final distribution' : 'final'},
            options=[('Temporal evolution','evo'), ('Network','graph'), ('Final distribution', 'final')],
            value = MAParams['visualisationType'],
            description='Plot:',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Population change over time', 'Network topology', 'Number of agents in each state at final timestep'],
        #     icons=['check'] * 3
        )
        self._widgetsPlotOnly['visualisationType'] = plotToggle
        #advancedWidgets.append(plotToggle)
        
        # Particle display checkboxes
        widget = widgets.Checkbox(
            value = MAParams['showTrace'],
            description='Show particle trace',
            disabled = False #not (self._widgetsExtraParams['netType'].value == NetworkType.DYNAMIC)
        )
        self._widgetsPlotOnly['showTrace'] = widget
        #advancedWidgets.append(widget)
        widget = widgets.Checkbox(
            value = MAParams['showInteractions'],
            description='Show communication links',
            disabled = False #not (self._widgetsExtraParams['netType'].value == NetworkType.DYNAMIC)
        )
        self._widgetsPlotOnly['showInteractions'] = widget
        #advancedWidgets.append(widget)
        
        ## Checkbox for proportions or full populations plot 
        widget = widgets.Checkbox(
            value = MAParams.get('plotProportions',False),
            description='Plot population proportions',
            disabled = False
        )
        self._widgetsPlotOnly['plotProportions'] = widget
        
        ## Checkbox for realtime plot update
        widget = widgets.Checkbox(
            value = MAParams.get('realtimePlot',False),
            description='Runtime plot update',
            disabled = False
        )
        self._widgetsExtraParams['realtimePlot'] = widget
        #advancedWidgets.append(widget)
        
        #self._update_net_params()
        
        # define the widget order
        for state,pop in initialState.items():
            self._extraWidgetsOrder.append('init'+str(state))
        self._extraWidgetsOrder.append('maxTime')
        self._extraWidgetsOrder.append('timestepSize')
        self._extraWidgetsOrder.append('netType')
        self._extraWidgetsOrder.append('netParam')
        self._extraWidgetsOrder.append('particleSpeed')
        self._extraWidgetsOrder.append('motionCorrelatedness')
        self._extraWidgetsOrder.append('randomSeed')
        self._extraWidgetsOrder.append('visualisationType')
        self._extraWidgetsOrder.append('showTrace')
        self._extraWidgetsOrder.append('showInteractions')
        self._extraWidgetsOrder.append('plotProportions')
        self._extraWidgetsOrder.append('realtimePlot')
        
        # add widgets to the Advanced options tab
        if not self._silent:
            self._displayAdvancedOptionsTab()
        
        # Loading bar (useful to give user progress status for long executions)
        self._progressBar = widgets.FloatProgress(
            value=0,
            min=0,
            max=self._widgetsExtraParams['maxTime'].value,
            step=1,
            description='Loading:',
            bar_style='success', # 'success', 'info', 'warning', 'danger' or ''
            orientation='horizontal'
        )
        if not self._silent:
            display(self._progressBar)

    ## updates the widgets related to the netType (it is linked --through observe()-- before the _view is created) 
    def _update_net_params(self, _=None):
        self._view._update_net_params()

    ## updates the widgets related to the netType
#     def _update_net_params(self, _=None):
#         print("RUNNIG UP_NET_PAR")
#         # oder of assignment is important (first, update the min and max, later, the value)
#         self._widgetsExtraParams['netParam'].unobserve_all()
#         if (self._widgetsExtraParams['netType'].value == NetworkType.FULLY_CONNECTED):
# #             self._widgetsExtraParams['netParam'].min = 0
# #             self._widgetsExtraParams['netParam'].max = 1
# #             self._widgetsExtraParams['netParam'].step = 1
# #             self._widgetsExtraParams['netParam'].value = 0
# #             self._widgetsExtraParams['netParam'].disabled = True
# #             self._widgetsExtraParams['netParam'].description = "None"
#             self._widgetsExtraParams['netParam'].layout.display = 'none'
#         elif (self._widgetsExtraParams['netType'].value == NetworkType.ERSOS_RENYI):
# #             self._widgetsExtraParams['netParam'].disabled = False
#             self._widgetsExtraParams['netParam'].layout.display = 'flex'
#             self._widgetsExtraParams['netParam'].min = 0.1
#             self._widgetsExtraParams['netParam'].max = 1
#             self._widgetsExtraParams['netParam'].step = 0.1
#             self._widgetsExtraParams['netParam'].value = 0.5
#             self._widgetsExtraParams['netParam'].description = "Network connectivity parameter (link probability)"
#         elif (self._widgetsExtraParams['netType'].value == NetworkType.BARABASI_ALBERT):
# #             self._widgetsExtraParams['netParam'].disabled = False
#             self._widgetsExtraParams['netParam'].layout.display = 'flex'
#             maxVal = self._view._systemSize-1
#             self._widgetsExtraParams['netParam'].min = 1
#             self._widgetsExtraParams['netParam'].max = maxVal
#             self._widgetsExtraParams['netParam'].step = 1
#             self._widgetsExtraParams['netParam'].value = min(maxVal, 3)
#             self._widgetsExtraParams['netParam'].description = "Network connectivity parameter (new edges)"            
#         elif (self._widgetsExtraParams['netType'].value == NetworkType.SPACE):
#             self._widgetsExtraParams['netParam'].value = -1
#         
#         if (self._widgetsExtraParams['netType'].value == NetworkType.DYNAMIC):
# #             self._widgetsExtraParams['netParam'].disabled = False
#             self._widgetsExtraParams['netParam'].layout.display = 'flex'
#             self._widgetsExtraParams['netParam'].min = 0.0
#             self._widgetsExtraParams['netParam'].max = 1
#             self._widgetsExtraParams['netParam'].step = 0.05
#             self._widgetsExtraParams['netParam'].value = 0.1
#             self._widgetsExtraParams['netParam'].description = "Interaction range"
# #             self._widgetsExtraParams['particleSpeed'].disabled = False
# #             self._widgetsExtraParams['motionCorrelatedness'].disabled = False
# #             self._widgetsPlotOnly['showTrace'].disabled = False
# #             self._widgetsPlotOnly['showInteractions'].disabled = False
#             self._widgetsExtraParams['particleSpeed'].layout.display = 'flex'
#             self._widgetsExtraParams['motionCorrelatedness'].layout.display = 'flex'
#             self._widgetsPlotOnly['showTrace'].layout.display = 'flex'
#             self._widgetsPlotOnly['showInteractions'].layout.display = 'flex'
#         else:
#             #self._widgetsExtraParams['particleSpeed'].disabled = True
# #             self._widgetsExtraParams['motionCorrelatedness'].disabled = True
# #             self._widgetsPlotOnly['showTrace'].disabled = True
# #             self._widgetsPlotOnly['showInteractions'].disabled = True
#             self._widgetsExtraParams['particleSpeed'].layout.display = 'none'
#             self._widgetsExtraParams['motionCorrelatedness'].layout.display = 'none'
#             self._widgetsPlotOnly['showTrace'].layout.display = 'none'
#             self._widgetsPlotOnly['showInteractions'].layout.display = 'none'
#         self._widgetsExtraParams['netParam'].observe(self._replotFunction, 'value')
    
    def downloadTimeEvolution(self):
        return self._downloadFile(self._view._latestResults[0])
    



## class describing a view on a model
class MuMoTview:
    ## Model view is on
    _mumotModel = None
    ## Figure/axis object to plot view to
    _figure = None
    ## Unique figure number
    _figureNum = None
    ## 3d axes? (False => 2d)
    _axes3d = None
    ## Controller that controls this view @todo - could become None
    _controller = None
    ## Summary logs of view behaviour
    _logs = None
    ## parameter names when used without controller
    _paramNames = None
    ## parameter values when used without controller
    _paramValues = None
    ## silent flag (TRUE = do not try to acquire figure handle from pyplot)
    _silent = None
    ## plot limits (for non-constant system size) @todo: not used?
    _plotLimits = None
    ## command name that generates this view
    _generatingCommand = None
    ## generating keyword arguments
    _generatingKwargs = None
    
    def __init__(self, model, controller, figure = None, params = None, **kwargs):
        self._silent = kwargs.get('silent', False)
        self._mumotModel = model
        self._controller = controller
        self._logs = []
        self._axes3d = False

        self._plotLimits = 1
        #self._plotLimits = 6 ## @todo: why this magic number?
        self._generatingKwargs = kwargs
        if params != None:
            (self._paramNames, self._paramValues) = _process_params(params)
        if not(self._silent):
            _buildFig(self, figure)
    

    def _resetErrorMessage(self):
        if self._controller != None:
            if not (self._silent):
                self._controller._errorMessage.value= ''

    def _showErrorMessage(self, message):
        if self._controller != None:
            self._controller._errorMessage.value = self._controller._errorMessage.value + message
        else:
            print(message)
            
            
    def _setLog(self, log):
        self._logs = log

    def _log(self, analysis):
        print("Starting", analysis, "with parameters ", end='')
        paramNames = []
        paramValues = []
        if self._controller != None:
            ## @todo: if the alphabetic order is not good, the view could store the desired order in (paramNames) when the controller is constructed
            for name in sorted(self._controller._widgetsFreeParams.keys()):
                paramNames.append(name)
                paramValues.append(self._controller._widgetsFreeParams[name].value)
            for name in sorted(self._controller._widgetsExtraParams.keys()):
                paramNames.append(name)
                paramValues.append(self._controller._widgetsExtraParams[name].value)
        if self._paramNames is not None:
            paramNames += map(str, self._paramNames)
            paramValues += self._paramValues
            ## @todo: in soloView, this does not show the extra parameters (we should make clearer what the use of showLogs) 

        for i in zip(paramNames, paramValues):
            print('(' + i[0] + '=' + repr(i[1]) + '), ', end='')
        print("at", datetime.datetime.now())


    def _print_standalone_view_cmd(self, includeParams = True):
        logStr = self._build_bookmark(includeParams)
        if not self._silent and logStr is not None:
            with io.capture_output() as log:
                print(logStr)    
            self._logs.append(log)
        else:
            return logStr


    def _get_bookmarks_params(self, refModel = None):
        if refModel is not None:
            model = refModel
        else:
            model = self._mumotModel
        logStr = "params = ["
        
        paramInitCheck = []
        for reactant in model._reactants:
            if reactant not in model._constantReactants:
                paramInitCheck.append(latex(Symbol('Phi^0_'+str(reactant))))
                 
        for name, value in self._controller._widgetsFreeParams.items():
#                name = name.replace('\\', '\\\\')
            if name in model._ratesLaTeX:
                name = model._ratesLaTeX[name]
            name = name.replace('(', '')
            name = name.replace(')', '')
            if name not in paramInitCheck:
                logStr += "('" + latex(name) + "', " + str(value.value) + "), "
        if self._paramNames != None:
            for name, value in zip(self._paramNames, self._paramValues):
                name= repr(name)
                if name in model._ratesLaTeX:
                    name = model._ratesLaTeX[name]
                name = name.replace('(', '')
                name = name.replace(')', '')
                logStr += "('" + latex(name) + "', " + str(value) + "), "
        logStr += "('plotLimits', " + str(self._getPlotLimits()) + "), "
        logStr += "('systemSize', " + str(self._getSystemSize()) + "), "
#                if len(self._controller._widgetsFreeParams.items()) > 0:
        logStr = logStr[:-2] # throw away last ", "
        logStr += "]"
                
        return logStr        


    def _build_bookmark(self):
        self._resetErrorMessage()
        self._showErrorMessage("Bookmark functionality not implemented for class " + self._generatingCommand)

        return



    def _getPlotLimits(self, defaultLimits = 1):
        # if we don't do the check in this order setting plot limits via params will not work in multi controllers 
        if self._paramNames is not None and 'plotLimits' in self._paramNames:
            ## @todo: this is crying out to be refactored as a dictionary - no time just now
            plotLimits = self._paramValues[self._paramNames.index('plotLimits')]
        elif self._controller is not None and self._controller._plotLimitsWidget is not None:
            plotLimits = self._controller._plotLimitsWidget.value
        else:
            plotLimits = defaultLimits
            
        return plotLimits


    def _getSystemSize(self, defaultSize = 1):
        # if we don't do the check in this order setting system size via params will not work in multi controllers
        if self._paramNames is not None and 'systemSize' in self._paramNames:
            ## @todo: this is crying out to be refactored as a dictionary - no time just now
            systemSize = self._paramValues[self._paramNames.index('systemSize')]
        elif self._controller is not None and self._controller._systemSizeWidget is not None:
            systemSize = self._controller._systemSizeWidget.value
        else:
            systemSize = defaultSize
            
        return systemSize

    ## gets and returns names and values from widgets
    def _get_argDict(self):
        #plotLimits = self._getPlotLimits()
        #systemSize = Symbol('systemSize')
        #argDict[systemSize] = self._getSystemSize()
        paramNames = []
        paramValues = []
        if self._controller is not None:
            for name, value in self._controller._widgetsFreeParams.items():
                # throw away formatting for constant reactants
                #name = name.replace('(','')
                #name = name.replace(')','')
                
                paramNames.append(name)
                paramValues.append(value.value)
        if self._paramNames is not None:
            paramNames += map(str, self._paramNames)
            paramValues += self._paramValues   
                 
        #funcs = self._mumotModel._getFuncs()
        argNamesSymb = list(map(Symbol, paramNames))
        argDict = dict(zip(argNamesSymb, paramValues))
        #for key in argDict:
        #    if key in self._mumotModel._constantReactants:
        #        argDict[Symbol('Phi_'+str(key))] = argDict.pop(key)
                
        if self._mumotModel._systemSize:
            argDict[self._mumotModel._systemSize] = self._getSystemSize()
        else:
            systemSize = Symbol('systemSize')
            argDict[systemSize] = self._getSystemSize()
            
        return argDict
    
    def _getInitCondsFromSlider(self):
        #if self._controller != None:
        paramNames = []
        paramValues = []
        for reactant in self._mumotModel._reactants:
            if reactant not in self._mumotModel._constantReactants:
                for name, value in self._controller._widgetsFreeParams.items():
                    if name == latex(Symbol('Phi^0_'+str(reactant))):
                        paramNames.append(name)
                        paramValues.append(value.value)
                
        argNamesSymb = list(map(Symbol, paramNames))
        argDict = dict(zip(argNamesSymb, paramValues))
        
        return argDict


    def _get_fixedPoints2d(self):
#         #plotLimits = self._controller._getPlotLimits()
#         paramNames = []
#         paramValues = []
#         if self._controller is not None:
#             for name, value in self._controller._widgetsFreeParams.items():
#                 # throw away formatting for constant reactants
# #                 name = name.replace('(','')
# #                 name = name.replace(')','')
#                 paramNames.append(name)
#                 paramValues.append(value.value)
#         if self._paramNames is not None:
#             paramNames += map(str, self._paramNames)
#             paramValues += self._paramValues            
#         funcs = self._mumotModel._getFuncs()
#         argNamesSymb = list(map(Symbol, paramNames))
#         argDict = dict(zip(argNamesSymb, paramValues))
#         argDict[self._mumotModel._systemSize] = 1
       
        argDict = self._get_argDict()   
        for arg in argDict:
            if '_{init}' in str(arg):
                if str(arg)[0] == '\\':
                    arg1 = str(arg)[1:]
                    argReplace = Symbol(arg1.replace('_{init}', ''))
                    argDict[argReplace] = argDict.pop(arg)
                else:
                    argReplace = Symbol(str(arg).replace('_{init}', ''))
                    argDict[argReplace] = argDict.pop(arg)
            #if arg[0] == '\\':
            #    argDict[arg[1:]] = argDict.pop(arg)
        EQ1 = self._mumotModel._equations[self._stateVariable1].subs(argDict)
        EQ2 = self._mumotModel._equations[self._stateVariable2].subs(argDict)
        
        eps=1e-8
        EQsol = solve((EQ1, EQ2), (self._stateVariable1, self._stateVariable2), dict=True)

        for nn in range(len(EQsol)):
            if len(EQsol[nn]) != 2:
                self._showErrorMessage('Some or all solutions are NOT unique.')
                return None, None
        
        realEQsol = [{self._stateVariable1: re(EQsol[kk][self._stateVariable1]), self._stateVariable2: re(EQsol[kk][self._stateVariable2])} for kk in range(len(EQsol)) if (Abs(im(EQsol[kk][self._stateVariable1]))<=eps and Abs(im(EQsol[kk][self._stateVariable2]))<=eps)]
        
        MAT = Matrix([EQ1, EQ2])
        JAC = MAT.jacobian([self._stateVariable1,self._stateVariable2])
        
        eigList = []
        #for nn in range(len(realEQsol)): 
        #    JACsub=JAC.subs([(self._stateVariable1, realEQsol[nn][self._stateVariable1]), (self._stateVariable2, realEQsol[nn][self._stateVariable2])])
        #    evSet = JACsub.eigenvals()
        #    eigList.append(evSet)
        for nn in range(len(realEQsol)): 
            evSet = {}
            JACsub=JAC.subs([(self._stateVariable1, realEQsol[nn][self._stateVariable1]), (self._stateVariable2, realEQsol[nn][self._stateVariable2])])
            #evSet = JACsub.eigenvals()
            eigVects = JACsub.eigenvects()
            for kk in range(len(eigVects)):
                evSet[eigVects[kk][0]] = (eigVects[kk][1], eigVects[kk][2])
            eigList.append(evSet)
        return realEQsol, eigList #returns two lists of dictionaries
    
    ## calculates stationary states of 3d system
    def _get_fixedPoints3d(self):
#         #plotLimits = self._controller._getPlotLimits()
#         paramNames = []
#         paramValues = []
#         if self._controller is not None:
#             for name, value in self._controller._widgetsFreeParams.items():
#                 # throw away formatting for constant reactants
# #                 name = name.replace('(','')
# #                 name = name.replace(')','')
#                 paramNames.append(name)
#                 paramValues.append(value.value)
#         if self._paramNames is not None:
#             paramNames += map(str, self._paramNames)
#             paramValues += self._paramValues            
#         funcs = self._mumotModel._getFuncs()
#         argNamesSymb = list(map(Symbol, paramNames))
#         argDict = dict(zip(argNamesSymb, paramValues))
#         argDict[self._mumotModel._systemSize] = 1
        
        argDict = self._get_argDict()
        
        EQ1 = self._mumotModel._equations[self._stateVariable1].subs(argDict)
        EQ2 = self._mumotModel._equations[self._stateVariable2].subs(argDict)
        EQ3 = self._mumotModel._equations[self._stateVariable3].subs(argDict)
        eps=1e-8
        EQsol = solve((EQ1, EQ2, EQ3), (self._stateVariable1, self._stateVariable2, self._stateVariable3), dict=True)
        
        for nn in range(len(EQsol)):
            if len(EQsol[nn]) != 3:
                self._showErrorMessage('Some or all solutions are NOT unique.')
                return None, None
        
        realEQsol = [{self._stateVariable1: re(EQsol[kk][self._stateVariable1]), self._stateVariable2: re(EQsol[kk][self._stateVariable2]), self._stateVariable3: re(EQsol[kk][self._stateVariable3])} for kk in range(len(EQsol)) if (Abs(im(EQsol[kk][self._stateVariable1]))<=eps and Abs(im(EQsol[kk][self._stateVariable2]))<=eps and Abs(im(EQsol[kk][self._stateVariable3]))<=eps)]
        
        MAT = Matrix([EQ1, EQ2, EQ3])
        JAC = MAT.jacobian([self._stateVariable1,self._stateVariable2,self._stateVariable3])
        
        eigList = []
        #for nn in range(len(realEQsol)): 
        #    JACsub=JAC.subs([(self._stateVariable1, realEQsol[nn][self._stateVariable1]), (self._stateVariable2, realEQsol[nn][self._stateVariable2]), (self._stateVariable3, realEQsol[nn][self._stateVariable3])])
        #    evSet = JACsub.eigenvals()
        #    eigList.append(evSet)
        for nn in range(len(realEQsol)): 
            evSet = {}
            JACsub=JAC.subs([(self._stateVariable1, realEQsol[nn][self._stateVariable1]), (self._stateVariable2, realEQsol[nn][self._stateVariable2]), (self._stateVariable3, realEQsol[nn][self._stateVariable3])])
            #evSet = JACsub.eigenvals()
            eigVects = JACsub.eigenvects()
            for kk in range(len(eigVects)):
                evSet[eigVects[kk][0]] = (eigVects[kk][1], eigVects[kk][2])
            eigList.append(evSet)
            
        return realEQsol, eigList #returns two lists of dictionaries
    

        
                        
    def showLogs(self, tail = False):
        if tail:
            tailLength = 5
            print("Showing last " + str(min(tailLength, len(self._logs))) + " of " + str(len(self._logs)) + " log entries:")
            for log in self._logs[-tailLength:]:
                log.show()
        else:
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
        if (self._visualisationType == "evo"):
#             if not hasattr(self._controller, '_initialState'):
#                 print("ERROR! in multirun arguments. The specified controller does not have the attribute _initialState which is required for visualisationType 'evo'.")
#                 return
            systemSize = sum(self._controller._view._initialState.values())
            maxTime = self._controller._widgetsExtraParams['maxTime'].value
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
    

## multi-view view (tied closely to MuMoTmultiController)
class MuMoTmultiView(MuMoTview):
    ## view list
    _views = None
    ## axes are used for subplots ('shareAxes = True')
    _axes = None
    ## number of subplots
    _subPlotNum = None
    ## subplot rows
    _numRows = None
    ## subplot columns
    _numColumns = None
    ## use common axes for all plots (False = use subplots)
    _shareAxes = None

    def __init__(self, controller, views, subPlotNum, **kwargs):
        super().__init__(None, controller, **kwargs)
        self._generatingCommand = "mmt.MuMoTmultiController"
        self._views = views
        self._subPlotNum = subPlotNum
        for view in self._views:
            view._figure = self._figure
            view._figureNum = self._figureNum
            view._setLog(self._logs)
            view._controller = controller
        self._shareAxes = kwargs.get('shareAxes', False)        
        if not(self._shareAxes):
            self._numColumns = MULTIPLOT_COLUMNS
            self._numRows = math.ceil(self._subPlotNum / self._numColumns)
        
    def _plot(self, _=None):
        plt.figure(self._figureNum)
        plt.clf()
        self._resetErrorMessage()
        if self._shareAxes:
            # hold should already be on
            for func, subPlotNum, axes3d in self._controller._replotFunctions:
                func()
        else:
#            subplotNum = 1
            for func, subPlotNum, axes3d in self._controller._replotFunctions:
                if axes3d:
#                    self._figure.add_subplot(self._numRows, self._numColumns, subPlotNum, projection = '3d')
                    plt.subplot(self._numRows, self._numColumns, subPlotNum, projection = '3d')
                else:
                    plt.subplot(self._numRows, self._numColumns, subPlotNum)
                func()
#                subplotNum += 1


    def _setLog(self, log):
        for view in self._views:
            view._setLog(log)


    def _print_standalone_view_cmd(self):
        for view in self._views:
            pass
        model = view._mumotModel
        with io.capture_output() as log:
            logStr = "bookmark = " + self._generatingCommand + "(["
            for view in self._views:
                logStr += view._print_standalone_view_cmd(False) + ", "
            logStr = logStr[:-2]  # throw away last ", "
            logStr += "], "
            logStr += self._get_bookmarks_params(model)
            if len(self._generatingKwargs) > 0:
                logStr += ", "
                for key in self._generatingKwargs:
                    logStr += key + " = " + str(self._generatingKwargs[key]) + ", "
                logStr = logStr[:-2]  # throw away last ", "
            logStr += ", bookmark = False"
            logStr += ")"
            logStr = logStr.replace('\\', '\\\\')

            print(logStr)
        self._logs.append(log)



## multi-view controller
class MuMoTmultiController(MuMoTcontroller):
    ## replot function list to invoke on views
    _replotFunctions = None

    def __init__(self, controllers, params = None, **kwargs):
        global figureCounter

        self._silent = kwargs.get('silent', False)
        self._replotFunctions = []
        paramNames = []
        paramValues = []
        fixedParamNames = None
        paramValueDict = {}
        paramLabelDict = {}
        plotLimits = False
        systemSize = False
        #widgetsExtraParamsTmp = {} # cannot be the final dict already, because it will be erased when constructor is called
        views = []
        subPlotNum = 1
        if params is not None:
            (fixedParamNames, fixedParamValues) = _process_params(params)
        for controller in controllers:
            # pass through the fixed params to each constituent view
            view = controller._view
            if params is not None:
                (view._paramNames, view._paramValues) = _process_params(params)
            for name, value in controller._widgetsFreeParams.items():
                if params is None or name not in fixedParamNames:
                    paramValueDict[name] = (value.value, value.min, value.max, value.step)
            if controller._plotLimitsWidget is not None:
                plotLimits = True
            if controller._systemSizeWidget is not None:
                systemSize = True
            paramLabelDict.update(controller._paramLabelDict)
#             for name, value in controller._widgetsExtraParams.items():
#                 widgetsExtraParamsTmp[name] = value
            if controller._replotFunction == None: ## presume this controller is a multi controller (@todo check?)
                for view in controller._view._views:
                    views.append(view)         
                               
#                if view._controller._replotFunction == None: ## presume this controller is a multi controller (@todo check?)
                for func, _, axes3d in controller._replotFunctions:
                    self._replotFunctions.append((func, subPlotNum, axes3d))                    
#                else:
#                    self._replotFunctions.append((view._controller._replotFunction, subPlotNum, view._axes3d))                    
            else:
                views.append(controller._view)
#                 if controller._replotFunction == None: ## presume this controller is a multi controller (@todo check?)
#                     for func, foo in controller._replotFunctions:
#                         self._replotFunctions.append((func, subPlotNum))                    
#                 else:
                self._replotFunctions.append((controller._replotFunction, subPlotNum, controller._view._axes3d))                    
            subPlotNum += 1

        for name, value in paramValueDict.items():
            paramNames.append(name)
            paramValues.append(value)
            
#         for view in self._views:
#             if view._controller._replotFunction == None: ## presume this controller is a multi controller (@todo check?)
#                 for func in view._controller._replotFunctions:
#                     self._replotFunctions.append(func)                    
#             else:
#                 self._replotFunctions.append(view._controller._replotFunction)
#             view._controller = self
        
        super().__init__(paramValues, paramNames, paramLabelDict, False, plotLimits, systemSize, params = params, **kwargs)
        
        # handle Extra and PlotOnly params
        addProgressBar = False
        for controller in controllers:
            # retrieve the _widgetsExtraParams from each controller
            for name, widget in controller._widgetsExtraParams.items():
                widget.unobserve_all()
                if name == "netType": # netType widget is special
                    ## this is necessary due to ipywidget bug #1868 (https://github.com/jupyter-widgets/ipywidgets/issues/1868)
                    widget = widgets.Dropdown( 
                        options=[('Full graph', NetworkType.FULLY_CONNECTED), 
                                 ('Erdos-Renyi', NetworkType.ERSOS_RENYI),
                                 ('Barabasi-Albert', NetworkType.BARABASI_ALBERT),
                                 ('Moving particles', NetworkType.DYNAMIC)
                                 ],
                        description='Network topology:',
                        value = widget.value, 
                        disabled=False
                    )
                    widget.observe(controller._update_net_params, names='value')
                self._widgetsExtraParams[name] = widget
            # retrieve the _widgetsPlotOnly from each controller
            for name, widget in controller._widgetsPlotOnly.items():
                widget.unobserve_all()
                # in multiController, plotProportions=True is forced 
                if name == "plotProportions":
                    widget.value = True
                    widget.disabled = True
                ## this is necessary due to ipywidget bug #1868 (https://github.com/jupyter-widgets/ipywidgets/issues/1868)
                if name == "visualisationType": 
                    ## Toggle buttons for plotting style 
                    widget = widgets.ToggleButtons(
                        options=[('Temporal evolution', 'evo'), ('Final distribution', 'final')],
                        value = widget.value,
                        description='Plot:',
                        disabled=False,
                        button_style='', # 'success', 'info', 'warning', 'danger' or ''
                        tooltips=['Population change over time', 'Population distribution in each state at final timestep'],
                    )
                self._widgetsPlotOnly[name] = widget
            # retrieve the _extraWidgetsOrder from each controller
            self._extraWidgetsOrder.extend(x for x in controller._extraWidgetsOrder if x not in self._extraWidgetsOrder)
            if controller._progressBar:
                addProgressBar = True
        if self._widgetsExtraParams or self._widgetsPlotOnly:
            # set widgets to possible values set as multi-controller arguments
            for key, value in kwargs.items():
                if self._widgetsExtraParams.get(key):
                    if (self._widgetsExtraParams[key].min > value):
                        self._widgetsExtraParams[key].min = value
                    if (self._widgetsExtraParams[key].max < value):
                        self._widgetsExtraParams[key].max = value
                    self._widgetsExtraParams[key].value = value
                if self._widgetsPlotOnly.get(key):
                    if (self._widgetsPlotOnly[key].min > value):
                        self._widgetsPlotOnly[key].min = value
                    if (self._widgetsPlotOnly[key].max < value):
                        self._widgetsPlotOnly[key].max = value
                    self._widgetsPlotOnly[key].value = value
            
            # create the "Advanced options" tab
            if not self._silent:
                self._displayAdvancedOptionsTab()
        
        # if necessary adding the progress bar
        if addProgressBar:
            # Loading bar (useful to give user progress status for long executions)
            self._progressBar = widgets.FloatProgress(
                value=0,
                min=0,
                max=self._widgetsExtraParams['maxTime'].value,
                #step=1,
                description='Loading:',
                bar_style='success', # 'success', 'info', 'warning', 'danger' or ''
                orientation='horizontal'
            )
            if not self._silent:
                display(self._progressBar)

        self._view = MuMoTmultiView(self, views, subPlotNum - 1, **kwargs)
        if fixedParamNames is not None:
            self._view._paramNames = fixedParamNames
            self._view._paramValues = fixedParamValues
                
        for controller in controllers:
            controller._setErrorWidget(self._errorMessage)
        
        ## @todo handle correctly the re-draw only widgets and function
        self._setReplotFunction( self._view._plot, self._view._plot)
        
        #silent = kwargs.get('silent', False)
        if not self._silent:
            self._view._plot()
            
    def helloWorld(self, _=None):
        print("HELLO WORLD")

## time evolution view on model including state variables and noise (specialised by MuMoTtimeEvoStateVarView and ...)
class MuMoTtimeEvolutionView(MuMoTview):
    ## list of all state variables
    _stateVarList = None
    ## list of all state variables displayed in figure
    _stateVarListDisplay = None
    ## 1st state variable
    _stateVariable1 = None
    ## 2nd state variable 
    _stateVariable2 = None
    ## 3rd state variable 
    _stateVariable3 = None
#     ## 4th state variable 
#     _stateVariable4 = None
    ## initial conditions of state variables for numerical solution of ODE system
    _initCondsSV = None
    ## end time of numerical simulation of ODE system of the state variables
    _tend = None
    ## time step of numerical simulation of ODE system of the state variables
    _tstep = None
    ## defines fontsize on the axes
    _chooseFontSize = None
    ## string that defines the x-label
    _xlab = None
    ## legend location: combinations like 'upper left', lower right, or 'center center' are allowed (9 options in total)
    _legend_loc = None

    #def __init__(self, model, controller, stateVariable1, stateVariable2, stateVariable3 = None, stateVariable4 = None, figure = None, params = None, **kwargs):
    def __init__(self, model, controller, showStateVars = None, figure = None, params = None, **kwargs):
        #if model._systemSize == None and model._constantSystemSize == True:
        #    print("Cannot construct time evolution -based plot until system size is set, using substitute()")
        #    return
        silent = kwargs.get('silent', False)
        
        super().__init__(model, controller, figure, params, **kwargs)
        
        if 'fontsize' in kwargs:
            self._chooseFontSize = kwargs['fontsize']
        else:
            self._chooseFontSize=None
        self._xlab = kwargs.get('xlab', r'time t')
        #self._ylab = kwargs.get('ylab', r'evolution of states')
        
        self._legend_loc = kwargs.get('legend_loc', 'upper left')
        self._legend_fontsize = kwargs.get('legend_fontsize', 18)
        
        self._stateVarList = []
        for reactant in self._mumotModel._reactants:
            if reactant not in self._mumotModel._constantReactants:
                self._stateVarList.append(reactant)
              
        if showStateVars:
            if type(showStateVars) == list:
                self._stateVarListDisplay = showStateVars    
                for kk in range(len(self._stateVarListDisplay)):
                    self._stateVarListDisplay[kk] = process_sympy(self._stateVarListDisplay[kk])
            else:
                self._showErrorMessage('Check input: should be of type list!')
        else:
            self._stateVarListDisplay = copy.deepcopy(self._stateVarList)
        
        self._stateVariable1 = self._stateVarList[0]
        self._stateVariable2 = self._stateVarList[1]
        if len(self._stateVarList) == 3:
            self._stateVariable3 = self._stateVarList[2]
        
#         
#         self._stateVariable1 = process_sympy(stateVariable1)
#         self._stateVariable2 = process_sympy(stateVariable2)
#         if stateVariable3 != None:
#             self._stateVariable3 = process_sympy(stateVariable3)
#         if stateVariable4 != None:
#             self._stateVariable4 = process_sympy(stateVariable4)
#             

        self._tend = kwargs.get('tend', 100)
        self._tstep = kwargs.get('tstep', 0.01)
        
        self._initCondsSV = kwargs.get('initCondsSV', False)
        
        # check input of initCondsSV
        if self._initCondsSV != False:
            for reactIC in self._initCondsSV:
                if process_sympy(reactIC) not in self._stateVarList:
                    print('Warning: symbol for reactant  '+str(reactIC)+'  is not a time-dependent model reactant. Check that symbols for the initial conditions are the same as the ones introduced when creating the model, and that it is not a constant reactant.')
                #elif str(reactIC) in self._mumotModel._constantReactants:
                #    print('Warning: Reactant  '+str(reactIC)+'  is a constant reactant. Give initial conditions only for time-dependent reactants.')

        if not(silent):
            self._plot_NumSolODE()

    
    ## calculates right-hand side of ODE system
    def _get_eqsODE(self, y_old, time):
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
#         
#         SVsub[self._stateVariable1] = y_old[0]
#         SVsub[self._stateVariable2] = y_old[1]
#         if self._stateVariable3:
#             SVsub[self._stateVariable3] = y_old[2]
#         if self._stateVariable4:
#             SVsub[self._stateVariable4] = y_old[3]
#             


#         #plotLimits = self._controller._getPlotLimits()
#         paramNames = []
#         paramValues = []
#         if self._controller is not None:
#             for name, value in self._controller._widgetsFreeParams.items():
#                 # throw away formatting for constant reactants
# #                 name = name.replace('(','')
# #                 name = name.replace(')','')
#                 paramNames.append(name)
#                 paramValues.append(value.value)
#         if self._paramNames is not None:
#             paramNames += map(str, self._paramNames)
#             paramValues += self._paramValues            
#         #funcs = self._mumotModel._getFuncs()
#         argNamesSymb = list(map(Symbol, paramNames))
#         argDict = dict(zip(argNamesSymb, paramValues))
# #         argDict[self._mumotModel._systemSize] = 1
            
#             
#         EQ1 = self._mumotModel._equations[self._stateVariable1].subs(argDict)
#         EQ1 = EQ1.subs(SVsub)
#         EQ2 = self._mumotModel._equations[self._stateVariable2].subs(argDict)
#         EQ2 = EQ2.subs(SVsub)
#         ode_sys = [EQ1, EQ2]
#         if self._stateVariable3:
#             EQ3 = self._mumotModel._equations[self._stateVariable3].subs(argDict)
#             EQ3 = EQ3.subs(SVsub)
#             ode_sys.append(EQ3)
#         if self._stateVariable4:
#             EQ4 = self._mumotModel._equations[self._stateVariable4].subs(argDict)
#             EQ4 = EQ4.subs(SVsub)
#             ode_sys.append(EQ4)
#                

    
    def _plot_NumSolODE(self):
        if not(self._silent): ## @todo is this necessary?
            plt.figure(self._figureNum)
            plt.clf()
            self._resetErrorMessage()
        self._showErrorMessage(str(self))
        
#         
#     def _numericSol2ndOrdMoment(self, EOM_2ndOrdMomDict, steadyStateDict, argDict):
#         for sol in EOM_2ndOrdMomDict:
#             EOM_2ndOrdMomDict[sol] = EOM_2ndOrdMomDict[sol].subs(steadyStateDict)
#             EOM_2ndOrdMomDict[sol] = EOM_2ndOrdMomDict[sol].subs(argDict)
# #         
# #         eta_SV1 = Symbol('eta_'+str(self._stateVarList[0]))
# #         eta_SV2 = Symbol('eta_'+str(self._stateVarList[1]))
# #         M_1, M_2 = symbols('M_1 M_2')
# #         if len(self._stateVarList) == 3:
# #             eta_SV3 = Symbol('eta_'+str(self._stateVarList[2]))
# #             
#              
#         eta_SV1 = Symbol('eta_'+str(self._stateVariable1))
#         eta_SV2 = Symbol('eta_'+str(self._stateVariable2))
#         M_1, M_2 = symbols('M_1 M_2')
#         if self._stateVariable3:
#             eta_SV3 = Symbol('eta_'+str(self._stateVariable3))
#              
# 
#         SOL_2ndOrdMomDict = {} 
#         EQsys2ndOrdMom = []
#         if self._stateVariable3:
#             EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV1*eta_SV1)])
#             EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV2*eta_SV2)])
#             EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV3*eta_SV3)])
#             EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV1*eta_SV2)])
#             EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV1*eta_SV3)])
#             EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV2*eta_SV3)])
#             SOL_2ndOrderMom = list(linsolve(EQsys2ndOrdMom, [M_2(eta_SV1*eta_SV1), 
#                                                              M_2(eta_SV2*eta_SV2), 
#                                                              M_2(eta_SV3*eta_SV3), 
#                                                              M_2(eta_SV1*eta_SV2), 
#                                                              M_2(eta_SV1*eta_SV3), 
#                                                              M_2(eta_SV2*eta_SV3)]))[0] #only one set of solutions (if any) in linear system of equations; hence index [0]
#             
#             SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV1)] = SOL_2ndOrderMom[0]
#             SOL_2ndOrdMomDict[M_2(eta_SV2*eta_SV2)] = SOL_2ndOrderMom[1]
#             SOL_2ndOrdMomDict[M_2(eta_SV3*eta_SV3)] = SOL_2ndOrderMom[2]
#             SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV2)] = SOL_2ndOrderMom[3]
#             SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV3)] = SOL_2ndOrderMom[4]
#             SOL_2ndOrdMomDict[M_2(eta_SV2*eta_SV3)] = SOL_2ndOrderMom[5]
#         
#         else:
#             EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV1*eta_SV1)])
#             EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV2*eta_SV2)])
#             EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV1*eta_SV2)])
#             SOL_2ndOrderMom = list(linsolve(EQsys2ndOrdMom, [M_2(eta_SV1*eta_SV1), 
#                                                              M_2(eta_SV2*eta_SV2), 
#                                                              M_2(eta_SV1*eta_SV2)]))[0] #only one set of solutions (if any) in linear system of equations
#             
#             SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV1)] = SOL_2ndOrderMom[0]
#             SOL_2ndOrdMomDict[M_2(eta_SV2*eta_SV2)] = SOL_2ndOrderMom[1]
#             SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV2)] = SOL_2ndOrderMom[2]
#             
#         return SOL_2ndOrdMomDict
#     

# 
#     def _build_bookmark(self, includeParams = True):
#         if not self._silent:
#             logStr = "bookmark = "
#         else:
#             logStr = ""
#         
#         logStr += "<modelName>." + self._generatingCommand + "(["
#         for nn in range(len(self._stateVarListDisplay)):
#             if nn == len(self._stateVarListDisplay)-1:
#                 logStr += "'" + str(self._stateVarListDisplay[nn]) + "'], "
#             else:
#                 logStr += "'" + str(self._stateVarListDisplay[nn]) + "', "
#         
# #         
# #         logStr += "<modelName>." + self._generatingCommand + "('" + str(self._stateVariable1) + "', '" + str(self._stateVariable2) +"', "
# #         if self._stateVariable3 != None:
# #             logStr += "'" + str(self._stateVariable3) + "', "
# #         if self._stateVariable4 != None:
# #             logStr += "'" + str(self._stateVariable4) + "', "
# #         
#      
#         if includeParams:
#             logStr += self._get_bookmarks_params() + ", "
#         if len(self._generatingKwargs) > 0:
#             for key in self._generatingKwargs:
#                 if type(self._generatingKwargs[key]) == str:
#                     logStr += key + " = " + "\'"+ str(self._generatingKwargs[key]) + "\'" + ", "
#                 else:
#                     logStr += key + " = " + str(self._generatingKwargs[key]) + ", "
#         logStr += "bookmark = False"
#         logStr += ")"
#         logStr = logStr.replace('\\', '\\\\')
#         
#         return logStr
#     


# class MuMoTtimeEvolutionView(MuMoTview):
#     ## 1st state variable
#     _stateVariable1 = None
#     ## 2nd state variable 
#     _stateVariable2 = None
#     ## 3rd state variable 
#     _stateVariable3 = None
#     ## 4th state variable 
#     _stateVariable4 = None
#     ## end time of numerical simulation
#     _tend = None
#     ## time step of numerical simulation
#     _tstep = None
#     
#   
#     def __init__(self, model, controller, stateVariable1, stateVariable2, stateVariable3 = None, stateVariable4 = None, tend = 100, tstep = 0.01, figure = None, params = None, **kwargs):
#         #if model._systemSize == None and model._constantSystemSize == True:
#         #    print("Cannot construct time evolution -based plot until system size is set, using substitute()")
#         #    return
#         silent = kwargs.get('silent', False)
#         super().__init__(model, controller, figure, params, **kwargs)
#         
#         if 'fontsize' in kwargs:
#             self._chooseFontSize = kwargs['fontsize']
#         else:
#             self._chooseFontSize=None
#         self._xlab = kwargs.get('xlab', r'time t')
#         self._ylab = kwargs.get('ylab', r'evolution of states')
#         
#         self._legend_loc = kwargs.get('legend_loc', 'upper left')
#         
#         self._stateVariable1 = process_sympy(stateVariable1)
#         self._stateVariable2 = process_sympy(stateVariable2)
#         if stateVariable3 != None:
#             self._stateVariable3 = process_sympy(stateVariable3)
#         if stateVariable4 != None:
#             self._stateVariable4 = process_sympy(stateVariable4)
#             
#         self._tend = tend
#         self._tstep = tstep
#         if not(silent):
#             self._plot_NumSolODE()           
# 
#  
#    ## calculates stationary states of 2d system
#     def _get_fixedPoints2d(self):
#         #plotLimits = self._controller._getPlotLimits()
#         paramNames = []
#         paramValues = []
#         if self._controller is not None:
#             for name, value in self._controller._widgetsFreeParams.items():
#                 # throw away formatting for constant reactants
# #                 name = name.replace('(','')
# #                 name = name.replace(')','')
#                 paramNames.append(name)
#                 paramValues.append(value.value)
#         if self._paramNames is not None:
#             paramNames += map(str, self._paramNames)
#             paramValues += self._paramValues            
#         funcs = self._mumotModel._getFuncs()
#         argNamesSymb = list(map(Symbol, paramNames))
#         argDict = dict(zip(argNamesSymb, paramValues))
#         argDict[self._mumotModel._systemSize] = 1
#         
#         EQ1 = self._mumotModel._equations[self._stateVariable1].subs(argDict)
#         EQ2 = self._mumotModel._equations[self._stateVariable2].subs(argDict)
#         eps=1e-8
#         EQsol = solve((EQ1, EQ2), (self._stateVariable1, self._stateVariable2), dict=True)
#         realEQsol = [{self._stateVariable1: re(EQsol[kk][self._stateVariable1]), self._stateVariable2: re(EQsol[kk][self._stateVariable2])} for kk in range(len(EQsol)) if (Abs(im(EQsol[kk][self._stateVariable1]))<=eps and Abs(im(EQsol[kk][self._stateVariable2]))<=eps)]
#         
#         MAT = Matrix([EQ1, EQ2])
#         JAC = MAT.jacobian([self._stateVariable1,self._stateVariable2])
#         
#         eigList = []
#         for nn in range(len(realEQsol)): 
#             evSet = {}
#             JACsub=JAC.subs([(self._stateVariable1, realEQsol[nn][self._stateVariable1]), (self._stateVariable2, realEQsol[nn][self._stateVariable2])])
#             #evSet = JACsub.eigenvals()
#             eigVects = JACsub.eigenvects()
#             for kk in range(len(eigVects)):
#                 evSet[eigVects[kk][0]] = (eigVects[kk][1], eigVects[kk][2])
#             eigList.append(evSet)
#         return realEQsol, eigList #returns two lists of dictionaries
#     
#     ## calculates stationary states of 3d system
#     def _get_fixedPoints3d(self):
#         #plotLimits = self._controller._getPlotLimits()
#         paramNames = []
#         paramValues = []
#         if self._controller != None:
#             for name, value in self._controller._widgetsFreeParams.items():
#                 # throw away formatting for constant reactants
# #                 name = name.replace('(','')
# #                 name = name.replace(')','')
#                 paramNames.append(name)
#                 paramValues.append(value.value)
#         if self._paramNames is not None:
#             paramNames += map(str, self._paramNames)
#             paramValues += self._paramValues            
#         funcs = self._mumotModel._getFuncs()
#         argNamesSymb = list(map(Symbol, paramNames))
#         argDict = dict(zip(argNamesSymb, paramValues))
#         argDict[self._mumotModel._systemSize] = 1
#         
#         EQ1 = self._mumotModel._equations[self._stateVariable1].subs(argDict)
#         EQ2 = self._mumotModel._equations[self._stateVariable2].subs(argDict)
#         EQ3 = self._mumotModel._equations[self._stateVariable3].subs(argDict)
#         eps=1e-8
#         EQsol = solve((EQ1, EQ2, EQ3), (self._stateVariable1, self._stateVariable2, self._stateVariable3), dict=True)
#         realEQsol = [{self._stateVariable1: re(EQsol[kk][self._stateVariable1]), self._stateVariable2: re(EQsol[kk][self._stateVariable2]), self._stateVariable3: re(EQsol[kk][self._stateVariable3])} for kk in range(len(EQsol)) if (Abs(im(EQsol[kk][self._stateVariable1]))<=eps and Abs(im(EQsol[kk][self._stateVariable2]))<=eps and Abs(im(EQsol[kk][self._stateVariable3]))<=eps)]
#         
#         MAT = Matrix([EQ1, EQ2, EQ3])
#         JAC = MAT.jacobian([self._stateVariable1,self._stateVariable2,self._stateVariable3])
#         
#         eigList = []
#         for nn in range(len(realEQsol)): 
#             evSet = {}
#             JACsub=JAC.subs([(self._stateVariable1, realEQsol[nn][self._stateVariable1]), (self._stateVariable2, realEQsol[nn][self._stateVariable2]), (self._stateVariable3, realEQsol[nn][self._stateVariable3])])
#             #evSet = JACsub.eigenvals()
#             eigVects = JACsub.eigenvects()
#             for kk in range(len(eigVects)):
#                 evSet[eigVects[kk][0]] = (eigVects[kk][1], eigVects[kk][2])
#             eigList.append(evSet)
#         return realEQsol, eigList #returns two lists of dictionaries
#     
#
#     def _print_standalone_view_cmd(self):
#         with io.capture_output() as log:
#             print('Bookmark feature for standalone view not implemented for time-dependent soultion of ODE.')
# #             logStr = self._generatingCommand + "(<modelName>, None, '" + str(self._stateVariable1) + "', '" + str(self._stateVariable2) +"', "
# #             if self._stateVariable3 != None:
# #                 logStr += "'" + str(self._stateVariable3) + "', "
# #             logStr += "params = ["
# #             for name, value in self._controller._widgetsFreeParams.items():
# # #                name = name.replace('\\', '\\\\')
# #                 name = name.replace('(', '')
# #                 name = name.replace(')', '')
# #                 logStr += "('" + name + "', " + str(value.value) + "), "
# #             if len(self._controller._widgetsFreeParams.items()) > 0:
# #                 logStr = logStr[:-2] # throw away last ", "
# #             logStr += "]"
# #             if self._generatingKwargs != None:
# #                 logStr += ", "
# #                 for key in self._generatingKwargs:
# #                     logStr += key + " = " + str(self._generatingKwargs[key]) + ", "
# #                 if len(self._generatingKwargs) > 0:
# #                     logStr = logStr[:-2]  # throw away last ", "
# #             logStr += ")"
# #             print(logStr)    
#         self._logs.append(log)
        


## numerical solution of state variables plot view on model
class MuMoTtimeEvoStateVarView(MuMoTtimeEvolutionView):
    ## y-label with default specific to this MuMoTtimeEvoStateVarView class (can be set via keyword)
    _ylab = None
    
    def __init__(self, *args, **kwargs):
        
        self._ylab = kwargs.get('ylab', r'evolution of states')
        super().__init__(*args, **kwargs)
        self._generatingCommand = "numSimStateVar"
        

    def _plot_NumSolODE(self, _=None):
        super()._plot_NumSolODE()
        
        # check input
        for nn in range(len(self._stateVarListDisplay)):
            if self._stateVarListDisplay[nn] not in self._stateVarList:
                self._showErrorMessage('Warning:  '+str(self._stateVarListDisplay[nn])+'  is no reactant in the current model.')
                return None
      
#         
#         if self._stateVariable1 not in self._mumotModel._reactants:
#             self._showErrorMessage('Warning:  '+str(self._stateVariable1)+'  is no reactant in the current model.')
#         if self._stateVariable2 not in self._mumotModel._reactants:
#             self._showErrorMessage('Warning:  '+str(self._stateVariable2)+'  is no reactant in the current model.')
#         if self._stateVariable3:
#             if self._stateVariable3 not in self._mumotModel._reactants:
#                 self._showErrorMessage('Warning:  '+str(self._stateVariable3)+'  is no reactant in the current model.')
#         if self._stateVariable4:
#             if self._stateVariable4 not in self._mumotModel._reactants:
#                 self._showErrorMessage('Warning:  '+str(self._stateVariable4)+'  is no reactant in the current model.')
#         
        
        NrDP = int(self._tend/self._tstep) + 1
        time = np.linspace(0, self._tend, NrDP)
        
        initDict = self._getInitCondsFromSlider()
        #if len(initDict) < 2 or len(initDict) > 4:
        #    self._showErrorMessage("Not implemented: This feature is available only for systems with 2, 3 or 4 time-dependent reactants!")

        y0 = []
        for nn in range(len(self._stateVarList)):
            SVi0 = initDict[Symbol(latex(Symbol('Phi^0_'+str(self._stateVarList[nn]))))]
            y0.append(SVi0)
          
#             
#         SV1_0 = initDict[Symbol(latex(Symbol('Phi^0_'+str(self._stateVariable1))))]
#         SV2_0 = initDict[Symbol(latex(Symbol('Phi^0_'+str(self._stateVariable2))))]
#         y0 = [SV1_0, SV2_0]
#         if len(initDict) > 2:
#             SV3_0 = initDict[Symbol(latex(Symbol('Phi^0_'+str(self._stateVariable3))))]
#             y0.append(SV3_0)
#         if len(initDict) > 3:
#             SV4_0 = initDict[Symbol(latex(Symbol('Phi^0_'+str(self._stateVariable4))))]
#             y0.append(SV4_0)
#         

        sol_ODE = odeint(self._get_eqsODE, y0, time)  
        
        sol_ODE_dict = {}
        for nn in range(len(self._stateVarList)):
            sol_ODE_dict[str(self._stateVarList[nn])] = sol_ODE[:, nn]
          
        #x_data = [time for kk in range(len(self._get_eqsODE(y0, time)))]
        x_data = [time for kk in range(len(self._stateVarListDisplay))]
        #y_data = [sol_ODE[:, kk] for kk in range(len(self._get_eqsODE(y0, time)))]
        y_data = [sol_ODE_dict[str(self._stateVarListDisplay[kk])] for kk in range(len(self._stateVarListDisplay))]
        
        c_labels = [r'$'+latex(Symbol('Phi_'+str(self._stateVarListDisplay[nn]))) +'$' for nn in range(len(self._stateVarListDisplay))] 
        
#         
#         c_labels = [r'$'+latex(Symbol('Phi_'+str(self._stateVariable1)))+'$', r'$'+latex(Symbol('Phi_'+str(self._stateVariable2)))+'$'] 
#         if self._stateVariable3:
#             c_labels.append(r'$'+latex(Symbol('Phi_'+str(self._stateVariable3)))+'$')
#         if self._stateVariable4:
#             c_labels.append(r'$'+latex(Symbol('Phi_'+str(self._stateVariable4)))+'$')         
#         

        _fig_formatting_2D(xdata=x_data, ydata = y_data , xlab = self._xlab, ylab = self._ylab, 
                           fontsize=self._chooseFontSize, curvelab=c_labels, legend_loc=self._legend_loc, grid = True,
                           legend_fontsize=self._legend_fontsize)
#        plt.set_aspect('equal') ## @todo


    def _build_bookmark(self, includeParams = True):
        if not self._silent:
            logStr = "bookmark = "
        else:
            logStr = ""
        
        logStr += "<modelName>." + self._generatingCommand + "(["
        for nn in range(len(self._stateVarListDisplay)):
            if nn == len(self._stateVarListDisplay)-1:
                logStr += "'" + str(self._stateVarListDisplay[nn]) + "'], "
            else:
                logStr += "'" + str(self._stateVarListDisplay[nn]) + "', "
     
        if includeParams:
            logStr += self._get_bookmarks_params() + ", "
        if len(self._generatingKwargs) > 0:
            for key in self._generatingKwargs:
                if type(self._generatingKwargs[key]) == str:
                    logStr += key + " = " + "\'"+ str(self._generatingKwargs[key]) + "\'" + ", "
                else:
                    logStr += key + " = " + str(self._generatingKwargs[key]) + ", "
        logStr += "bookmark = False"
        logStr += ")"
        logStr = logStr.replace('\\', '\\\\')
        
        return logStr



# class MuMoTtimeEvoStateVarView(MuMoTtimeEvolutionView):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._generatingCommand = "mmt.MuMoTtimeEvoStateVarView"
#     def _plot_NumSolODE(self):
#         super()._plot_NumSolODE()
#         NrDP = int(self._tend/self._tstep) + 1
#         time = np.linspace(0, self._tend, NrDP)
#         initDict = self._getInitCondsFromSlider()
#         assert (2 <= len(initDict) <= 4),"Not implemented: This feature is available only for systems with 2, 3 or 4 time-dependent reactants!"
# 
#         
#         SV1_0 = initDict[Symbol(latex(Symbol('Phi^0_'+str(self._stateVariable1))))]
#         SV2_0 = initDict[Symbol(latex(Symbol('Phi^0_'+str(self._stateVariable2))))]
#         y0 = [SV1_0, SV2_0]
#         if len(initDict) > 2:
#             SV3_0 = initDict[Symbol(latex(Symbol('Phi^0_'+str(self._stateVariable3))))]
#             y0.append(SV3_0)
#         if len(initDict) > 3:
#             SV4_0 = initDict[Symbol(latex(Symbol('Phi^0_'+str(self._stateVariable4))))]
#             y0.append(SV4_0)
#         
#         sol_ODE = odeint(self._get_eqsODE, y0, time)  
#           
#         x_data = [time for kk in range(len(self._get_eqsODE(y0, time)))]
#         y_data = [sol_ODE[:, kk] for kk in range(len(self._get_eqsODE(y0, time)))]
#         
#         c_labels = [r'$'+latex(Symbol('Phi_'+str(self._stateVariable1)))+'$', r'$'+latex(Symbol('Phi_'+str(self._stateVariable2)))+'$'] 
#         if self._stateVariable3:
#             c_labels.append(r'$'+latex(Symbol('Phi_'+str(self._stateVariable3)))+'$')
#         if self._stateVariable4:
#             c_labels.append(r'$'+latex(Symbol('Phi_'+str(self._stateVariable4)))+'$')         
#         
#         _fig_formatting_2D(xdata=x_data, ydata = y_data , xlab = self._xlab, ylab = self._ylab, 
#                            fontsize=self._chooseFontSize, curvelab=c_labels, legend_loc=self._legend_loc, grid = True)
# #        plt.set_aspect('equal') ## @todo

class MuMoTtimeEvoNoiseCorrView(MuMoTtimeEvolutionView):
    ## equations of motion for first order moments of noise variables
    _EOM_1stOrdMomDict = None
    ## equations of motion for second order moments of noise variables
    _EOM_2ndOrdMomDict = None
    ## upper bound of simulation time for noise-noise correlation functions (can be set via keyword)
    _tendNoise = None
    ## time step of simulation for noise-noise correlation functions (can be set via keyword)
    _tstepNoise = None
    ## y-label with default specific to this MuMoTtimeEvoNoiseCorrView class (can be set via keyword)
    _ylab = None
    

#    def __init__(self, model, controller, EOM_1stOrdMom, EOM_2ndOrdMom, stateVariable1, stateVariable2, stateVariable3=None, stateVariable4=None, figure = None, params = None, **kwargs):
    def __init__(self, model, controller, EOM_1stOrdMom, EOM_2ndOrdMom, figure = None, params = None, **kwargs):
        self._EOM_1stOrdMomDict = EOM_1stOrdMom
        self._EOM_2ndOrdMomDict = EOM_2ndOrdMom
        self._tendNoise = kwargs.get('tendNoise', 50)
        self._tstepNoise= kwargs.get('tstepNoise', 0.01)
        self._ylab = kwargs.get('ylab', 'noise-noise correlation')
        silent = kwargs.get('silent', False)
        super().__init__(model, controller, None, figure, params, **kwargs)
        #super().__init__(*args, **kwargs)
        
        if len(self._stateVarList) < 2 or len(self._stateVarList) > 3:
            self._showErrorMessage("Not implemented: This feature is available only for systems with 2 or 3 time-dependent reactants!")
            return None
        
        self._generatingCommand = "numSimNoiseCorrelations"
    
        
        
    def _plot_NumSolODE(self):
        super()._plot_NumSolODE()
        
        # check input
        for nn in range(len(self._stateVarListDisplay)):
            if self._stateVarListDisplay[nn] not in self._stateVarList:
                self._showErrorMessage('Warning:  '+str(self._stateVarListDisplay[nn])+'  is no reactant in the current model.')
                return None

#         
#         # check input
#         if self._stateVariable1 not in self._mumotModel._reactants:
#             self._showErrorMessage('Warning:  '+str(self._stateVariable1)+'  is no reactant in the current model.')
#         if self._stateVariable2 not in self._mumotModel._reactants:
#             self._showErrorMessage('Warning:  '+str(self._stateVariable2)+'  is no reactant in the current model.')
#         if self._stateVariable3:
#             if self._stateVariable3 not in self._mumotModel._reactants:
#                 self._showErrorMessage('Warning:  '+str(self._stateVariable3)+'  is no reactant in the current model.')
#         if self._stateVariable4:
#             if self._stateVariable4 not in self._mumotModel._reactants:
#                 self._showErrorMessage('Warning:  '+str(self._stateVariable4)+'  is no reactant in the current model.')

        eps = 5e-3
        systemSize = Symbol('systemSize')
        
        NrDP = int(self._tend/self._tstep) + 1
        time = np.linspace(0, self._tend, NrDP)
        
        initDict = self._getInitCondsFromSlider()
        
        SV1_0 = initDict[Symbol(latex(Symbol('Phi^0_'+str(self._stateVariable1))))]
        SV2_0 = initDict[Symbol(latex(Symbol('Phi^0_'+str(self._stateVariable2))))]
        y0 = [SV1_0, SV2_0]
        if self._stateVariable3:
            SV3_0 = initDict[Symbol(latex(Symbol('Phi^0_'+str(self._stateVariable3))))]
            y0.append(SV3_0)
            
        sol_ODE = odeint(self._get_eqsODE, y0, time)
        
        if self._stateVariable3:
            realEQsol, eigList = self._get_fixedPoints3d()
        else:
            realEQsol, eigList = self._get_fixedPoints2d()
   
        y_stationary = [sol_ODE[-1, kk] for kk in range(len(y0))]
        
        if realEQsol != [] and realEQsol != None:
            steadyStateReached = False
            for kk in range(len(realEQsol)):
                if self._stateVariable3:
                    if abs(realEQsol[kk][self._stateVariable1] - y_stationary[0]) <= eps and abs(realEQsol[kk][self._stateVariable2] - y_stationary[1]) <= eps and abs(realEQsol[kk][self._stateVariable3] - y_stationary[2]) <= eps:
                        if all(sign(re(lam)) < 0 for lam in eigList[kk]) == True:
                            steadyStateReached = True
                            steadyStateDict = {self._stateVariable1: realEQsol[kk][self._stateVariable1], self._stateVariable2: realEQsol[kk][self._stateVariable2], self._stateVariable3: realEQsol[kk][self._stateVariable3]}
    
                else:
                    if abs(realEQsol[kk][self._stateVariable1] - y_stationary[0]) <= eps and abs(realEQsol[kk][self._stateVariable2] - y_stationary[1]) <= eps:
                        if all(sign(re(lam)) < 0 for lam in eigList[kk]) == True:
                            steadyStateReached = True
                            steadyStateDict = {self._stateVariable1: realEQsol[kk][self._stateVariable1], self._stateVariable2: realEQsol[kk][self._stateVariable2]}
            
            if steadyStateReached == False:
                self._showErrorMessage('Stable steady state has not been reached: Try changing the initial conditions or model parameters using the sliders provided, increase simulation time, or decrease timestep tstep.') 
                return None
        else:
            steadyStateReached = 'uncertain'
            self._showErrorMessage('Warning: steady state may have not been reached. Substituted values of state variables at t=tend (tend can be set via keyword \'tend = <number>\').')
            if self._stateVariable3:
                steadyStateDict = {self._stateVariable1: y_stationary[0], self._stateVariable2: y_stationary[1], self._stateVariable3: y_stationary[2]}
            else:
                steadyStateDict = {self._stateVariable1: y_stationary[0], self._stateVariable2: y_stationary[1]}    
        
        with io.capture_output() as log:
            if steadyStateReached == 'uncertain':
                print('This plot depicts the noise-noise auto-correlation and cross-correlation functions around the following state (this might NOT be a steady state).')  
            else:  
                print('This plot depicts the noise-noise auto-correlation and cross-correlation functions around the following stable steady state:')
            for reactant in steadyStateDict:
                out = latex(Symbol('Phi^s_'+str(reactant))) + '=' + latex(steadyStateDict[reactant])
                display(Math(out))
        self._logs.append(log)
        
        argDict = self._get_argDict()
        for key in argDict:
            if key in self._mumotModel._constantReactants:
                argDict[Symbol('Phi_'+str(key))] = argDict.pop(key)
        for key in steadyStateDict:
            if key in self._mumotModel._reactants:
                steadyStateDict[Symbol('Phi_'+str(key))] = steadyStateDict.pop(key)
        
        EOM_1stOrdMomDict = copy.deepcopy(self._EOM_1stOrdMomDict)
        for sol in EOM_1stOrdMomDict:
            EOM_1stOrdMomDict[sol] = EOM_1stOrdMomDict[sol].subs(steadyStateDict)
            EOM_1stOrdMomDict[sol] = EOM_1stOrdMomDict[sol].subs(argDict)
        
        EOM_2ndOrdMomDict = copy.deepcopy(self._EOM_2ndOrdMomDict)
        
        SOL_2ndOrdMomDict = self._numericSol2ndOrdMoment(EOM_2ndOrdMomDict, steadyStateDict, argDict)
        
        time_depend_noise=[]
        for reactant in self._mumotModel._reactants:
            if reactant not in self._mumotModel._constantReactants:
                time_depend_noise.append(Symbol('eta_'+str(reactant)))
        
        noiseCorrEOM = []
        noiseCorrEOMdict = {}
        for sym in time_depend_noise:
            for key in EOM_1stOrdMomDict:
                noiseCorrEOMdict[sym*key] = expand(sym*EOM_1stOrdMomDict[key])
        
        eta_SV1 = Symbol('eta_'+str(self._stateVariable1))
        eta_SV2 = Symbol('eta_'+str(self._stateVariable2))
        M_1, M_2 = symbols('M_1 M_2')
        cVar1, cVar2, cVar3, cVar4 = symbols('cVar1 cVar2 cVar3 cVar4')
        if self._stateVariable3:
            eta_SV3 = Symbol('eta_'+str(self._stateVariable3))
            cVar5, cVar6, cVar7, cVar8, cVar9 = symbols('cVar5 cVar6 cVar7 cVar8 cVar9')
        
        cVarSubdict = {}    
        if len(time_depend_noise) == 2:
            cVarSubdict[eta_SV1*M_1(eta_SV1)] = cVar1
            cVarSubdict[eta_SV2*M_1(eta_SV2)] = cVar2
            cVarSubdict[eta_SV1*M_1(eta_SV2)] = cVar3
            cVarSubdict[eta_SV2*M_1(eta_SV1)] = cVar4
            #auto-correlation
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV1*M_1(eta_SV1)])
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV2*M_1(eta_SV2)])
            # cross-correlation
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV1*M_1(eta_SV2)])
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV2*M_1(eta_SV1)])
        elif len(time_depend_noise) == 3:
            cVarSubdict[eta_SV1*M_1(eta_SV1)] = cVar1
            cVarSubdict[eta_SV2*M_1(eta_SV2)] = cVar2
            cVarSubdict[eta_SV3*M_1(eta_SV3)] = cVar3
            cVarSubdict[eta_SV1*M_1(eta_SV2)] = cVar4
            cVarSubdict[eta_SV2*M_1(eta_SV1)] = cVar5
            cVarSubdict[eta_SV1*M_1(eta_SV3)] = cVar6
            cVarSubdict[eta_SV3*M_1(eta_SV1)] = cVar7
            cVarSubdict[eta_SV2*M_1(eta_SV3)] = cVar8
            cVarSubdict[eta_SV3*M_1(eta_SV2)] = cVar9
            #auto-correlation
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV1*M_1(eta_SV1)])
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV2*M_1(eta_SV2)])
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV3*M_1(eta_SV3)])
            # cross-correlation
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV1*M_1(eta_SV2)])
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV2*M_1(eta_SV1)])
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV1*M_1(eta_SV3)])
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV3*M_1(eta_SV1)])
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV2*M_1(eta_SV3)])
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV3*M_1(eta_SV2)])
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
                else:
                    dydt[kk] = dydt[kk].subs({cVar1: yin[0], cVar2: yin[1], cVar3: yin[2], cVar4: yin[3]})
                dydt[kk] = dydt[kk].evalf()
            return dydt
        
        NrDP = int(self._tendNoise/self._tstepNoise) + 1
        time = np.linspace(0, self._tendNoise, NrDP)
        if self._stateVariable3:
            y0 = [SOL_2ndOrdMomDict[M_2(eta_SV1**2)], SOL_2ndOrdMomDict[M_2(eta_SV2**2)], SOL_2ndOrdMomDict[M_2(eta_SV3**2)], 
                  SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV2)], SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV2)],
                  SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV3)], SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV3)],
                  SOL_2ndOrdMomDict[M_2(eta_SV2*eta_SV3)], SOL_2ndOrdMomDict[M_2(eta_SV2*eta_SV3)]]
        else:
            y0 = [SOL_2ndOrdMomDict[M_2(eta_SV1**2)], SOL_2ndOrdMomDict[M_2(eta_SV2**2)], 
                  SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV2)], SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV2)]]
        sol_ODE = odeint(noiseODEsys, y0, time) # sol_ODE overwritten
        
        x_data = [time for kk in range(len(y0))]  
        y_data = [sol_ODE[:, kk] for kk in range(len(y0))]
        noiseNorm = systemSize.subs(argDict)
        noiseNorm = N(noiseNorm)
        for nn in range(len(y_data)):
            y_temp=np.copy(y_data[nn])
            for kk in range(len(y_temp)):
                y_temp[kk] = y_temp[kk]/noiseNorm
            y_data[nn] = y_temp
            
        if self._stateVariable3:
            c_labels = [r'$<'+latex(eta_SV1)+'(t)'+latex(eta_SV1)+'(0)'+ '>$', 
                        r'$<'+latex(eta_SV2)+'(t)'+latex(eta_SV2)+'(0)'+ '>$',
                        r'$<'+latex(eta_SV3)+'(t)'+latex(eta_SV3)+'(0)'+ '>$', 
                        r'$<'+latex(eta_SV2)+'(t)'+latex(eta_SV1)+'(0)'+ '>$', 
                        r'$<'+latex(eta_SV1)+'(t)'+latex(eta_SV2)+'(0)'+ '>$',
                        r'$<'+latex(eta_SV3)+'(t)'+latex(eta_SV1)+'(0)'+ '>$', 
                        r'$<'+latex(eta_SV1)+'(t)'+latex(eta_SV3)+'(0)'+ '>$',
                        r'$<'+latex(eta_SV3)+'(t)'+latex(eta_SV2)+'(0)'+ '>$', 
                        r'$<'+latex(eta_SV2)+'(t)'+latex(eta_SV3)+'(0)'+ '>$']
            
        else:
            c_labels = [r'$<'+latex(eta_SV1)+'(t)'+latex(eta_SV1)+'(0)'+ '>$', 
                        r'$<'+latex(eta_SV2)+'(t)'+latex(eta_SV2)+'(0)'+ '>$', 
                        r'$<'+latex(eta_SV2)+'(t)'+latex(eta_SV1)+'(0)'+ '>$', 
                        r'$<'+latex(eta_SV1)+'(t)'+latex(eta_SV2)+'(0)'+ '>$']
       
        _fig_formatting_2D(xdata=x_data, ydata = y_data , xlab = self._xlab, ylab = self._ylab, 
                           fontsize=self._chooseFontSize, curvelab=c_labels, legend_loc=self._legend_loc, grid = True, 
                           legend_fontsize=self._legend_fontsize)
#        plt.set_aspect('equal') ## @todo
        
        
    def _numericSol2ndOrdMoment(self, EOM_2ndOrdMomDict, steadyStateDict, argDict):
        for sol in EOM_2ndOrdMomDict:
            EOM_2ndOrdMomDict[sol] = EOM_2ndOrdMomDict[sol].subs(steadyStateDict)
            EOM_2ndOrdMomDict[sol] = EOM_2ndOrdMomDict[sol].subs(argDict)
#         
#         eta_SV1 = Symbol('eta_'+str(self._stateVarList[0]))
#         eta_SV2 = Symbol('eta_'+str(self._stateVarList[1]))
#         M_1, M_2 = symbols('M_1 M_2')
#         if len(self._stateVarList) == 3:
#             eta_SV3 = Symbol('eta_'+str(self._stateVarList[2]))
#             
             
        eta_SV1 = Symbol('eta_'+str(self._stateVariable1))
        eta_SV2 = Symbol('eta_'+str(self._stateVariable2))
        M_1, M_2 = symbols('M_1 M_2')
        if self._stateVariable3:
            eta_SV3 = Symbol('eta_'+str(self._stateVariable3))
             

        SOL_2ndOrdMomDict = {} 
        EQsys2ndOrdMom = []
        if self._stateVariable3:
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV1*eta_SV1)])
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV2*eta_SV2)])
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV3*eta_SV3)])
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV1*eta_SV2)])
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV1*eta_SV3)])
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV2*eta_SV3)])
            SOL_2ndOrderMom = list(linsolve(EQsys2ndOrdMom, [M_2(eta_SV1*eta_SV1), 
                                                             M_2(eta_SV2*eta_SV2), 
                                                             M_2(eta_SV3*eta_SV3), 
                                                             M_2(eta_SV1*eta_SV2), 
                                                             M_2(eta_SV1*eta_SV3), 
                                                             M_2(eta_SV2*eta_SV3)]))[0] #only one set of solutions (if any) in linear system of equations; hence index [0]
            
            SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV1)] = SOL_2ndOrderMom[0]
            SOL_2ndOrdMomDict[M_2(eta_SV2*eta_SV2)] = SOL_2ndOrderMom[1]
            SOL_2ndOrdMomDict[M_2(eta_SV3*eta_SV3)] = SOL_2ndOrderMom[2]
            SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV2)] = SOL_2ndOrderMom[3]
            SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV3)] = SOL_2ndOrderMom[4]
            SOL_2ndOrdMomDict[M_2(eta_SV2*eta_SV3)] = SOL_2ndOrderMom[5]
        
        else:
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV1*eta_SV1)])
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV2*eta_SV2)])
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV1*eta_SV2)])
            SOL_2ndOrderMom = list(linsolve(EQsys2ndOrdMom, [M_2(eta_SV1*eta_SV1), 
                                                             M_2(eta_SV2*eta_SV2), 
                                                             M_2(eta_SV1*eta_SV2)]))[0] #only one set of solutions (if any) in linear system of equations
            
            SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV1)] = SOL_2ndOrderMom[0]
            SOL_2ndOrdMomDict[M_2(eta_SV2*eta_SV2)] = SOL_2ndOrderMom[1]
            SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV2)] = SOL_2ndOrderMom[2]
            
        return SOL_2ndOrdMomDict


    def _build_bookmark(self, includeParams = True):
        if not self._silent:
            logStr = "bookmark = "
        else:
            logStr = ""
        
        logStr += "<modelName>." + self._generatingCommand + "("
     
        if includeParams:
            logStr += self._get_bookmarks_params() + ", "
        if len(self._generatingKwargs) > 0:
            for key in self._generatingKwargs:
                if type(self._generatingKwargs[key]) == str:
                    logStr += key + " = " + "\'"+ str(self._generatingKwargs[key]) + "\'" + ", "
                else:
                    logStr += key + " = " + str(self._generatingKwargs[key]) + ", "
        logStr += "bookmark = False"
        logStr += ")"
        logStr = logStr.replace('\\', '\\\\')
        
        return logStr



## field view on model (specialised by MuMoTvectorView and MuMoTstreamView)
class MuMoTfieldView(MuMoTview):
    ## 1st state variable (x-dimension)
    _stateVariable1 = None
    ## 2nd state variable (y-dimension)
    _stateVariable2 = None
    ## 3rd state variable (z-dimension) 
    _stateVariable3 = None
    ## X ordinates array
    _X = None
    ## Y ordinates array
    _Y = None 
    ## Z ordinates array
    _Z = None
    ## X derivatives array 
    _X = None
    ## Y derivatives array
    _Y = None
    ## Z derivatives array
    _Z = None 
    ## speed array
    _speed = None 
    ## class-global dictionary of memoised masks with (mesh size, dimension) as key
    _mask = {} 
    
    def __init__(self, model, controller, stateVariable1, stateVariable2, stateVariable3 = None, figure = None, params = None, **kwargs):
        if model._systemSize == None and model._constantSystemSize == True:
            print("Cannot construct field-based plot until system size is set, using substitute()")
            return
        silent = kwargs.get('silent', False)
        super().__init__(model, controller, figure, params, **kwargs)
        
        if 'fontsize' in kwargs:
            self._chooseFontSize = kwargs['fontsize']
        else:
            self._chooseFontSize=None
        self._showFixedPoints = kwargs.get('showFixedPoints', False)
        self._xlab = kwargs.get('xlab', r'$'+latex(Symbol('Phi_'+str(stateVariable1)))+'$')
        self._ylab = kwargs.get('ylab', r'$'+latex(Symbol('Phi_'+str(stateVariable2)))+'$')
        if stateVariable3:
            self._zlab = kwargs.get('zlab', r'$'+latex(Symbol('Phi_'+str(stateVariable3)))+'$') 
        
        self._stateVariable1 = process_sympy(stateVariable1)
        self._stateVariable2 = process_sympy(stateVariable2)
        if stateVariable3 != None:
            self._axes3d = True
            self._stateVariable3 = process_sympy(stateVariable3)
        _mask = {}

        if not(silent):
            self._plot_field()

    


#    def __init__(self, model, controller, stateVariable1, stateVariable2, stateVariable3 = None, figure = None, params = None, **kwargs):

    def _build_bookmark(self, includeParams = True):
        if not self._silent:
            logStr = "bookmark = "
        else:
            logStr = ""
        logStr += "<modelName>." + self._generatingCommand + "('" + str(self._stateVariable1) + "', '" + str(self._stateVariable2) +"', "
        if self._stateVariable3 != None:
            logStr += "'" + str(self._stateVariable3) + "', "
        if includeParams:
            logStr += self._get_bookmarks_params() + ", "
        if len(self._generatingKwargs) > 0:
            for key in self._generatingKwargs:
                logStr += key + " = " + str(self._generatingKwargs[key]) + ", "
        logStr += "bookmark = False"
        logStr += ")"
        logStr = logStr.replace('\\', '\\\\')
        
        return logStr

            
    ## calculates stationary states of 2d system
#     def _get_fixedPoints2d(self):
#         #plotLimits = self._controller._getPlotLimits()
#         if self._controller != None:
#             paramNames = []
#             paramValues = []
#             for name, value in self._controller._widgetsFreeParams.items():
#                 # throw away formatting for constant reactants
# #                 name = name.replace('(','')
# #                 name = name.replace(')','')
#                 paramNames.append(name)
#                 paramValues.append(value.value)
#         else:
#             paramNames = map(str, self._paramNames)
#             paramValues = self._paramValues            
#         funcs = self._mumotModel._getFuncs()
#         argNamesSymb = list(map(Symbol, paramNames))
#         argDict = dict(zip(argNamesSymb, paramValues))
#         argDict[self._mumotModel._systemSize] = 1
#        
# #         if self._controller != None:
# #             paramNames = []
# #             paramValues = []
# #             for name, value in self._controller._widgetsFreeParams.items():
# #                 paramNames.append(name)
# #                 paramValues.append(value.value)          
# #         else:
# #             paramNames = map(str, self._paramNames)
# #             paramValues = self._paramValues           
# #             
# #         argNamesSymb = list(map(Symbol, paramNames))
# #         argDict = dict(zip(argNamesSymb, paramValues))
# #         argDict[self._mumotModel._systemSize] = 1
#         
#         EQ1 = self._mumotModel._equations[self._stateVariable1].subs(argDict)
#         EQ2 = self._mumotModel._equations[self._stateVariable2].subs(argDict)
#         eps=1e-8
#         EQsol = solve((EQ1, EQ2), (self._stateVariable1, self._stateVariable2), dict=True)
#         #for kk in range(len(EQsol)):
#         #    print(EQsol[kk])
#         realEQsol = [{self._stateVariable1: re(EQsol[kk][self._stateVariable1]), self._stateVariable2: re(EQsol[kk][self._stateVariable2])} for kk in range(len(EQsol)) if (Abs(im(EQsol[kk][self._stateVariable1]))<=eps and Abs(im(EQsol[kk][self._stateVariable2]))<=eps)]
#         
#         MAT = Matrix([EQ1, EQ2])
#         JAC = MAT.jacobian([self._stateVariable1,self._stateVariable2])
#         
#         eigList = []
#         #for nn in range(len(realEQsol)): 
#         #    JACsub=JAC.subs([(self._stateVariable1, realEQsol[nn][self._stateVariable1]), (self._stateVariable2, realEQsol[nn][self._stateVariable2])])
#         #    evSet = JACsub.eigenvals()
#         #    eigList.append(evSet)
#         for nn in range(len(realEQsol)): 
#             evSet = {}
#             JACsub=JAC.subs([(self._stateVariable1, realEQsol[nn][self._stateVariable1]), (self._stateVariable2, realEQsol[nn][self._stateVariable2])])
#             #evSet = JACsub.eigenvals()
#             eigVects = JACsub.eigenvects()
#             for kk in range(len(eigVects)):
#                 evSet[eigVects[kk][0]] = (eigVects[kk][1], eigVects[kk][2])
#             eigList.append(evSet)
#         return realEQsol, eigList #returns two lists of dictionaries
#     
#     ## calculates stationary states of 3d system
#     def _get_fixedPoints3d(self):
#         #plotLimits = self._controller._getPlotLimits()
#         if self._controller != None:
#             paramNames = []
#             paramValues = []
#             for name, value in self._controller._widgetsFreeParams.items():
#                 # throw away formatting for constant reactants
# #                 name = name.replace('(','')
# #                 name = name.replace(')','')
#                 paramNames.append(name)
#                 paramValues.append(value.value)
#         else:
#             paramNames = map(str, self._paramNames)
#             paramValues = self._paramValues            
#         funcs = self._mumotModel._getFuncs()
#         argNamesSymb = list(map(Symbol, paramNames))
#         argDict = dict(zip(argNamesSymb, paramValues))
#         argDict[self._mumotModel._systemSize] = 1
#         
# #         if self._controller != None:
# #             paramNames = []
# #             paramValues = []
# #             for name, value in self._controller._widgetsFreeParams.items():
# #                 paramNames.append(name)
# #                 paramValues.append(value.value)          
# #         else:
# #             paramNames = map(str, self._paramNames)
# #             paramValues = self._paramValues           
# #             
# #         argNamesSymb = list(map(Symbol, paramNames))
# #         argDict = dict(zip(argNamesSymb, paramValues))
# #         argDict[self._mumotModel._systemSize] = 1
#         
#         EQ1 = self._mumotModel._equations[self._stateVariable1].subs(argDict)
#         EQ2 = self._mumotModel._equations[self._stateVariable2].subs(argDict)
#         EQ3 = self._mumotModel._equations[self._stateVariable3].subs(argDict)
#         eps=1e-8
#         EQsol = solve((EQ1, EQ2, EQ3), (self._stateVariable1, self._stateVariable2, self._stateVariable3), dict=True)
#         realEQsol = [{self._stateVariable1: re(EQsol[kk][self._stateVariable1]), self._stateVariable2: re(EQsol[kk][self._stateVariable2]), self._stateVariable3: re(EQsol[kk][self._stateVariable3])} for kk in range(len(EQsol)) if (Abs(im(EQsol[kk][self._stateVariable1]))<=eps and Abs(im(EQsol[kk][self._stateVariable2]))<=eps and Abs(im(EQsol[kk][self._stateVariable3]))<=eps)]
#         
#         MAT = Matrix([EQ1, EQ2, EQ3])
#         JAC = MAT.jacobian([self._stateVariable1,self._stateVariable2,self._stateVariable3])
#         
#         eigList = []
#         #for nn in range(len(realEQsol)): 
#         #    JACsub=JAC.subs([(self._stateVariable1, realEQsol[nn][self._stateVariable1]), (self._stateVariable2, realEQsol[nn][self._stateVariable2]), (self._stateVariable3, realEQsol[nn][self._stateVariable3])])
#         #    evSet = JACsub.eigenvals()
#         #    eigList.append(evSet)
#         for nn in range(len(realEQsol)): 
#             evSet = {}
#             JACsub=JAC.subs([(self._stateVariable1, realEQsol[nn][self._stateVariable1]), (self._stateVariable2, realEQsol[nn][self._stateVariable2]), (self._stateVariable3, realEQsol[nn][self._stateVariable3])])
#             #evSet = JACsub.eigenvals()
#             eigVects = JACsub.eigenvects()
#             for kk in range(len(eigVects)):
#                 evSet[eigVects[kk][0]] = (eigVects[kk][1], eigVects[kk][2])
#             eigList.append(evSet)
#         return realEQsol, eigList #returns two lists of dictionaries
    
    
    def _plot_field(self):
        if not(self._silent): ## @todo is this necessary?
            plt.figure(self._figureNum)
            plt.clf()
            self._resetErrorMessage()
        self._showErrorMessage(str(self))

    ## helper for _get_field_2d() and _get_field_3d()
    def _get_field(self):
        plotLimits = self._getPlotLimits()
        paramNames = []
        paramValues = []
        if self._controller is not None:
            for name, value in self._controller._widgetsFreeParams.items():
                # throw away formatting for constant reactants
#                 name = name.replace('(','')
#                 name = name.replace(')','')
                paramNames.append(name)
                paramValues.append(value.value)
        if self._paramNames is not None:
            paramNames += map(str, self._paramNames)
            paramValues += self._paramValues           
        funcs = self._mumotModel._getFuncs()
        argNamesSymb = list(map(Symbol, paramNames))
        argDict = dict(zip(argNamesSymb, paramValues))
        
        
        return (funcs, argNamesSymb, argDict, paramNames, paramValues, plotLimits)

    ## get 2-dimensional field for plotting
    def _get_field2d(self, kind, meshPoints, plotLimits = 1):
        with io.capture_output() as log:
            self._log(kind)
            (funcs, argNamesSymb, argDict, paramNames, paramValues, plotLimits) = self._get_field()
            self._Y, self._X = np.mgrid[0:plotLimits:complex(0, meshPoints), 0:plotLimits:complex(0, meshPoints)]
            if self._mumotModel._constantSystemSize:
                mask = self._mask.get((meshPoints, 2))
                if mask is None:
                    mask = np.zeros(self._X.shape, dtype=bool)
                    upperright = np.triu_indices(meshPoints) ## @todo: allow user to set mesh points with keyword 
                    mask[upperright] = True
                    np.fill_diagonal(mask, False)
                    mask = np.flipud(mask)
                    self._mask[(meshPoints, 2)] = mask
            self._Xdot = funcs[self._stateVariable1](*self._mumotModel._getArgTuple2d(paramNames, paramValues, argDict, self._stateVariable1, self._stateVariable2, self._X, self._Y))
            self._Ydot = funcs[self._stateVariable2](*self._mumotModel._getArgTuple2d(paramNames, paramValues, argDict, self._stateVariable1, self._stateVariable2, self._X, self._Y))
            try:
                self._speed = np.log(np.sqrt(self._Xdot ** 2 + self._Ydot ** 2))
            except:
#                self._speed = np.ones(self._X.shape, dtype=float)
                self._speed = None
            if self._mumotModel._constantSystemSize:
                self._Xdot = np.ma.array(self._Xdot, mask=mask)
                self._Ydot = np.ma.array(self._Ydot, mask=mask)        
        self._logs.append(log)

    ## get 3-dimensional field for plotting        
    def _get_field3d(self, kind, meshPoints, plotLimits = 1):
        with io.capture_output() as log:
            self._log(kind)
            (funcs, argNamesSymb, argDict, paramNames, paramValues, plotLimits) = self._get_field()
            self._Z, self._Y, self._X = np.mgrid[0:plotLimits:complex(0, meshPoints), 0:plotLimits:complex(0, meshPoints), 0:plotLimits:complex(0, meshPoints)]
            if self._mumotModel._constantSystemSize:
                mask = self._mask.get((meshPoints, 3))
                if mask is None:
    #                mask = np.zeros(self._X.shape, dtype=bool)
    #                 upperright = np.triu_indices(meshPoints) ## @todo: allow user to set mesh points with keyword 
    #                 mask[upperright] = True
    #                 np.fill_diagonal(mask, False)
    #                 mask = np.flipud(mask)
                    mask = self._X + self._Y + self._Z >= 1
    #                mask = mask.astype(int)
                    self._mask[(meshPoints, 3)] = mask
            self._Xdot = funcs[self._stateVariable1](*self._mumotModel._getArgTuple3d(paramNames, paramValues, argDict, self._stateVariable1, self._stateVariable2, self._stateVariable3, self._X, self._Y, self._Z))
            self._Ydot = funcs[self._stateVariable2](*self._mumotModel._getArgTuple3d(paramNames, paramValues, argDict, self._stateVariable1, self._stateVariable2, self._stateVariable3, self._X, self._Y, self._Z))
            self._Zdot = funcs[self._stateVariable3](*self._mumotModel._getArgTuple3d(paramNames, paramValues, argDict, self._stateVariable1, self._stateVariable2, self._stateVariable3, self._X, self._Y, self._Z))
            try:
                self._speed = np.log(np.sqrt(self._Xdot ** 2 + self._Ydot ** 2 + self._Zdot ** 2))
            except:
                self._speed = None
#            self._Xdot = self._Xdot * mask
#            self._Ydot = self._Ydot * mask
#            self._Zdot = self._Zdot * mask
            if self._mumotModel._constantSystemSize:
                self._Xdot = np.ma.array(self._Xdot, mask=mask)
                self._Ydot = np.ma.array(self._Ydot, mask=mask)        
                self._Zdot = np.ma.array(self._Zdot, mask=mask)
        self._logs.append(log)


# ## stream plot with noise ellipses view on model
# class MuMoTnoiseView(MuMoTfieldView):
#     ## solution 1st order moments (noise in vKE)
#     _SOL_1stOrderMomDict = None
#     ## replacement dictionary for symbols of 1st order moments
#     _NoiseSubs1stOrder = None
#     ## solution 2nd order moments (noise in vKE)
#     _SOL_2ndOrdMomDict = None
#     ## replacement dictionary for symbols of 2nd order moments
#     _NoiseSubs2ndOrder = None
#     ## set for checking number of time-dependent reactants
#     _checkReactants = None
#     ## set for checking constant reactants
#     _checkConstReactants = None
#     
#     _noiseStatSol = None
#  
#     def __init__(self, model, controller, SOL_2ndOrd, stateVariable1, stateVariable2, stateVariable3=None, figure = None, params = None, **kwargs):
#         if model._systemSize == None and model._constantSystemSize == True:
#             print("Cannot construct field-based plot until system size is set, using substitute()")
#             return
#         #self._noiseStatSol = noiseStatSol
#         #self._SOL_1stOrderMomDict, self._NoiseSubs1stOrder, self._SOL_2ndOrdMomDict, self._NoiseSubs2ndOrder = self._noiseStatSol
#         self._SOL_2ndOrdMomDict = SOL_2ndOrd
#         self._checkReactants = model._reactants
#         if model._constantReactants:
#             self._checkConstReactants = model._constantReactants
#         silent = kwargs.get('silent', False)
#         super().__init__(model, controller, stateVariable1, stateVariable2, stateVariable3, figure, params, **kwargs) 
#         self._generatingCommand = "mmt.MuMoTnoiseView"
#         
#     def _plot_field(self):
#         super()._plot_field()
#         
#         Phi_stateVar1 = Symbol('Phi_'+str(self._stateVariable1)) 
#         Phi_stateVar2 = Symbol('Phi_'+str(self._stateVariable2))
#         eta_stateVar1 = Symbol('eta_'+str(self._stateVariable1)) 
#         eta_stateVar2 = Symbol('eta_'+str(self._stateVariable2))
#         M_2 = Symbol('M_2')
#         
#         eta_cross = eta_stateVar1*eta_stateVar2
#         for key in self._SOL_2ndOrdMomDict:
#             if key == self._SOL_2ndOrdMomDict[key] or key in self._SOL_2ndOrdMomDict[key].args:
#                 for key2 in self._SOL_2ndOrdMomDict:
#                     if key2 != key and key2 != M_2(eta_cross):
#                         self._SOL_2ndOrdMomDict[key] = self._SOL_2ndOrdMomDict[key].subs(key, self._SOL_2ndOrdMomDict[key2])
#                    
#         checkReactants = copy.deepcopy(self._checkReactants)
#         if self._checkConstReactants:
#             checkConstReactants = copy.deepcopy(self._checkConstReactants)
#             for reactant in checkReactants:
#                 if reactant in checkConstReactants:
#                     checkReactants.remove(reactant)
#         assert (len(checkReactants) == 2),"Not implemented: This feature is available only for systems with exactly 2 time-dependent reactants!"
#         realEQsol, eigList = self._get_fixedPoints2d()
#         systemSize = Symbol('systemSize')
#         argDict = self._get_argDict()
#         for key in argDict:
#             if key in self._mumotModel._constantReactants:
#                 argDict[Symbol('Phi_'+str(key))] = argDict.pop(key)
#         
#         PhiSubList = []
#         for kk in range(len(realEQsol)):
#             PhiSubDict={}
#             for solXi in realEQsol[kk]:
#                 PhiSubDict[Symbol('Phi_'+str(solXi))] = realEQsol[kk][solXi]
#             PhiSubList.append(PhiSubDict)
#         
#         SOL_2ndOrdMomDictList = []
#         for nn in range(len(PhiSubList)):
#             SOL_2ndOrdMomDict = copy.deepcopy(self._SOL_2ndOrdMomDict)
#             for sol in SOL_2ndOrdMomDict:
#                 SOL_2ndOrdMomDict[sol] = SOL_2ndOrdMomDict[sol].subs(PhiSubList[nn])
#                 SOL_2ndOrdMomDict[sol] = SOL_2ndOrdMomDict[sol].subs(argDict)
#             SOL_2ndOrdMomDictList.append(SOL_2ndOrdMomDict)
#             
#         Evects = []
#         EvectsPlot = []
#         EV = []
#         EVplot = []
#         for kk in range(len(eigList)):
#             EVsub = []
#             EvectsSub = []
#             for key in eigList[kk]:
#                 for jj in range(len(eigList[kk][key][1])):
#                     EvectsSub.append(eigList[kk][key][1][jj].evalf())
#                 if eigList[kk][key][0] >1:
#                     for jj in range(eigList[kk][key][0]):
#                         EVsub.append(key.evalf())
#                 else:
#                     EVsub.append(key.evalf())
#                     
#             EV.append(EVsub)
#             Evects.append(EvectsSub)
#             if self._mumotModel._constantSystemSize == True:
#                 if (0 <= re(realEQsol[kk][self._stateVariable1]) <= 1 and 0 <= re(realEQsol[kk][self._stateVariable2]) <= 1):
#                     EVplot.append(EVsub)
#                     EvectsPlot.append(EvectsSub)
#             else:
#                 EVplot.append(EVsub)
#                 EvectsPlot.append(EvectsSub)
# 
#         angle_ell_list = []
#         projection_angle_list = []
#         vec1 = Matrix([ [0], [1] ])
#         for nn in range(len(EvectsPlot)):
#             vec2 = EvectsPlot[nn][0]
#             vec2norm = vec2.norm()
#             vec2 = vec2/vec2norm
#             vec3 = EvectsPlot[nn][0]
#             vec3norm = vec3.norm()
#             vec3 = vec3/vec3norm
#             if vec2norm >= vec3norm:
#                 angle_ell = acos(vec1.dot(vec2)/(vec1.norm()*vec2.norm())).evalf()
#             else:
#                 angle_ell = acos(vec1.dot(vec3)/(vec1.norm()*vec3.norm())).evalf()
#             angle_ell = angle_ell.evalf()
#             projection_angle_list.append(angle_ell)
#             angle_ell_deg = 180*angle_ell/pi.evalf()
#             angle_ell_list.append(round(angle_ell_deg,5))
#         projection_angle_list = [abs(projection_angle_list[kk]) if abs(projection_angle_list[kk]) <= N(pi/2) else N(pi)-abs(projection_angle_list[kk]) for kk in range(len(projection_angle_list))]
#         
#         if self._mumotModel._constantSystemSize == True:
#             FixedPoints=[[PhiSubList[kk][Phi_stateVar1] for kk in range(len(PhiSubList)) if (0 <= re(PhiSubList[kk][Phi_stateVar1]) <= 1 and 0 <= re(PhiSubList[kk][Phi_stateVar2]) <= 1)], 
#                          [PhiSubList[kk][Phi_stateVar2] for kk in range(len(PhiSubList)) if (0 <= re(PhiSubList[kk][Phi_stateVar1]) <= 1 and 0 <= re(PhiSubList[kk][Phi_stateVar2]) <= 1)]]
#             Ell_width = [re(cos(N(pi/2)-projection_angle_list[kk])*sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar2**2)]/systemSize.subs(argDict)) + sin(N(pi/2)-projection_angle_list[kk])*sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar1**2)]/systemSize.subs(argDict))) for kk in range(len(SOL_2ndOrdMomDictList)) if (0 <= re(PhiSubList[kk][Phi_stateVar1]) <= 1 and 0 <= re(PhiSubList[kk][Phi_stateVar2]) <= 1)]
#             Ell_height = [re(cos(projection_angle_list[kk])*sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar2**2)]/systemSize.subs(argDict)) + sin(projection_angle_list[kk])*sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar1**2)]/systemSize.subs(argDict))) for kk in range(len(SOL_2ndOrdMomDictList)) if (0 <= re(PhiSubList[kk][Phi_stateVar1]) <= 1 and 0 <= re(PhiSubList[kk][Phi_stateVar2]) <= 1)]
#             #FixedPoints=[[realEQsol[kk][self._stateVariable1] for kk in range(len(realEQsol)) if (0 <= re(realEQsol[kk][self._stateVariable1]) <= 1 and 0 <= re(realEQsol[kk][self._stateVariable2]) <= 1)], 
#             #             [realEQsol[kk][self._stateVariable2] for kk in range(len(realEQsol)) if (0 <= re(realEQsol[kk][self._stateVariable1]) <= 1 and 0 <= re(realEQsol[kk][self._stateVariable2]) <= 1)]]
#         else:
#             FixedPoints=[[PhiSubList[kk][Phi_stateVar1] for kk in range(len(PhiSubList))], 
#                          [PhiSubList[kk][Phi_stateVar2] for kk in range(len(PhiSubList))]]
#             Ell_width = [re(cos(N(pi/2)-projection_angle_list[kk])*sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar2**2)]/systemSize.subs(argDict)) + sin(N(pi/2)-projection_angle_list[kk])*sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar1**2)]/systemSize.subs(argDict))) for kk in range(len(SOL_2ndOrdMomDictList))]
#             Ell_height = [re(cos(projection_angle_list[kk])*sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar2**2)]/systemSize.subs(argDict)) + sin(projection_angle_list[kk])*sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar1**2)]/systemSize.subs(argDict))) for kk in range(len(SOL_2ndOrdMomDictList))]
#             #FixedPoints=[[realEQsol[kk][self._stateVariable1] for kk in range(len(realEQsol))], 
#             #             [realEQsol[kk][self._stateVariable2] for kk in range(len(realEQsol))]]
#             #Ell_width = [SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar1**2)] for kk in range(len(SOL_2ndOrdMomDictList))]
#             #Ell_height = [SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar2**2)] for kk in range(len(SOL_2ndOrdMomDictList))] 
#         
#         # swap width and height of ellipse if width > height
#         for kk in range(len(Ell_width)):
#             ell_width_temp = Ell_width[kk]
#             ell_height_temp = Ell_height[kk]
#             if ell_width_temp > ell_height_temp:
#                 Ell_height[kk] = ell_width_temp
#                 Ell_width[kk] = ell_height_temp    
#         
#         FixedPoints.append(EVplot) 
#         
#         self._get_field2d("2d stream plot", 100) ## @todo: allow user to set mesh points with keyword
#         
#         if self._speed is not None:
#             fig_stream=plt.streamplot(self._X, self._Y, self._Xdot, self._Ydot, color = self._speed, cmap = 'gray') ## @todo: define colormap by user keyword
#         else:
#             fig_stream=plt.streamplot(self._X, self._Y, self._Xdot, self._Ydot, color = 'k') ## @todo: define colormap by user keyword
#         ells = [mpatch.Ellipse(xy=[FixedPoints[0][nn],FixedPoints[1][nn]], width=Ell_width[nn]/systemSize.subs(argDict), height=Ell_height[nn]/systemSize.subs(argDict), angle= round(angle_ell_list[nn],5)) for nn in range(len(FixedPoints[0]))]
#         ax = plt.gca()
#         for kk in range(len(ells)):
#             ax.add_artist(ells[kk])
#             ells[kk].set_alpha(0.25)
#             if re(EVplot[kk][0]) < 0 and re(EVplot[kk][1]) < 0:
#                 Fcolor = line_color_list[1]
#             elif re(EVplot[kk][0]) > 0 and re(EVplot[kk][1]) > 0:
#                 Fcolor = line_color_list[2]
#             else:
#                 Fcolor = line_color_list[0]
#             ells[kk].set_facecolor(Fcolor)
#         if self._mumotModel._constantSystemSize == True:
#             plt.fill_between([0,1], [1,0], [1,1], color='grey', alpha='0.25')            
#             plt.xlim(0,1)
#             plt.ylim(0,1)
#         else:
#             plt.xlim(0,self._X.max())
#             plt.ylim(0,self._Y.max())
#         
#         _fig_formatting_2D(figure=fig_stream, xlab = self._xlab, specialPoints=FixedPoints, showFixedPoints=True, 
#                            ax_reformat=False, curve_replot=False, ylab = self._ylab, fontsize=self._chooseFontSize)
# #        plt.set_aspect('equal') ## @todo
# 
#         if self._showFixedPoints==True:
#             with io.capture_output() as log:
#                 for kk in range(len(realEQsol)):
#                     print('Fixed point'+str(kk+1)+':', realEQsol[kk], 'with eigenvalues:', str(EV[kk]),
#                           'and eigenvectors:', str(Evects[kk]))
#             self._logs.append(log)
# 
# ## stream plot view on model
# class MuMoTstreamView(MuMoTfieldView):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._generatingCommand = "stream"
# 
#     def _plot_field(self):
#         super()._plot_field()
#         if self._showFixedPoints==True:
#             realEQsol, eigList = self._get_fixedPoints2d()
#             
#             EV = []
#             EVplot = []
#             for kk in range(len(eigList)):
#                 EVsub = []
#                 for key in eigList[kk]:
#                     if eigList[kk][key][0] >1:
#                         for jj in range(eigList[kk][key][0]):
#                             EVsub.append(key.evalf())
#                     else:
#                         EVsub.append(key.evalf())
#                         
#                 EV.append(EVsub)
#                 if self._mumotModel._constantSystemSize == True:
#                     if (0 <= re(realEQsol[kk][self._stateVariable1]) <= 1 and 0 <= re(realEQsol[kk][self._stateVariable2]) <= 1):
#                         EVplot.append(EVsub)
#                 else:
#                     EVplot.append(EVsub)
#             
#             #EV = []
#             #EVplot = []
#             #for kk in range(len(eigList)):
#             #    EVsub = []
#             #    for key in eigList[kk]:
#             #        EVsub.append(key.evalf())
#             #    EV.append(EVsub)
#             #    if (0 <= re(realEQsol[kk][self._stateVariable1]) <= 1 and 0 <= re(realEQsol[kk][self._stateVariable2]) <= 1):
#             #        EVplot.append(EVsub)
# 
#             if self._mumotModel._constantSystemSize == True:
#                 FixedPoints=[[realEQsol[kk][self._stateVariable1] for kk in range(len(realEQsol)) if (0 <= re(realEQsol[kk][self._stateVariable1]) <= 1 and 0 <= re(realEQsol[kk][self._stateVariable2]) <= 1)], 
#                              [realEQsol[kk][self._stateVariable2] for kk in range(len(realEQsol)) if (0 <= re(realEQsol[kk][self._stateVariable1]) <= 1 and 0 <= re(realEQsol[kk][self._stateVariable2]) <= 1)]]
#             else:
#                 FixedPoints=[[realEQsol[kk][self._stateVariable1] for kk in range(len(realEQsol))], 
#                              [realEQsol[kk][self._stateVariable2] for kk in range(len(realEQsol))]]
#             FixedPoints.append(EVplot)
#         else:
#             FixedPoints = None
#             
#         self._get_field2d("2d stream plot", 100) ## @todo: allow user to set mesh points with keyword
#         if self._speed is not None:
#             fig_stream=plt.streamplot(self._X, self._Y, self._Xdot, self._Ydot, color = self._speed, cmap = 'gray') ## @todo: define colormap by user keyword
#         else:
#             fig_stream=plt.streamplot(self._X, self._Y, self._Xdot, self._Ydot, color = 'k') ## @todo: define colormap by user keyword
#         if self._mumotModel._constantSystemSize == True:
#             plt.fill_between([0,1], [1,0], [1,1], color='grey', alpha='0.25')            
#             plt.xlim(0,1)
#             plt.ylim(0,1)
#         else:
#             plt.xlim(0,self._X.max())
#             plt.ylim(0,self._Y.max())
#         
#         _fig_formatting_2D(figure=fig_stream, xlab = self._xlab, specialPoints=FixedPoints, showFixedPoints=self._showFixedPoints, ax_reformat=False, curve_replot=False,
#                    ylab = self._ylab, fontsize=self._chooseFontSize)
# #        plt.set_aspect('equal') ## @todo
# 
#         if self._showFixedPoints==True:
#             with io.capture_output() as log:
#                 for kk in range(len(realEQsol)):
#                     print('Fixed point'+str(kk+1)+':', realEQsol[kk], 'with eigenvalues:', str(EV[kk]))
#             self._logs.append(log)


class MuMoTstreamView(MuMoTfieldView):
    ## dictionary containing the solutions of the second order noise moments in the stationary state
    _SOL_2ndOrdMomDict = None
    ## set of all reactants
    _checkReactants = None
    ## set of all constant reactants to get intersection with _checkReactants
    _checkConstReactants = None
    
    def __init__(self, model, controller, SOL_2ndOrd, stateVariable1, stateVariable2, stateVariable3=None, figure = None, params = None, **kwargs):
        #if model._systemSize == None and model._constantSystemSize == True:
        #    self._showErrorMessage("Cannot construct field-based plot until system size is set, using substitute()")
        #    return
        self._SOL_2ndOrdMomDict = SOL_2ndOrd
        #if self._SOL_2ndOrdMomDict == None:
        #    self._showErrorMessage('Noise in the system could not be calculated: \'showNoise\' automatically disabled.')

        self._checkReactants = model._reactants
        if model._constantReactants:
            self._checkConstReactants = model._constantReactants
        else:
            self._checkConstReactants = None
        silent = kwargs.get('silent', False)
        super().__init__(model, controller, stateVariable1, stateVariable2, stateVariable3, figure, params, **kwargs) 
        self._generatingCommand = "stream"

    def _plot_field(self, _=None):
        super()._plot_field()
        
        # check number of time-dependent reactants
        checkReactants = copy.deepcopy(self._checkReactants)
        if self._checkConstReactants:
            checkConstReactants = copy.deepcopy(self._checkConstReactants)
            for reactant in checkReactants:
                if reactant in checkConstReactants:
                    checkReactants.remove(reactant)
        if len(checkReactants) != 2:
            self._showErrorMessage("Not implemented: This feature is available only for systems with exactly 2 time-dependent reactants!")
                        
        if self._showFixedPoints==True or self._SOL_2ndOrdMomDict != None:
            Phi_stateVar1 = Symbol('Phi_'+str(self._stateVariable1)) 
            Phi_stateVar2 = Symbol('Phi_'+str(self._stateVariable2))
            eta_stateVar1 = Symbol('eta_'+str(self._stateVariable1)) 
            eta_stateVar2 = Symbol('eta_'+str(self._stateVariable2))
            M_2 = Symbol('M_2')
            
            systemSize = Symbol('systemSize')
            argDict = self._get_argDict()
            for key in argDict:
                if key in self._mumotModel._constantReactants:
                    argDict[Symbol('Phi_'+str(key))] = argDict.pop(key)
         
            realEQsol, eigList = self._get_fixedPoints2d()
        
            PhiSubList = []
            for kk in range(len(realEQsol)):
                PhiSubDict={}
                for solXi in realEQsol[kk]:
                    PhiSubDict[Symbol('Phi_'+str(solXi))] = realEQsol[kk][solXi]
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
                    if eigList[kk][key][0] >1:
                        for jj in range(eigList[kk][key][0]):
                            EVsub.append(key.evalf())
                    else:
                        EVsub.append(key.evalf())
                        
                EV.append(EVsub)
                Evects.append(EvectsSub)
                if self._mumotModel._constantSystemSize == True:
                    if (0 <= re(realEQsol[kk][self._stateVariable1]) <= 1 and 0 <= re(realEQsol[kk][self._stateVariable2]) <= 1):
                        EVplot.append(EVsub)
                        EvectsPlot.append(EvectsSub)
                else:
                    EVplot.append(EVsub)
                    EvectsPlot.append(EvectsSub)
        
        if self._SOL_2ndOrdMomDict:
            eta_cross = eta_stateVar1*eta_stateVar2
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
            vec1 = Matrix([ [0], [1] ])
            for nn in range(len(EvectsPlot)):
                vec2 = EvectsPlot[nn][0]
                vec2norm = vec2.norm()
                vec2 = vec2/vec2norm
                vec3 = EvectsPlot[nn][0]
                vec3norm = vec3.norm()
                vec3 = vec3/vec3norm
                if vec2norm >= vec3norm:
                    angle_ell = acos(vec1.dot(vec2)/(vec1.norm()*vec2.norm())).evalf()
                else:
                    angle_ell = acos(vec1.dot(vec3)/(vec1.norm()*vec3.norm())).evalf()
                angle_ell = angle_ell.evalf()
                projection_angle_list.append(angle_ell)
                angle_ell_deg = 180*angle_ell/pi.evalf()
                angle_ell_list.append(round(angle_ell_deg,5))
            projection_angle_list = [abs(projection_angle_list[kk]) if abs(projection_angle_list[kk]) <= N(pi/2) else N(pi)-abs(projection_angle_list[kk]) for kk in range(len(projection_angle_list))]
        
        if self._showFixedPoints==True or self._SOL_2ndOrdMomDict != None:
            if self._mumotModel._constantSystemSize == True:
                FixedPoints=[[PhiSubList[kk][Phi_stateVar1] for kk in range(len(PhiSubList)) if (0 <= re(PhiSubList[kk][Phi_stateVar1]) <= 1 and 0 <= re(PhiSubList[kk][Phi_stateVar2]) <= 1)], 
                             [PhiSubList[kk][Phi_stateVar2] for kk in range(len(PhiSubList)) if (0 <= re(PhiSubList[kk][Phi_stateVar1]) <= 1 and 0 <= re(PhiSubList[kk][Phi_stateVar2]) <= 1)]]
                if self._SOL_2ndOrdMomDict:
                    Ell_width = [re(cos(N(pi/2)-projection_angle_list[kk])*sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar2**2)]/systemSize.subs(argDict)) + sin(N(pi/2)-projection_angle_list[kk])*sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar1**2)]/systemSize.subs(argDict))) for kk in range(len(SOL_2ndOrdMomDictList)) if (0 <= re(PhiSubList[kk][Phi_stateVar1]) <= 1 and 0 <= re(PhiSubList[kk][Phi_stateVar2]) <= 1)]
                    Ell_height = [re(cos(projection_angle_list[kk])*sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar2**2)]/systemSize.subs(argDict)) + sin(projection_angle_list[kk])*sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar1**2)]/systemSize.subs(argDict))) for kk in range(len(SOL_2ndOrdMomDictList)) if (0 <= re(PhiSubList[kk][Phi_stateVar1]) <= 1 and 0 <= re(PhiSubList[kk][Phi_stateVar2]) <= 1)]
                #FixedPoints=[[realEQsol[kk][self._stateVariable1] for kk in range(len(realEQsol)) if (0 <= re(realEQsol[kk][self._stateVariable1]) <= 1 and 0 <= re(realEQsol[kk][self._stateVariable2]) <= 1)], 
                #             [realEQsol[kk][self._stateVariable2] for kk in range(len(realEQsol)) if (0 <= re(realEQsol[kk][self._stateVariable1]) <= 1 and 0 <= re(realEQsol[kk][self._stateVariable2]) <= 1)]]
            else:
                FixedPoints=[[PhiSubList[kk][Phi_stateVar1] for kk in range(len(PhiSubList))], 
                             [PhiSubList[kk][Phi_stateVar2] for kk in range(len(PhiSubList))]]
                if self._SOL_2ndOrdMomDict:
                    Ell_width = [re(cos(N(pi/2)-projection_angle_list[kk])*sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar2**2)]/systemSize.subs(argDict)) + sin(N(pi/2)-projection_angle_list[kk])*sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar1**2)]/systemSize.subs(argDict))) for kk in range(len(SOL_2ndOrdMomDictList))]
                    Ell_height = [re(cos(projection_angle_list[kk])*sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar2**2)]/systemSize.subs(argDict)) + sin(projection_angle_list[kk])*sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar1**2)]/systemSize.subs(argDict))) for kk in range(len(SOL_2ndOrdMomDictList))]
                #FixedPoints=[[realEQsol[kk][self._stateVariable1] for kk in range(len(realEQsol))], 
                #             [realEQsol[kk][self._stateVariable2] for kk in range(len(realEQsol))]]
                #Ell_width = [SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar1**2)] for kk in range(len(SOL_2ndOrdMomDictList))]
                #Ell_height = [SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar2**2)] for kk in range(len(SOL_2ndOrdMomDictList))] 
           
            FixedPoints.append(EVplot)
        
        else:
            FixedPoints=None   
        
        self._get_field2d("2d stream plot", 100) ## @todo: allow user to set mesh points with keyword
        
        if self._speed is not None:
            fig_stream=plt.streamplot(self._X, self._Y, self._Xdot, self._Ydot, color = self._speed, cmap = 'gray') ## @todo: define colormap by user keyword
        else:
            fig_stream=plt.streamplot(self._X, self._Y, self._Xdot, self._Ydot, color = 'k') ## @todo: define colormap by user keyword
        
        if self._SOL_2ndOrdMomDict:
            # swap width and height of ellipse if width > height
            for kk in range(len(Ell_width)):
                ell_width_temp = Ell_width[kk]
                ell_height_temp = Ell_height[kk]
                if ell_width_temp > ell_height_temp:
                    Ell_height[kk] = ell_width_temp
                    Ell_width[kk] = ell_height_temp
                    
            ells = [mpatch.Ellipse(xy=[FixedPoints[0][nn],FixedPoints[1][nn]], width=Ell_width[nn]/systemSize.subs(argDict), height=Ell_height[nn]/systemSize.subs(argDict), angle= round(angle_ell_list[nn],5)) for nn in range(len(FixedPoints[0]))]
            ax = plt.gca()
            for kk in range(len(ells)):
                ax.add_artist(ells[kk])
                ells[kk].set_alpha(0.25)
                if re(EVplot[kk][0]) < 0 and re(EVplot[kk][1]) < 0:
                    Fcolor = line_color_list[1]
                elif re(EVplot[kk][0]) > 0 and re(EVplot[kk][1]) > 0:
                    Fcolor = line_color_list[2]
                else:
                    Fcolor = line_color_list[0]
                ells[kk].set_facecolor(Fcolor)
        if self._mumotModel._constantSystemSize == True:
            plt.fill_between([0,1], [1,0], [1,1], color='grey', alpha='0.25')            
            plt.xlim(0,1)
            plt.ylim(0,1)
        else:
            plt.xlim(0,self._X.max())
            plt.ylim(0,self._Y.max())
        
        _fig_formatting_2D(figure=fig_stream, xlab = self._xlab, specialPoints=FixedPoints, showFixedPoints=self._showFixedPoints, 
                           ax_reformat=False, curve_replot=False, ylab = self._ylab, fontsize=self._chooseFontSize)
#        plt.set_aspect('equal') ## @todo

        if self._showFixedPoints==True or self._SOL_2ndOrdMomDict != None:
            with io.capture_output() as log:
                for kk in range(len(realEQsol)):
                    print('Fixed point'+str(kk+1)+':', realEQsol[kk], 'with eigenvalues:', str(EV[kk]),
                          'and eigenvectors:', str(Evects[kk]))
            self._logs.append(log)



## vector plot view on model
class MuMoTvectorView(MuMoTfieldView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._generatingCommand = "vector"

    def _plot_field(self, _=None):
        super()._plot_field()
        if self._stateVariable3 == None:
            if self._showFixedPoints==True:
                realEQsol, eigList = self._get_fixedPoints2d()
                
                EV = []
                EVplot = []
                for kk in range(len(eigList)):
                    EVsub = []
                    for key in eigList[kk]:
                        if eigList[kk][key][0] >1:
                            for jj in range(eigList[kk][key][0]):
                                EVsub.append(key.evalf())
                        else:
                            EVsub.append(key.evalf())
                            
                    EV.append(EVsub)
                    if self._mumotModel._constantSystemSize == True:
                        if (0 <= re(realEQsol[kk][self._stateVariable1]) <= 1 and 0 <= re(realEQsol[kk][self._stateVariable2]) <= 1):
                            EVplot.append(EVsub)
                    else:
                        EVplot.append(EVsub)

                #EV = []
                #EVplot = []
                #for kk in range(len(eigList)):
                #    EVsub = []
                #    for key in eigList[kk]:
                #        EVsub.append(key.evalf())
                #    EV.append(EVsub)
                #    if (0 <= re(realEQsol[kk][self._stateVariable1]) <= 1 and 0 <= re(realEQsol[kk][self._stateVariable2]) <= 1):
                #        EVplot.append(EVsub)
                
                if self._mumotModel._constantSystemSize == True:
                    FixedPoints=[[realEQsol[kk][self._stateVariable1] for kk in range(len(realEQsol)) if (0 <= re(realEQsol[kk][self._stateVariable1]) <= 1 and 0 <= re(realEQsol[kk][self._stateVariable2]) <= 1)], 
                                 [realEQsol[kk][self._stateVariable2] for kk in range(len(realEQsol)) if (0 <= re(realEQsol[kk][self._stateVariable1]) <= 1 and 0 <= re(realEQsol[kk][self._stateVariable2]) <= 1)]]
                else:
                    FixedPoints=[[realEQsol[kk][self._stateVariable1] for kk in range(len(realEQsol))], 
                                 [realEQsol[kk][self._stateVariable2] for kk in range(len(realEQsol))]]
                FixedPoints.append(EVplot)
            else:
                FixedPoints = None
            
            self._get_field2d("2d vector plot", 10) ## @todo: allow user to set mesh points with keyword
            fig_vector=plt.quiver(self._X, self._Y, self._Xdot, self._Ydot, units='width', color = 'black') ## @todo: define colormap by user keyword

            if self._mumotModel._constantSystemSize == True:
                plt.fill_between([0,1], [1,0], [1,1], color='grey', alpha='0.25')
            else:
                plt.xlim(0,self._X.max())
                plt.ylim(0,self._Y.max())
            _fig_formatting_2D(figure=fig_vector, xlab = self._xlab, specialPoints=FixedPoints, showFixedPoints=self._showFixedPoints, ax_reformat=False, curve_replot=False,
                   ylab = self._ylab, fontsize=self._chooseFontSize)
    #        plt.set_aspect('equal') ## @todo
            if self._showFixedPoints==True:
                with io.capture_output() as log:
                    for kk in range(len(realEQsol)):
                        print('Fixed point'+str(kk+1)+':', realEQsol[kk], 'with eigenvalues:', str(EV[kk]))
                self._logs.append(log)
        
        else:
            if self._showFixedPoints==True:
                realEQsol, eigList = self._get_fixedPoints3d()
                EV = []
                EVplot = []
                for kk in range(len(eigList)):
                    EVsub = []
                    for key in eigList[kk]:
                        if eigList[kk][key][0] >1:
                            for jj in range(eigList[kk][key][0]):
                                EVsub.append(key.evalf())
                        else:
                            EVsub.append(key.evalf())
                            
                    EV.append(EVsub)
                    if self._mumotModel._constantSystemSize == True:
                        if (0 <= re(realEQsol[kk][self._stateVariable1]) <= 1 and 0 <= re(realEQsol[kk][self._stateVariable2]) <= 1 and 0 <= re(realEQsol[kk][self._stateVariable3]) <= 1):
                            EVplot.append(EVsub)
                    else:
                        EVplot.append(EVsub)
                    
                if self._mumotModel._constantSystemSize == True:
                    FixedPoints=[[realEQsol[kk][self._stateVariable1] for kk in range(len(realEQsol)) if (0 <= re(realEQsol[kk][self._stateVariable1]) <= 1) and (0 <= re(realEQsol[kk][self._stateVariable2]) <= 1) and (0 <= re(realEQsol[kk][self._stateVariable3]) <= 1)], 
                                 [realEQsol[kk][self._stateVariable2] for kk in range(len(realEQsol)) if (0 <= re(realEQsol[kk][self._stateVariable1]) <= 1) and (0 <= re(realEQsol[kk][self._stateVariable2]) <= 1) and (0 <= re(realEQsol[kk][self._stateVariable3]) <= 1)],
                                 [realEQsol[kk][self._stateVariable3] for kk in range(len(realEQsol)) if (0 <= re(realEQsol[kk][self._stateVariable1]) <= 1) and (0 <= re(realEQsol[kk][self._stateVariable2]) <= 1) and (0 <= re(realEQsol[kk][self._stateVariable3]) <= 1)]]
                else:
                    FixedPoints=[[realEQsol[kk][self._stateVariable1] for kk in range(len(realEQsol))], 
                                 [realEQsol[kk][self._stateVariable2] for kk in range(len(realEQsol))],
                                 [realEQsol[kk][self._stateVariable3] for kk in range(len(realEQsol))]]
                FixedPoints.append(EVplot)
            else:
                FixedPoints = None
                
            self._get_field3d("3d vector plot", 10)
            ax = self._figure.gca(projection = '3d')
            fig_vec3d=ax.quiver(self._X, self._Y, self._Z, self._Xdot, self._Ydot, self._Zdot, length = 0.01, color = 'black') ## @todo: define colormap by user keyword; normalise off maximum value in self._speed, and meshpoints?
            _fig_formatting_3D(figure=fig_vec3d, xlab= self._xlab, ylab= self._ylab, zlab= self._zlab, specialPoints=FixedPoints,
                               showFixedPoints=self._showFixedPoints, ax_reformat=True, showPlane=self._mumotModel._constantSystemSize)
#           plt.set_aspect('equal') ## @todo

            if self._showFixedPoints==True:
                with io.capture_output() as log:
                    for kk in range(len(realEQsol)):
                        print('Fixed point'+str(kk+1)+':', realEQsol[kk], 'with eigenvalues:', str(EV[kk]))
                self._logs.append(log)


## bifurcation view on model 
class MuMoTbifurcationView(MuMoTview):
    ## model for bifurcation analysis
    _pyDSmodel = None
    ## critical parameter for bifurcation analysis
    _bifurcationParameter = None
    ## first state variable of 2D system
    _stateVariable1 = None
    ## second state variable of 2D system
    _stateVariable2 = None
    ## state variable of 2D system used for bifurcation analysis, can either be _stateVariable1 or _stateVariable2
    _stateVarBif1 = None
    ## state variable of 2D system used for bifurcation analysis, can either be _stateVariable1 or _stateVariable2
    _stateVarBif2 = None
    
    ## generates command for bookmark functionality
    _generatingCommand = None
    ## fontsize on figure axes
    _chooseFontSize = None
    ## label for vertical axis
    _LabelY = None
    ## label for horizontal axis
    _LabelX = None
    ## displayed range for vertical axis
    _chooseXrange = None
    ## displayed range for horizontal axis
    _chooseYrange = None
    ## maximum number of points in one continuation calculation 
    _MaxNumPoints = None
    ## information about the mathematical expression displayed on vertical axis; can be 'None', '+' or '-'
    _SVoperation = None
    ## initial conditions specified on corresponding sliders, will be used when calculation of fixed points fails
    _pyDSmodel_ics = None
    
    
    #_bifInit = None   
    # Plotting method to use
    #_plottingMethod = None
    
    def __init__(self, model, controller, bifurcationParameter, stateVarExpr1, stateVarExpr2 = None, 
                 figure = None, params = None, **kwargs):
        
        super().__init__(model, controller, figure, params, **kwargs)
        self._generatingCommand = "bifurcation"

        self._chooseFontSize = kwargs.get('fontsize', None)
        self._LabelY =  kwargs.get('ylab', r'$' + stateVarExpr1 +'$') 
        self._LabelX = kwargs.get('xlab', r'$' + bifurcationParameter +'$')
        self._chooseXrange = kwargs.get('choose_xrange', None)
        self._chooseYrange = kwargs.get('choose_yrange', None)
        
        self._MaxNumPoints = kwargs.get('ContMaxNumPoints',100)
        
        self._bifurcationParameter = self._pydstoolify(bifurcationParameter)
        self._stateVarExpr1 = stateVarExpr1
        stateVarExpr1 = self._pydstoolify(stateVarExpr1)
        if stateVarExpr2:
            stateVarExpr2 = self._pydstoolify(stateVarExpr2)
        
        self._SVoperation = None
        try:
            stateVarExpr1.index('-')
            self._stateVarBif1 = stateVarExpr1[:stateVarExpr1.index('-')]
            self._stateVarBif2 = stateVarExpr1[stateVarExpr1.index('-')+1:]
            self._SVoperation = '-'
            
        except ValueError:
            try:
                stateVarExpr1.index('+')
                self._stateVarBif1 = stateVarExpr1[:stateVarExpr1.index('+')]
                self._stateVarBif2 = stateVarExpr1[stateVarExpr1.index('+')+1:] 
                self._SVoperation = '+'
            except ValueError:
                self._stateVarBif1 = stateVarExpr1
                self._stateVarBif2 = stateVarExpr2
                
        #self._bifInit = kwargs.get('BifParInit', None)
                  
        self._pyDSmodel = dst.args(name = 'MuMoT Model' + str(id(self)))
        #self._bifurcationParameter = bifurcationParameter
        varspecs = {}
        stateVariableList = []
        for reactant in self._mumotModel._reactants:
            if reactant not in self._mumotModel._constantReactants:
                stateVariableList.append(reactant)
                varspecs[self._pydstoolify(reactant)] = self._pydstoolify(self._mumotModel._equations[reactant])
        self._pyDSmodel.varspecs = varspecs
        if len(stateVariableList) != 2:
            self._showErrorMessage('Bifurcation diagrams are currently only supported for 2D systems (2 time-dependent variables in the ODE system)!')
            return None
        self._stateVariable1 = stateVariableList[0]
        self._stateVariable2 = stateVariableList[1]
        if self._stateVarBif2 == None:
            if self._stateVarBif1 == self._pydstoolify(self._stateVariable1):
                self._stateVarBif2 = self._pydstoolify(self._stateVariable2)
            elif self._stateVarBif1 == self._pydstoolify(self._stateVariable2):
                self._stateVarBif2 = self._pydstoolify(self._stateVariable1)
        with io.capture_output() as log:
            print('Bifurcation diagram with state variables: ', self._stateVarBif1, 'and ', self._stateVarBif2, '.')
            print('The bifurcation parameter chosen is: ', self._bifurcationParameter, '.')
        self._logs.append(log)
        
        #self._plottingMethod = kwargs.get('plottingMethod', 'mumot')
        self._plot_bifurcation()     

    def _plot_bifurcation(self, _=None):
        if not(self._silent): ## @todo is this necessary?
            #plt.close()
            #plt.clf()
            #plt.gcf()
            plt.figure(self._figureNum)
            plt.clf()
            self._resetErrorMessage()
        #self._showErrorMessage(str(self))
        #self._resetErrorMessage()
        #plt.figure(self._figureNum)
        argDict = self._get_argDict()
        paramDict = {}
        for arg in argDict:
            if self._pydstoolify(arg) == self._bifurcationParameter+'init':
                paramDict[self._bifurcationParameter] = argDict[arg]
            else:
                if 'Phi0' not in self._pydstoolify(arg):
                    paramDict[self._pydstoolify(arg)] = argDict[arg]
        #print(self._getBifParInitCondFromSlider())
        #print(self._bifurcationParameter)
        #print(paramDict)
        
        with io.capture_output() as log:
            self._pyDSmodel.pars = paramDict #len(self._mumotModel._reactants)}
            
            XDATA = [] # list of arrays containing the bifurcation-parameter data for bifurcation diagram data 
            YDATA = [] # list of arrays containing the state variable data (either one variable, or the sum or difference of the two SVs) for bifurcation diagram data
            
            initDictList = []
            self._pyDSmodel_ics   = self._getInitCondsFromSlider()
            for ic in self._pyDSmodel_ics:
                if 'Phi0' in self._pydstoolify(ic):
                    self._pyDSmodel_ics[self._pydstoolify(ic)[self._pydstoolify(ic).index('0')+1:]] = self._pyDSmodel_ics.pop(ic)  #{'A': 0.1, 'B': 0.9 }  
                
            realEQsol, eigList = self._get_fixedPoints2d()
            if realEQsol != [] and realEQsol != None:
                for kk in range(len(realEQsol)):
                    if all(sign(re(lam)) < 0 for lam in eigList[kk]) == True:
                        initDictList.append(realEQsol[kk])
                print(len(initDictList), 'stable steady state(s) detected and continuated. Initial conditions for state variables specified on sliders were not used.')
            else:
                initDictList.append(self._pyDSmodel_ics)
                print('Stationary states could not be calculated; used initial conditions specified on sliders instead: ', self._pyDSmodel_ics, '. This means only one branch was continuated and the starting point might not have been a stationary state.')   
            #print(initDictList)                                   
            #if self._plottingMethod.lower() == 'mumot':
            specialPoints=[]  # list of special points: LP and BP
            sPoints_X=[] #bifurcation parameter
            sPoints_Y=[] #stateVarBif1
            sPoints_Labels=[]
            EIGENVALUES = []
            sPoints_Z=[] #stateVarBif2
            k_iter_BPlabel = 0
            k_iter_LPlabel = 0
            
            for nn in range(len(initDictList)):
# # The following few lines that are commented out prevent continuation of branches where the initial condition
# # will lead to a copy of a curve that has already been computed.                
#                 skip_init = False
#                 if nn > 0:
#                     for kk in range(nn):
#                         skip_init_count = 0
#                         for item1 in initDictList[nn]:
#                             for item2 in initDictList[kk]:
#                                 if item1 != item2:
#                                     if round(initDictList[nn][item1], 8) == round(initDictList[kk][item2], 8):
#                                         skip_init_count += 1
#                             if skip_init_count == len(initDictList[nn]):
#                                 skip_init = True
#                 print(skip_init)        
#                         
#                 if skip_init == False:      
                 
                #self._pyDSmodel.ics = initDictList[nn]
                pyDSode = dst.Generator.Vode_ODEsystem(self._pyDSmodel)  
                pyDSode.set(ics =  initDictList[nn] ) 
                pyDSode.set(pars = self._getBifParInitCondFromSlider())      
                pyDScont = dst.ContClass(pyDSode)
                EQ_iter = 1+nn
                k_iter_BP = 1
                k_iter_LP = 1
                pyDScontArgs = dst.args(name='EQ'+str(EQ_iter), type='EP-C')     # 'EP-C' stands for Equilibrium Point Curve. The branch will be labelled with the string aftr name='name'.
                pyDScontArgs.freepars     = [self._bifurcationParameter]   # control parameter(s) (should be among those specified in self._pyDSmodel.pars)
                pyDScontArgs.MaxNumPoints = self._MaxNumPoints    # The following 3 parameters should work for most cases, as there should be a step-size adaption within PyDSTool.
                pyDScontArgs.MaxStepSize  = 1e-1
                pyDScontArgs.MinStepSize  = 1e-5
                pyDScontArgs.StepSize     = 2e-3
                pyDScontArgs.LocBifPoints = ['LP', 'BP']       # 'Limit Points' and 'Branch Points may be detected'
                pyDScontArgs.SaveEigen    = True            # to tell unstable from stable branches
                
                pyDScont.newCurve(pyDScontArgs)   
                
                try:
                    try:
                        pyDScont['EQ'+str(EQ_iter)].backward()
                    except:
                        self._showErrorMessage('Continuation failure (backward) on initial branch<br>')
                    try:
                        pyDScont['EQ'+str(EQ_iter)].forward()              
                    except:
                        self._showErrorMessage('Continuation failure (forward) on initial branch<br>')
                except ZeroDivisionError:
                    self._showErrorMessage('Division by zero<br>')  
                #pyDScont['EQ'+str(EQ_iter)].info()
                if self._stateVarBif2 != None:
                    try:
                        XDATA.append(pyDScont['EQ'+str(EQ_iter)].sol[self._bifurcationParameter])
                        if self._SVoperation:
                            if self._SVoperation == '-':
                                YDATA.append(pyDScont['EQ'+str(EQ_iter)].sol[self._stateVarBif1] -
                                             pyDScont['EQ'+str(EQ_iter)].sol[self._stateVarBif2])
                            elif self._SVoperation == '+':
                                YDATA.append(pyDScont['EQ'+str(EQ_iter)].sol[self._stateVarBif1] +
                                             pyDScont['EQ'+str(EQ_iter)].sol[self._stateVarBif2])  
                            else:
                                self._showErrorMessage('Only \' +\' and \'-\' are supported operations between state variables.')  
                        else:
                            YDATA.append(pyDScont['EQ'+str(EQ_iter)].sol[self._stateVarBif1])
                        
                        EIGENVALUES.append(np.array([pyDScont['EQ'+str(EQ_iter)].sol[kk].labels['EP']['data'].evals 
                                                     for kk in range(len(pyDScont['EQ'+str(EQ_iter)].sol[self._stateVarBif1]))]))
                        
                        while pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('LP'+str(k_iter_LP)):
                            if (round(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('LP'+str(k_iter_LP))[self._bifurcationParameter], 4) not in [round(kk ,4) for kk in sPoints_X] 
                                and round(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('LP'+str(k_iter_LP))[self._stateVarBif1], 4) not in [round(kk ,4) for kk in sPoints_Y]
                                and round(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('LP'+str(k_iter_LP))[self._stateVarBif2], 4) not in [round(kk ,4) for kk in sPoints_Z]):
                                sPoints_X.append(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('LP'+str(k_iter_LP))[self._bifurcationParameter])
                                sPoints_Y.append(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('LP'+str(k_iter_LP))[self._stateVarBif1])
                                sPoints_Z.append(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('LP'+str(k_iter_LP))[self._stateVarBif2])
                                k_iter_LPlabel += 1
                                sPoints_Labels.append('LP'+str(k_iter_LPlabel))
                            k_iter_LP+=1
                        
                        
                        k_iter_BPlabel_previous = k_iter_BPlabel
                        while pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('BP'+str(k_iter_BP)):
                            if (round(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('BP'+str(k_iter_BP))[self._bifurcationParameter], 4) not in [round(kk ,4) for kk in sPoints_X] 
                                and round(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('BP'+str(k_iter_BP))[self._stateVarBif1], 4) not in [round(kk ,4) for kk in sPoints_Y]
                                and round(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('BP'+str(k_iter_BP))[self._stateVarBif2], 4) not in [round(kk ,4) for kk in sPoints_Z]):
                                sPoints_X.append(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('BP'+str(k_iter_BP))[self._bifurcationParameter])
                                sPoints_Y.append(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('BP'+str(k_iter_BP))[self._stateVarBif1])
                                sPoints_Z.append(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('BP'+str(k_iter_BP))[self._stateVarBif2])
                                k_iter_BPlabel += 1
                                sPoints_Labels.append('BP'+str(k_iter_BPlabel))
                            k_iter_BP+=1
                        for jj in range(1,k_iter_BP):
                            if 'BP'+str(jj+k_iter_BPlabel_previous) in sPoints_Labels:
                                EQ_iter_BP = jj
                                print(EQ_iter_BP)
                                k_iter_next = 1
                                pyDScontArgs = dst.args(name='EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP), type='EP-C')     # 'EP-C' stands for Equilibrium Point Curve. The branch will be labelled with the string aftr name='name'.
                                pyDScontArgs.freepars     = [self._bifurcationParameter]   # control parameter(s) (should be among those specified in self._pyDSmodel.pars)
                                pyDScontArgs.MaxNumPoints = self._MaxNumPoints    # The following 3 parameters should work for most cases, as there should be a step-size adaption within PyDSTool.
                                pyDScontArgs.MaxStepSize  = 1e-1
                                pyDScontArgs.MinStepSize  = 1e-5
                                pyDScontArgs.StepSize     = 5e-3
                                pyDScontArgs.LocBifPoints = ['LP', 'BP']        # 'Limit Points' and 'Branch Points may be detected'
                                pyDScontArgs.SaveEigen    = True             # to tell unstable from stable branches
                                pyDScontArgs.initpoint = 'EQ'+str(EQ_iter)+':BP'+str(jj)
                                pyDScont.newCurve(pyDScontArgs)
                                
                                try:
                                    try:
                                        pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].backward()
                                    except:
                                        self._showErrorMessage('Continuation failure (backward) starting from branch point<br>')
                                    try:
                                        pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].forward()              
                                    except:
                                        self._showErrorMessage('Continuation failure (forward) starting from branch point<br>')
                                except ZeroDivisionError:
                                    self._showErrorMessage('Division by zero<br>')
                                
                                XDATA.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].sol[self._bifurcationParameter])
                                if self._SVoperation:
                                    if self._SVoperation == '-':
                                        YDATA.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].sol[self._stateVarBif1] -
                                                     pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].sol[self._stateVarBif2])
                                    elif self._SVoperation == '+':
                                        YDATA.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].sol[self._stateVarBif1] +
                                                     pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].sol[self._stateVarBif2])
                                    else:
                                        self._showErrorMessage('Only \' +\' and \'-\' are supported operations between state variables.')
                                else:
                                    YDATA.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].sol[self._stateVarBif1])
                                    
                                EIGENVALUES.append(np.array([pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].sol[kk].labels['EP']['data'].evals 
                                                             for kk in range(len(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].sol[self._stateVarBif1]))]))
                                while pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('BP'+str(k_iter_next)):
                                    if (round(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('BP'+str(k_iter_next))[self._bifurcationParameter], 4) not in [round(kk ,4) for kk in sPoints_X] 
                                        and round(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('BP'+str(k_iter_next))[self._stateVarBif1], 4) not in [round(kk ,4) for kk in sPoints_Y]
                                        and round(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('BP'+str(k_iter_next))[self._stateVarBif2], 4) not in [round(kk ,4) for kk in sPoints_Z]):    
                                        sPoints_X.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('BP'+str(k_iter_next))[self._bifurcationParameter])
                                        sPoints_Y.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('BP'+str(k_iter_next))[self._stateVarBif1])
                                        sPoints_Z.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('BP'+str(k_iter_next))[self._stateVarBif2])
                                        sPoints_Labels.append('EQ_BP_BP'+str(k_iter_next))
                                    k_iter_next += 1
                                k_iter_next = 1
                                while pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('LP'+str(k_iter_next)):
                                    if (round(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('LP'+str(k_iter_next))[self._bifurcationParameter], 4) not in [round(kk ,4) for kk in sPoints_X] 
                                        and round(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('LP'+str(k_iter_next))[self._stateVarBif1], 4) not in [round(kk ,4) for kk in sPoints_Y]
                                        and round(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('LP'+str(k_iter_next))[self._stateVarBif2], 4) not in [round(kk ,4) for kk in sPoints_Z]):
                                        sPoints_X.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('LP'+str(k_iter_next))[self._bifurcationParameter])
                                        sPoints_Y.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('LP'+str(k_iter_next))[self._stateVarBif1])
                                        sPoints_Z.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('LP'+str(k_iter_next))[self._stateVarBif2])
                                        sPoints_Labels.append('EQ_BP_LP'+str(k_iter_next))
                                    k_iter_next += 1
    
                    except TypeError:
                        print('Continuation failed; try with different parameters - use sliders. If that does not work, try changing maximum number of continuation points using the keyword ContMaxNumPoints. If not set, default value is ContMaxNumPoints=100.')       
                
                
                del(pyDScontArgs)
                del(pyDScont)
                del(pyDSode)
            if self._SVoperation:
                if self._SVoperation == '-':    
                    specialPoints=[sPoints_X, np.asarray(sPoints_Y)-np.asarray(sPoints_Z), sPoints_Labels]
                elif self._SVoperation == '+':    
                    specialPoints=[sPoints_X, np.asarray(sPoints_Y)+np.asarray(sPoints_Z), sPoints_Labels]
                else:
                    self._showErrorMessage('Only \' +\' and \'-\' are supported operations between state variables.')
            else:
                specialPoints=[sPoints_X, np.asarray(sPoints_Y), sPoints_Labels]
                
            #print('Special Points on curve: ', specialPoints)
            print('Special Points on curve:')
            for kk in range(len(specialPoints[0])):
                print(specialPoints[2][kk], ': ', self._bifurcationParameter, '=', str(round(specialPoints[0][kk], 3)), ',',
                      self._stateVarExpr1, '=', str(round(specialPoints[1][kk], 3)) )
            
            if self._chooseYrange == None:
                if self._mumotModel._systemSize:
                    self._chooseYrange = [-self._getSystemSize(), self._getSystemSize()]
                          
            if XDATA != [] and self._chooseXrange == None:
                xmaxbif = np.max([np.max(XDATA[kk]) for kk in range(len(XDATA))])
                self._chooseXrange = [0, xmaxbif]
                
            if XDATA !=[] and YDATA != []:
                #plt.clf()
                _fig_formatting_2D(xdata=XDATA, 
                                ydata=YDATA,
                                xlab = self._LabelX, 
                                ylab = self._LabelY,
                                specialPoints=specialPoints, 
                                eigenvalues=EIGENVALUES, 
                                choose_xrange=self._chooseXrange, choose_yrange=self._chooseYrange,
                                ax_reformat=False, curve_replot=False, fontsize=self._chooseFontSize)
            else:
                self._showErrorMessage('Bifurcation diagram could not be computed. Try changing parameter values on the sliders')
                return None

        self._logs.append(log)

    ## utility function to mangle variable names in equations so they are accepted by PyDStool
    def _pydstoolify(self, equation):
        equation = str(equation)
        equation = equation.replace('{', '')
        equation = equation.replace('}', '')
        equation = equation.replace('_', '')
        equation = equation.replace('\\', '')
        equation = equation.replace('^', '')
        
        return equation
    
    
    def _build_bookmark(self, includeParams = True):
        if not self._silent:
            logStr = "bookmark = "
        else:
            logStr = ""
        logStr += "<modelName>." + self._generatingCommand + "('" + str(self._bifurcationParameter) + "', '" + str(self._stateVarExpr1) +"', "
        #if self._stateVarBif2 != None:
        #    logStr += "'" + str(self._stateVarBif2) + "', "
        if includeParams:
            logStr += self._get_bookmarks_params() + ", "
        if len(self._generatingKwargs) > 0:
            for key in self._generatingKwargs:
                if key == 'initCondsSV':
                    logStr += key + " = " + str(self._pyDSmodel_ics) + ", "
                else:
                    if key != 'chooseBifParam':
                        logStr += key + " = " + str(self._generatingKwargs[key]) + ", "
        logStr += "bookmark = False"
        logStr += ")"
        logStr = logStr.replace('\\', '\\\\')
        
        return logStr    

    ## gets initial value of bifurcation parameter from corresponding slider. This determines the staring points for the numerical continuation.
    def _getBifParInitCondFromSlider(self):
        #if self._controller != None:
        
        if self._bifurcationParameter[0]=='\\':
            bifPar = self._bifurcationParameter[1:]
        elif self._bifurcationParameter[0]=='\\' and self._bifurcationParameter[1]=='\\':
            bifPar = self._bifurcationParameter[2:]
        else:
            bifPar = self._bifurcationParameter
            
        paramName = []
        paramValue = []
        for name, value in self._controller._widgetsFreeParams.items():
            if name == latex(Symbol(str(bifPar)+'_init')):
                #paramName.append(self._pydstoolify(self._bifurcationParameter))
                paramName.append(self._bifurcationParameter)
                paramValue.append(value.value)
                
        argNameSymb = list(map(Symbol, paramName))
        argDict = dict(zip(argNameSymb, paramValue))
        
        return argDict

# 
# ## bifurcation view on model 
# class MuMoTbifurcationView(MuMoTview):
#     _pyDSmodel = None
#     _bifurcationParameter = None
#     _stateVariable1 = None
#     _stateVariable2 = None
#     
#     ## Plotting method to use
#     _plottingMethod = None
#     
#     def __init__(self, model, controller, bifurcationParameter, stateVariable1, stateVariable2 = None, 
#                  figure = None, params = None, **kwargs):
#         super().__init__(model, controller, figure, params, **kwargs)
#         self._generatingCommand = "bifurcation"
#         bifurcationParameter = self._pydstoolify(bifurcationParameter)
#         stateVariable1 = self._pydstoolify(stateVariable1)
#         stateVariable2 = self._pydstoolify(stateVariable2)
# 
#         paramDict = {}
#         initialRateValue = INITIAL_RATE_VALUE ## @todo was 1 (choose initial values sensibly)
#         rateLimits = (0, RATE_BOUND) ## @todo choose limit values sensibly
#         rateStep = RATE_STEP ## @todo choose rate step sensibly
#         for rate in self._mumotModel._rates:
#             paramDict[self._pydstoolify(rate)] = initialRateValue ## @todo choose initial values sensibly
#         paramDict[self._pydstoolify(self._mumotModel._systemSize)] = 1 ## @todo choose system size sensibly
#         
#         if 'fontsize' in kwargs:
#             self._chooseFontSize = kwargs['fontsize']
#         else:
#             self._chooseFontSize=None
#             
#         if 'xlab' in kwargs:
#             self._xlab = kwargs['xlab']
#         else:
#             self._xlab=None
#             
#         if 'ylab' in kwargs:
#             self._ylab = kwargs['ylab']
#         else:
#             self._ylab=None
#             
#         self._bifInit = kwargs.get('BifParInit', 5)
#         
#         tmpInitSV = kwargs.get('initSV', [])
#         self._initSV= []
#         if tmpInitSV != []:
#             assert (len(tmpInitSV) == len(self._mumotModel._reactants)),"Number of state variables and initial conditions must coincide!"
#             for pair in tmpInitSV:
#                 self._initSV.append([self._pydstoolify(pair[0]), pair[1]])
#         else:
#             if kwargs.get('initRandom', False) == True:
#                 for reactant in self._mumotModel._reactants:
#                     self._initSV.append([self._pydstoolify(reactant), round(np.random.rand(), 2)])
#             else:
#                 for reactant in self._mumotModel._reactants:
#                     self._initSV.append([self._pydstoolify(reactant), 0.0])
#                 
#         with io.capture_output() as log:      
#             name = 'MuMoT Model' + str(id(self))
#             self._pyDSmodel = dst.args(name = name)
#             self._pyDSmodel.pars = paramDict
#             varspecs = {}
#             for reactant in self._mumotModel._reactants:
#                 varspecs[self._pydstoolify(reactant)] = self._pydstoolify(self._mumotModel._equations[reactant])
#             self._pyDSmodel.varspecs = varspecs
#     
#             if model._systemSize != None:
#                 ## @todo: shouldn't allow system size to be varied?
#                 pass
#     #                self._paramValues.append(1)
#     #                self._paramNames.append(str(self._systemSize))
#     #                widget = widgets.FloatSlider(value = 1, min = _rateLimits[0], max = _rateLimits[1], step = _rateStep, description = str(self._systemSize), continuous_update = False)
#     #                widget.on_trait_change(self._replot_bifurcation2D, 'value')
#     #                display(widget)
#             else:
#                 print('Cannot attempt bifurcation plot until system size is set, using substitute()')
#                 return
#             
#             if stateVariable2 != None:
#                 ## 3-d bifurcation diagram (@todo: currently unsupported)
#                 assert True #was 'false' before. @todo: Specify assertion rule.
#                 
#             # Prepare the system to start close to a steady state
#             self._bifurcationParameter = bifurcationParameter
#             self._LabelX = self._bifurcationParameter if self._xlab == None else self._xlab
#             try:
#                 stateVariable1.index('-')
#                 self._stateVariable1 = stateVariable1[:stateVariable1.index('-')]
#                 self._stateVariable2 = stateVariable1[stateVariable1.index('-')+1:]
#                 self._LabelY = self._stateVariable1+'-'+self._stateVariable2 if self._ylab == None else self._ylab
#             except ValueError:
#                 self._stateVariable1 = stateVariable1
#                 self._stateVariable2 = stateVariable2
#                 self._LabelY = self._stateVariable1 if self._ylab == None else self._ylab
#     #            self._pyDSode.set(pars = {bifurcationParameter: 0} )       ## Lower bound of the bifurcation parameter (@todo: set dynamically)
#     #            self._pyDSode.set(pars = self._pyDSmodel.pars )       ## Lower bound of the bifurcation parameter (@todo: set dynamically)
#     #            self._pyDSode.pars = {bifurcationParameter: 0}             ## Lower bound of the bifurcation parameter (@todo: set dynamically?)
#             initconds = {stateVariable1: self._pyDSmodel.pars[self._pydstoolify(self._mumotModel._systemSize)] / len(self._mumotModel._reactants)} ## @todo: guess where steady states are?
#             
#             self._pyDSmodel.ics   = {}
#             for kk in range(len(self._initSV)):
#                 self._pyDSmodel.ics[self._initSV[kk][0]] = self._initSV[kk][1]  #{'A': 0.1, 'B': 0.9 }  
#             print('Initial conditions chosen for state variables: ',self._pyDSmodel.ics)   
#     #            self._pyDSode.set(ics = initconds)
#             self._pyDSode = dst.Generator.Vode_ODEsystem(self._pyDSmodel)  
#             self._pyDSode.set(pars = {bifurcationParameter: self._bifInit})      ## @@todo remove magic number
#             self._pyDScont = dst.ContClass(self._pyDSode)              # Set up continuation class 
#             ## @todo: add self._pyDScontArgs to __init__()
#             self._pyDScontArgs = dst.args(name='EQ1', type='EP-C')     # 'EP-C' stands for Equilibrium Point Curve. The branch will be labeled 'EQ1'.
#             self._pyDScontArgs.freepars     = [bifurcationParameter]   # control parameter(s) (should be among those specified in self._pyDSmodel.pars)
#             self._pyDScontArgs.MaxNumPoints = kwargs.get('ContMaxNumPoints',450)    ## The following 3 parameters are set after trial-and-error @todo: how to automate this?
#             self._pyDScontArgs.MaxStepSize  = 1e-1
#             self._pyDScontArgs.MinStepSize  = 1e-5
#             self._pyDScontArgs.StepSize     = 2e-3
#             self._pyDScontArgs.LocBifPoints = ['LP', 'BP']        ## @todo WAS 'LP' (detect limit points / saddle-node bifurcations)
#             self._pyDScontArgs.SaveEigen    = True             # to tell unstable from stable branches
# #            self._pyDScontArgs.CalcStab     = True
#         self._logs.append(log)
#         
# #            self._bifurcation2Dfig = plt.figure(1)                    
# 
#         #if kwargs != None:
#         #    self._plottingMethod = kwargs.get('plottingMethod', 'pyDS')
#         #else:
#         #    self._plottingMethod = 'pyDS'
#         
#         self._plottingMethod = kwargs.get('plottingMethod', 'mumot')
#         self._plot_bifurcation()
#             
# 
#     def _plot_bifurcation(self):
#         self._resetErrorMessage()
#         with io.capture_output() as log:
#             self._log("bifurcation analysis")
#             self._pyDScont.newCurve(self._pyDScontArgs)
#             try:
#                 try:
#                     self._pyDScont['EQ1'].backward()
#                 except:
#                     self._showErrorMessage('Continuation failure (backward)<br>')
#                 try:
#                     self._pyDScont['EQ1'].forward()              ## @todo: how to choose direction?
#                 except:
#                     self._showErrorMessage('Continuation failure (forward)<br>')
#             except ZeroDivisionError:
#                 self._showErrorMessage('Division by zero<br>')                
#     #            self._pyDScont['EQ1'].info()
#         
#             if self._plottingMethod.lower() == 'mumot':
#                 ## use internal plotting routines: now supported!   
#                 #self._stateVariable2 == None:
#                 # 2-d bifurcation diagram
#                 self._specialPoints=[]  #todo: modify to include several equations not only EQ1
#                 k_iter=1
#                 self.sPoints_X=[] #bifurcation parameter
#                 self.sPoints_Y=[] #state variable 1
#                 self.sPoints_Labels=[]
#                 
#                 #while self._pyDScont['EQ1'].getSpecialPoint('BP'+str(k_iter)):
#                 #    self.sPoints_X.append(self._pyDScont['EQ1'].getSpecialPoint('BP'+str(k_iter))[2])
#                 #    self.sPoints_Y.append(self._pyDScont['EQ1'].getSpecialPoint('BP'+str(k_iter))[0])
#                 #    self.sPoints_Z.append(self._pyDScont['EQ1'].getSpecialPoint('BP'+str(k_iter))[1])
#                 #    self.sPoints_Labels.append('BP'+str(k_iter))
#                 #    k_iter+=1
#                 #k_iter=1
#                 #while self._pyDScont['EQ1'].getSpecialPoint('LP'+str(k_iter)):
#                 #    self.sPoints_X.append(self._pyDScont['EQ1'].getSpecialPoint('LP'+str(k_iter))[2])
#                 #    self.sPoints_Y.append(self._pyDScont['EQ1'].getSpecialPoint('LP'+str(k_iter))[0])
#                 #    self.sPoints_Z.append(self._pyDScont['EQ1'].getSpecialPoint('LP'+str(k_iter))[1])
#                 #    self.sPoints_Labels.append('LP'+str(k_iter))
#                 #    k_iter+=1
#     
#                 
#                 
#                 if self._stateVariable2 != None:
#                     try:
#                         self.sPoints_Z=[] #state variable 2
#                         self._YDATA = (self._pyDScont['EQ1'].sol[self._stateVariable1] -
#                                   self._pyDScont['EQ1'].sol[self._stateVariable2])
#                         while self._pyDScont['EQ1'].getSpecialPoint('BP'+str(k_iter)):
#                             self.sPoints_X.append(self._pyDScont['EQ1'].getSpecialPoint('BP'+str(k_iter))[self._bifurcationParameter])
#                             self.sPoints_Y.append(self._pyDScont['EQ1'].getSpecialPoint('BP'+str(k_iter))[self._stateVariable1])
#                             self.sPoints_Z.append(self._pyDScont['EQ1'].getSpecialPoint('BP'+str(k_iter))[self._stateVariable2])
#                             self.sPoints_Labels.append('BP'+str(k_iter))
#                             k_iter+=1
#                         k_iter=1
#                         while self._pyDScont['EQ1'].getSpecialPoint('LP'+str(k_iter)):
#                             self.sPoints_X.append(self._pyDScont['EQ1'].getSpecialPoint('LP'+str(k_iter))[self._bifurcationParameter])
#                             self.sPoints_Y.append(self._pyDScont['EQ1'].getSpecialPoint('LP'+str(k_iter))[self._stateVariable1])
#                             self.sPoints_Z.append(self._pyDScont['EQ1'].getSpecialPoint('LP'+str(k_iter))[self._stateVariable2])
#                             self.sPoints_Labels.append('LP'+str(k_iter))
#                             k_iter+=1
#                         self._specialPoints=[self.sPoints_X, 
#                                              np.asarray(self.sPoints_Y)-np.asarray(self.sPoints_Z), 
#                                              self.sPoints_Labels]
#                     except TypeError:
#                         print('Continuation failed; try with different parameters.')
#                     
#                 else:
#                     try:
#                         self._YDATA = self._pyDScont['EQ1'].sol[self._stateVariable1]
#                         while self._pyDScont['EQ1'].getSpecialPoint('BP'+str(k_iter)):
#                             self.sPoints_X.append(self._pyDScont['EQ1'].getSpecialPoint('BP'+str(k_iter))[self._bifurcationParameter])
#                             self.sPoints_Y.append(self._pyDScont['EQ1'].getSpecialPoint('BP'+str(k_iter))[self._stateVariable1])
#                             self.sPoints_Labels.append('BP'+str(k_iter))
#                             k_iter+=1
#                         k_iter=1
#                         while self._pyDScont['EQ1'].getSpecialPoint('LP'+str(k_iter)):
#                             self.sPoints_X.append(self._pyDScont['EQ1'].getSpecialPoint('LP'+str(k_iter))[self._bifurcationParameter])
#                             self.sPoints_Y.append(self._pyDScont['EQ1'].getSpecialPoint('LP'+str(k_iter))[self._stateVariable1])
#                             self.sPoints_Labels.append('LP'+str(k_iter))
#                             k_iter+=1
#                         
#                         self._specialPoints=[self.sPoints_X, self.sPoints_Y, self.sPoints_Labels]    
#                     except TypeError:
#                         print('Continuation failed; try with different parameters.')
#                 
#                 #The following was an attempt to include also EQ2 if a BP or LP was found using EQ1    
#                 #if self._pyDScont['EQ1'].getSpecialPoint('BP1'):
#                 #    self._pyDSode.set(pars = {self._bifurcationParameter: 5} )
#                 #    self._pyDScontArgs.freepars     = [self._bifurcationParameter]
#                 #    self._pyDScontArgs = dst.args(name='EQ2', type='EP-C')
#                 #    self._pyDScontArgs.initpoint    = 'EQ1:BP1'
#                 #    
#                 #   self._pyDScont.newCurve(self._pyDScontArgs)
#                 #   try:
#                 #        try:
#                 #            self._pyDScont['EQ2'].backward()
#                 #        except:
#                 #            self._showErrorMessage('Continuation failure (backward)<br>')
#                 #        try:
#                 #            self._pyDScont['EQ2'].forward()
#                 #        except:
#                 #            self._showErrorMessage('Continuation failure (forward)<br>')
#                 #    except ZeroDivisionError:
#                 #        self._showErrorMessage('Division by zero<br>')  
#                 
#                 
#                 #    self.sPoints_X.append(self._pyDScont['EQ1'].getSpecialPoint('BP'+str(k_iter))[2])
#                 #    self.sPoints_Y.append(self._pyDScont['EQ1'].getSpecialPoint('BP'+str(k_iter))[0])
#                 #    self.sPoints_Z.append(self._pyDScont['EQ1'].getSpecialPoint('BP'+str(k_iter))[1])
#                 #    self.sPoints_Labels.append('BP'+str(k_iter))
#                 
#                 #elif self._pyDScont['EQ2'].getSpecialPoint('LP1'):
#                 #    self.sPoints_X.append(self._pyDScont['EQ1'].getSpecialPoint('LP'+str(k_iter))[2])
#                 #    self.sPoints_Y.append(self._pyDScont['EQ1'].getSpecialPoint('LP'+str(k_iter))[0])
#                 #    self.sPoints_Z.append(self._pyDScont['EQ1'].getSpecialPoint('LP'+str(k_iter))[1])
#                 #    self.sPoints_Labels.append('LP'+str(k_iter))
#                 #    k_iter+=1
#                 
#                 print('Special Points on curve: ', self._specialPoints)
#                 
#                 plt.clf()
#                 _fig_formatting_2D(xdata=[self._pyDScont['EQ1'].sol[self._bifurcationParameter]], 
#                                 ydata=[self._YDATA],
#                                 xlab = self._LabelX, 
#                                 ylab = self._LabelY,
#                                 specialPoints=self._specialPoints, 
#                                 eigenvalues=[np.array([self._pyDScont['EQ1'].sol[kk].labels['EP']['data'].evals for kk in range(len(self._pyDScont['EQ1'].sol[self._stateVariable1]))])], 
#                                 ax_reformat=False, curve_replot=False, fontsize=self._chooseFontSize)
# 
# #                self._pyDScont.plot.fig1.axes1.axes.set_title('Bifurcation Diagram')
#                 #else:
#                 #    pass
#                 #assert false
#             else:
#                 if self._plottingMethod.lower() != 'pyds':
#                     self._showErrorMessage('Unknown plotType argument: using default pyDS tool plotting<br>')    
#                 if self._stateVariable2 == None:
#                     # 2-d bifurcation diagram
#                     if self._silent == True:
#                         #assert false ## @todo: enable when Thomas implements in-housse plotting routines
#                         self._pyDScont.display([self._bifurcationParameter, self._stateVariable1], axes = None, stability = True)
#                     else:
#                         self._pyDScont.display([self._bifurcationParameter, self._stateVariable1], stability = True, figure = self._figureNum)
#     #                self._pyDScont.plot.fig1.axes1.axes.set_title('Bifurcation Diagram')
#                 else:
#                     pass
#         self._logs.append(log)
# 
#     def _replot_bifurcation(self):
#         for name, value in self._controller._widgetsFreeParams.items():
#             self._pyDSmodel.pars[name] = value.value
#  
#         self._pyDScont.plot.clearall()
#         
# #        self._pyDSmodel.ics      = {'A': 0.1, 'B': 0.9 }    ## @todo: replace           
#         self._pyDSode = dst.Generator.Vode_ODEsystem(self._pyDSmodel)  ## @todo: add to __init__()
#         self._pyDSode.set(pars = {self._bifurcationParameter: self._bifInit} )                       ## @todo remove magic number
#         self._pyDScont = dst.ContClass(self._pyDSode)              ## Set up continuation class (@todo: add to __init__())
# ##        self._pyDScont.newCurve(self._pyDScontArgs)
# #        self._pyDScont['EQ1'].reset(self._pyDSmodel.pars)
# #        self._pyDSode.set(pars = self._pyDSmodel.pars)
# #        self._pyDScont['EQ1'].reset()
# #        self._pyDScont.update(self._pyDScontArgs)                         ## @todo: what does this do?
#         self._plot_bifurcation()
# 
#     # utility function to mangle variable names in equations so they are accepted by PyDStool
#     def _pydstoolify(self, equation):
#         equation = str(equation)
#         equation = equation.replace('{', '')
#         equation = equation.replace('}', '')
#         equation = equation.replace('_', '')
#         equation = equation.replace('\\', '')
#         
#         return equation
#     
#     
#     def _build_bookmark(self, includeParams = True):
#         if not self._silent:
#             logStr = "bookmark = "
#         else:
#             logStr = ""
#         logStr += "<modelName>." + self._generatingCommand + "('" + str(self._bifurcationParameter) + "', '" + str(self._stateVariable1) +"', "
#         if self._stateVariable2 != None:
#             logStr += "'" + str(self._stateVariable2) + "', "
#         if includeParams:
#             logStr += self._get_bookmarks_params() + ", "
#         if len(self._generatingKwargs) > 0:
#             for key in self._generatingKwargs:
#                 logStr += key + " = " + str(self._generatingKwargs[key]) + ", "
#         logStr += "bookmark = False"
#         logStr += ")"
#         logStr = logStr.replace('\\', '\\\\')
#         
#         return logStr



## agent on networks view on model 
class MuMoTmultiagentView(MuMoTview):
    _colors = None
    ## structure to store the communication network
    _graph = None
    ## type of network used in the M-A simulation
    _netType = None
    ## network parameter which varies with respect to each type of network
    ## (for Erdos-Renyi is linking probability, for Barabasi-Albert is the number of edge per new node, for spatial is the communication range)
    _netParam = None
    ## speed of the particle on one timestep (only for dynamic netType)
    _particleSpeed = None
    ## corelatedness in the random walk motion (only for dynamic netType)
    _motionCorrelatedness = None
    ## list of agents involved in the simulation
    _agents = None
    ## list of agents' positions
    _positions = None
    _positionHistory = None
    ## Arena size: width
    _arena_width = 1
    ## Arena size: height
    _arena_height = 1
    ## total number of agents in the simulation
    _systemSize = None
    ## the system state at the start of the simulation (timestep zero) described as proportion of _systemSize
    _initialState = None
    ## random seed
    _randomSeed = None
    ## simulation length (in the same time unit of the rates)
    _maxTime = None
    ## number of simulation timesteps
    _maxTimeSteps = None
    ## time scaling (i.e. lenght of each timestep)
    _timestepSize = None
    ## dictionary of rates
    _ratesDict = None
    ## visualisation type
    _visualisationType = None
    ## flag to plot proportions or full populations
    _plotProportions = None
    ## visualise the agent trace (on moving particles)
    _showTrace = None
    ## visualise the agent trace (on moving particles)
    _showInteractions = None
    ## realtimePlot flag (TRUE = the plot is updated each timestep of the simulation; FALSE = it is updated once at the end of the simulation)
    _realtimePlot = None
    ## latest computed results
    _latestResults = None

    def __init__(self, model, controller, MAParams, figure = None, params = None, **kwargs):
        super().__init__(model=model, controller=controller, figure=figure, params=params, **kwargs)

        with io.capture_output() as log:
#         if True:
                      
            if self._controller == None:
                # storing the rates for each rule
                ## @todo moving it to general method?
                self._ratesDict = {}
                rates_input_dict = dict( params )
                for key in rates_input_dict.keys():
                    rates_input_dict[str(process_sympy(str(key)))] = rates_input_dict.pop(key) # replace the dictionary key with the str of the SymPy symbol
                # create the self._ratesDict 
                for rule in self._mumotModel._rules:
                    self._ratesDict[str(rule.rate)] = rates_input_dict[str(rule.rate)] 
            
            # storing the initial state
            self._initialState = {}
            for state,pop in MAParams["initialState"].items():
                if isinstance(state, str):
                    self._initialState[process_sympy(state)] = pop # convert string into SymPy symbol
                else:
                    self._initialState[state] = pop
            self._maxTime = MAParams["maxTime"]
            self._randomSeed = MAParams["randomSeed"]
            self._visualisationType = MAParams["visualisationType"]
            self._plotProportions = MAParams["plotProportions"]
            self._netType = _decodeNetworkTypeFromString(MAParams['netType'])
            self._netParam = MAParams['netParam']
            self._motionCorrelatedness = MAParams['motionCorrelatedness']
            self._particleSpeed = MAParams['particleSpeed']
            self._timestepSize = MAParams.get('timestepSize',1)
            self._showTrace = MAParams.get('showTrace',False)
            self._showInteractions = MAParams.get('showInteractions',False)
            self._realtimePlot = MAParams.get('realtimePlot', False)
            
            self._update_net_params()
            
            self._mumotModel._getSingleAgentRules()
            #print(self._mumotModel._agentProbabilities)
            
            # check if any network is available or only moving particles
            onlyDynamic = False
            for rule in self._mumotModel._rules:
                if EMPTYSET_SYMBOL in rule.lhsReactants or EMPTYSET_SYMBOL in rule.rhsReactants:
                    onlyDynamic = True
                    break
            if onlyDynamic:
                #if (not self._controller) and (not self._netType == NetworkType.DYNAMIC): # if the user has specified the network type, we notify him/her through error-message
                #    self._errorMessage.value = "Only Moving-Particle netType is available when rules contain the emptyset."
                if not self._netType == NetworkType.DYNAMIC: print("Only Moving-Particle netType is available when rules contain the emptyset.")
                self._netType = NetworkType.DYNAMIC
                if self._controller: # updating value and disabling widget
                    self._controller._widgetsExtraParams['netType'].value = NetworkType.DYNAMIC
                    self._controller._widgetsExtraParams['netType'].disabled = True
                    self._controller._update_net_params()
            
#             # init graph
#             if self._silent:
#                 self._initGraph(graphType=self._netType, numNodes=sum(self._initialState.values()), netParam=self._netParam)

            # map colouts to each reactant
            #colors = cm.rainbow(np.linspace(0, 1, len(self._mumotModel._reactants) ))  # @UndefinedVariable
            self._colors = {}
            i = 0
            for state in sorted(self._initialState.keys(), key=str): #sorted(self._mumotModel._reactants, key=str):
                self._colors[state] = line_color_list[i] 
                i += 1            
            
        self._logs.append(log)
        if not self._silent:
            self._plot_timeEvolution()
        
    def _print_standalone_view_cmd(self):
        MAParams = {}
        initState_str = {}
        for state,pop in self._initialState.items():
            initState_str[str(state)] = pop
        MAParams["initialState"] = initState_str 
        MAParams["maxTime"] = self._maxTime 
        MAParams["randomSeed"] = self._randomSeed
        MAParams["visualisationType"] = self._visualisationType
        MAParams["plotProportions"] = self._plotProportions
        MAParams['netType'] = _encodeNetworkTypeToString(self._netType)
        MAParams['netParam'] = self._netParam 
        MAParams['motionCorrelatedness'] = self._motionCorrelatedness
        MAParams['particleSpeed'] = self._particleSpeed
        MAParams['timestepSize'] = self._timestepSize
        MAParams['showTrace'] = self._showTrace
        MAParams['showInteractions'] = self._showInteractions
#         sortedDict = "{"
#         for key,value in sorted(MAParams.items()):
#             sortedDict += "'" + key + "': " + str(value) + ", "
#         sortedDict += "}"
        print( "mmt.MuMoTmultiagentView(model1, None, MAParams = " + str(MAParams) + ", params = " + str( list(self._ratesDict.items()) ) + " )")
    
    ## reads the new parameters (in case they changed in the controller)
    ## this function should only update local parameters and not compute data
    def _update_params(self):
        if self._controller != None:
            # getting the rates
            ## @todo moving it to general method?
#             freeParamDict = {}
#             for name, value in self._controller._widgetsFreeParams.items():
#                 freeParamDict[ Symbol(name) ] = value.value
            freeParamDict = self._get_argDict()
            self._ratesDict = {}
            for rule in self._mumotModel._rules:
                self._ratesDict[str(rule.rate)] = rule.rate.subs(freeParamDict)
            #print("RATES DICT IS: " + str(self._ratesDict))
            self._systemSize = self._getSystemSize()
            
            # getting other parameters specific to M-A view
            for state in self._initialState.keys():
                self._initialState[state] = self._controller._widgetsExtraParams['init'+str(state)].value
            #numNodes = sum(self._initialState.values())
            self._randomSeed = self._controller._widgetsExtraParams['randomSeed'].value
            self._visualisationType = self._controller._widgetsPlotOnly['visualisationType'].value
            self._plotProportions = self._controller._widgetsPlotOnly['plotProportions'].value
            self._maxTime = self._controller._widgetsExtraParams['maxTime'].value
            self._netType = self._controller._widgetsExtraParams['netType'].value
            self._netParam = self._controller._widgetsExtraParams['netParam'].value
            self._motionCorrelatedness = self._controller._widgetsExtraParams['motionCorrelatedness'].value
            self._particleSpeed = self._controller._widgetsExtraParams['particleSpeed'].value
            self._timestepSize = self._controller._widgetsExtraParams['timestepSize'].value
            self._showTrace = self._controller._widgetsPlotOnly['showTrace'].value
            self._showInteractions = self._controller._widgetsPlotOnly['showInteractions'].value
            self._realtimePlot = self._controller._widgetsExtraParams['realtimePlot'].value

    
    def _plot_timeEvolution(self, _=None):
        with io.capture_output() as log:
#         if True:
            # update parameters (needed only if there's a controller)
            self._update_params()
            self._log("Multiagent simulation")
            self._print_standalone_view_cmd()

            # init the random seed
            np.random.seed(self._randomSeed)
            
            #self._convertRatesIntoProbabilities()
            self._computeScalingFactor()

            # Clearing the plot and setting the axes
            self._initFigure()
            
            self._latestResults = self._runMultiagent()
#             print("Temporal evolution per state: " + str(self._latestResults[0]))
            
            ## Final Plot
            if not self._realtimePlot:
                self._updateMultiagentFigure(0, self._latestResults[0], positionHistory=self._latestResults[1], pos_layout=self._latestResults[2])
            
        self._logs.append(log)      
    
    def _redrawOnly(self, _=None):
        self._update_params()
        self._initFigure()
        self._updateMultiagentFigure(0, self._latestResults[0], positionHistory=self._latestResults[1], pos_layout=self._latestResults[2])
        
    def _runMultiagent(self):
        # init the controller variables
        currentState = self._initMultiagent()
        
        # init the network
        self._initGraph(graphType=self._netType, numNodes=sum(currentState.values()), netParam=self._netParam)
        
        # init logging structs
#         historyState = []
#         historyState.append(initialState)
        evo = {}
        for state,pop in currentState.items():
            evo[state] = []
            evo[state].append(pop)
            
        dynamicNetwork = self._netType == NetworkType.DYNAMIC
        if dynamicNetwork:
            self._positionHistory = []
            for _ in np.arange(sum(currentState.values())):
                self._positionHistory.append( [] )
        else:
            self._positionHistory = None
            
        # init progress bar
        if self._controller != None: self._controller._progressBar.max = self._maxTime
        
        # store the graph layout (only for 'graph' visualisation)
        if (not dynamicNetwork): # and self._visualisationType == "graph":
            pos_layout = nx.circular_layout(self._graph)
        else:
            pos_layout = None
        
        for i in np.arange(1, self._maxTimeSteps+1):
            #print("Time: " + str(i))
            if self._controller != None: self._controller._progressBar.value = i*self._timestepSize
            if self._controller != None: self._controller._progressBar.description = "Loading " + str(round(i/self._maxTimeSteps*100)) + "%:"
            if dynamicNetwork: # and self._showTrace:
                for idx, _ in enumerate(self._agents): # second element _ is the agent (unused)
                    self._positionHistory[idx].append( self._positions[idx] )
            
            currentState = self._stepMultiagent()
            #print("T:" + str(i) + " - " + str(currentState))
#             historyState.append(currentState)
            for state,pop in currentState.items():
                evo[state].append(pop)
            
            ## Plotting
            if self._realtimePlot:
                self._updateMultiagentFigure(i, evo, positionHistory=self._positionHistory, pos_layout=pos_layout)

                
        if self._controller != None: self._controller._progressBar.description = "Completed 100%:"
#         print("State distribution each timestep: " + str(historyState))
        return (evo, self._positionHistory, pos_layout)
        
    def _initFigure(self):
        if not self._silent:
            plt.figure(self._figureNum)
            plt.clf()

        if (self._visualisationType == 'evo'):
            plt.axes().set_aspect('auto')
            # create the frame
            #self._plot.axis([0, self._maxTime, 0, totAgents])
            #plt.xlim((0, self._maxTime))
            #plt.ylim((0, self._systemSize))
            #self._figure.show()
            actualMaxTime = self._maxTimeSteps * self._timestepSize 
            _fig_formatting_2D(self._figure, xlab="Time", ylab="Reactants", curve_replot=(not self._silent), choose_xrange=(0, actualMaxTime), choose_yrange=(0, self._systemSize) )
        elif self._visualisationType == "graph": #and self._netType == NetworkType.DYNAMIC:
            plt.axes().set_aspect('equal')
        elif (self._visualisationType == "final"):
#             plt.axes().set_aspect('equal') #for piechart
            plt.axes().set_aspect('auto') # for barchart
            if self._plotProportions:                
                plt.ylim((0, 1.0))
            else:
                plt.ylim((0, self._systemSize))
    
    
    def _updateMultiagentFigure(self, i, evo, positionHistory, pos_layout):
        if (self._visualisationType == "evo"):
            actualMaxTime = self._maxTimeSteps * self._timestepSize
            actualMaxTime = self._maxTime
#             print("ACTUAL MAX TIME " + str(actualMaxTime))
            # If realtime-plot mode, draw only the last timestep rather than overlay all
            if (i>1):
                xdata = []
                ydata = []
                for state in sorted(self._initialState.keys(), key=str):
                    xdata.append( list(np.arange(len(evo[state]))*self._timestepSize) )
                    if self._plotProportions:
                        ytmp = [y/self._systemSize for y in evo[state]]
                        ydata.append(ytmp)
                        yrange = max(1.0, max(ytmp))
                    else:
                        ydata.append( evo[state] )
                        yrange = max(self._systemSize, max(evo[state]))
                    ## @todo replot only the last part (rather than all the line)
#                     xdata.append( [i-1,i] )
#                     pop = evo[state]
#                     ydata.append( pop[len(pop)-2:len(pop)] )
                    #plt.plot(evo[state], color=self._colors[state])
                _fig_formatting_2D(xdata=xdata, ydata=ydata, curve_replot=False, choose_xrange=(0, actualMaxTime), choose_yrange=(0, yrange) )
            else: # otherwise, plot all time-evolution
                xdata = []
                ydata = []
                labels = []
                for state in sorted(self._initialState.keys(), key=str):
                    xdata.append( list(np.arange(len(evo[state]))*self._timestepSize) )
                    if self._plotProportions:
                        ytmp = [y/self._systemSize for y in evo[state]]
                        ydata.append(ytmp)
                        yrange = max(1.0, max(ytmp))
                    else:
                        ydata.append( evo[state] )
                        yrange = max(self._systemSize, max(evo[state]))
                    labels.append(state)
                #xdata=[list(np.arange(len(list(evo.values())[0])))]*len(evo.values()), ydata=list(evo.values()), curvelab=list(evo.keys())
                    #plt.plot(evo[state], color=self._colors[state])
                _fig_formatting_2D(xdata=xdata, ydata=ydata, curvelab=labels, curve_replot=False, choose_xrange=(0, actualMaxTime), choose_yrange=(0, yrange) )

        elif (self._visualisationType == "graph"):
            self._initFigure()
            #plt.clf()
            #plt.axes().set_aspect('equal')
            if self._netType == NetworkType.DYNAMIC:
                plt.xlim((0, 1))
                plt.ylim((0, 1))
                #plt.axes().set_aspect('equal')
#                     xs = [p[0] for p in positions]
#                     ys = [p[1] for p in positions]
#                     plt.plot(xs, ys, 'o' )
                xs = {}
                ys = {}
                for state in self._initialState.keys():
                    xs[state] = []
                    ys[state] = []
                for a in np.arange(len(self._positions)):
                    xs[self._agents[a]].append( self._positions[a][0] )
                    ys[self._agents[a]].append( self._positions[a][1] )
                    
                    if self._showInteractions:
                        for n in self._getNeighbours(a, self._positions, self._netParam): 
                            plt.plot((self._positions[a][0], self._positions[n][0]),(self._positions[a][1], self._positions[n][1]), '-', c='y')
                    
                    if self._showTrace:
                        trace_xs = [p[0] for p in positionHistory[a] ]
                        trace_ys = [p[1] for p in positionHistory[a] ]
                        trace_xs.append( self._positions[a][0] )
                        trace_ys.append( self._positions[a][1] )
                        plt.plot( trace_xs, trace_ys, '-', c='0.6') 
                for state in self._initialState.keys():
                    plt.plot(xs.get(state,[]), ys.get(state,[]), 'o', c=self._colors[state] )
            else:
                stateColors=[]
                for n in self._graph.nodes():
                    stateColors.append( self._colors.get( self._agents[n], 'w') ) 
                nx.draw(self._graph, pos_layout, node_color=stateColors, with_labels=True)
                # plot legend
                markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='', markersize=15) for color in self._colors.values()]
                plt.legend(markers, self._colors.keys(), bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0., numpoints=1)
        elif (self._visualisationType == "final"):
            self._initFigure()
            
            finaldata = []
            labels = []
            colors = []
            for state in sorted(self._initialState.keys(), key=str):
                if self._plotProportions:
                    finaldata.append( evo[state][len(evo[state])-1]/self._systemSize )
                else:
                    finaldata.append( evo[state][len(evo[state])-1] )
                labels.append(state)
                colors.append(self._colors[state])
             
#             plt.pie(finaldata, labels=labels, autopct=_make_autopct(piedata), colors=colors) #shadow=True, startangle=90,
            xpos = np.arange(len( self._initialState.keys() ))  # the x locations for the bars
            width = 1       # the width of the bars
            plt.bar(xpos, finaldata, width, color=colors) #, yerr=stdev)
            plt.axes().set_xticks(xpos + width / 2)
            plt.axes().set_xticklabels(sorted(self._initialState.keys(), key=str))

        # update the figure
        if not self._silent:
            self._figure.canvas.draw()
    
    def _singleRun(self, randomSeed):
        # set random seed
        np.random.seed(randomSeed)
        
        # init the controller variables
        self._initMultiagent()
        
        # Create logging structs
        historyState = []
        evo = {}
        evo['time'] = [0]
        for state,pop in self._initialState.items():
            evo[state] = []
            evo[state].append(pop)
         
        t = 0
        currentState = []
        for reactant in sorted(self._mumotModel._reactants, key=str):
            currentState.append(self._initialState[reactant])
        historyState.append( [t] + currentState )
 
        for t in np.arange(1, self._maxTimeSteps+1):
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
  
    def _computeScalingFactor(self):
        # Determining the minimum speed of the process (thus the max-scaling factor)
        maxRatesAll = 0
        for reactant, reactions in self._mumotModel._agentProbabilities.items():
            if reactant == EMPTYSET_SYMBOL: continue # not considering the spontaneous births as limiting component for simulation step
            sumRates = 0
            for reaction in reactions:
                sumRates += self._ratesDict[str(reaction[1])]  
            if sumRates > maxRatesAll:
                maxRatesAll = sumRates
        
        if maxRatesAll>0: maxTimestepSize = 1/maxRatesAll 
        else: maxTimestepSize = 1
        if self._timestepSize > maxTimestepSize:
            self._timestepSize = maxTimestepSize
        self._maxTimeSteps = math.ceil( self._maxTime / self._timestepSize )
        if self._controller != None: self._update_timestepSize_widget(self._timestepSize, maxTimestepSize, self._maxTimeSteps)
#         print("timestepSize s=" + str(self._timestepSize))
    
    def _update_timestepSize_widget(self, timestepSize, maxTimestepSize, maxTimeSteps):
        if not self._controller._widgetsExtraParams['timestepSize'].value == timestepSize:
            self._controller._widgetsExtraParams['timestepSize'].unobserve_all()
            if (self._controller._widgetsExtraParams['timestepSize'].max < timestepSize):
                self._controller._widgetsExtraParams['timestepSize'].max = maxTimestepSize
            if (self._controller._widgetsExtraParams['timestepSize'].min > timestepSize):
                self._controller._widgetsExtraParams['timestepSize'].min = timestepSize
            self._controller._widgetsExtraParams['timestepSize'].value = timestepSize
            self._controller._widgetsExtraParams['timestepSize'].observe(self._controller._replotFunction, 'value')
        if not self._controller._widgetsExtraParams['timestepSize'].max == maxTimestepSize:
            self._controller._widgetsExtraParams['timestepSize'].max = maxTimestepSize
            self._controller._widgetsExtraParams['timestepSize'].min = min(maxTimestepSize/100, timestepSize)
            self._controller._widgetsExtraParams['timestepSize'].step = self._controller._widgetsExtraParams['timestepSize'].min
        self._controller._widgetsExtraParams['maxTime'].description = "Simulation time (equivalent to " + str(maxTimeSteps) + " simulation timesteps)"

                    
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
            ## @todo: implement network generate by placing points (with local communication range) randomly in 2D space
            print("ERROR: Graphs of type SPACE are not implemented yet.")
            return
        elif (graphType == NetworkType.DYNAMIC):
            self._positions = []
            for _ in range(numNodes):
                x = np.random.rand() * self._arena_width
                y = np.random.rand() * self._arena_height
                o = np.random.rand() * np.pi * 2.0
                self._positions.append( (x,y,o) )
            return

    def _initMultiagent(self):
        # initialise populations by multiplying proportion with _systemSize
        currentState = {s:math.floor(p*self._systemSize) for s,p in self._initialState.items()}
        # init the agents list
        self._agents = []
        for state, pop in currentState.items():
            self._agents.extend( [state]*pop )
        self._agents = np.random.permutation(self._agents).tolist() # random shuffling of elements (useful to avoid initial clusters in networks)
        return currentState
        
    def _stepMultiagent(self):
        tmp_agents = copy.deepcopy(self._agents)
        dynamic = self._netType == NetworkType.DYNAMIC
        if dynamic:
            tmp_positions = copy.deepcopy(self._positions)
            communication_range = self._netParam
        children = []
        activeAgents = [True]*len(self._agents)
        #for idx, a in enumerate(self._agents):
        # to execute in random order the agents I just create a shuffled list of idx and I follow that
        indexes = np.arange(0,len(self._agents))
        indexes = np.random.permutation(indexes).tolist() # shuffle the indexes
        for idx in indexes:
            a = self._agents[idx]
            # if moving-particles the agent moves
            if dynamic:
#                 print("Agent " + str(idx) + " moved from " + str(self._positions[idx]) )
                self._positions[idx] = self._updatePosition( self._positions[idx][0], self._positions[idx][1], self._positions[idx][2], self._particleSpeed, self._motionCorrelatedness)
#                 print("to position " + str(self._positions[idx]) )
            
            # the step is executed only if the agent is active
            if not activeAgents[idx]: continue
            
            # computing the list of neighbours for the given agent
            if dynamic:
                neighNodes = self._getNeighbours(idx, tmp_positions, communication_range)
            else:
                neighNodes = list(nx.all_neighbors(self._graph, idx))            
            neighNodes = np.random.permutation(neighNodes).tolist() # random shuffling of neighNodes (to randomise interactions)
            neighAgents = [tmp_agents[x] for x in neighNodes] # creating the list of neighbours' states 
            neighActive = [activeAgents[x] for x in neighNodes] # creating the list of neighbour' activity-status

#                 print("Neighs of agent " + str(idx) + " are " + str(neighNodes) + " with states " + str(neighAgents) )
            # run one simulation step for agent a
            oneStepOutput = self._stepOneAgent(a, neighAgents, neighActive)
            self._agents[idx] = oneStepOutput[0][0]
            # check for new particles generated in the step
            if len(oneStepOutput[0]) >  1: # new particles must be created
                for particle in oneStepOutput[0]:
                    children.append( (particle, tmp_positions[idx]) )
            for idx_c, neighChange in enumerate(oneStepOutput[1]):
                if neighChange:
                    activeAgents[ neighNodes[idx_c] ] = False
                    self._agents[ neighNodes[idx_c] ] = neighChange

        # add the new agents coming from splitting (possible only for moving-particles view)
        for child in children: 
            self._agents.append( child[0] )
            self._positions.append( child[1] )
            self._positionHistory.append( [] )
            idx = len(self._positions)-1
            self._positionHistory[idx].append( self._positions[idx] )
            self._positions[idx] = ( self._positions[idx][0], self._positions[idx][1], np.random.rand() * np.pi * 2.0) # set random orientation 
#             self._positions[idx][2] = np.random.rand() * np.pi * 2.0 # set random orientation 
            self._positions[idx] = self._updatePosition( self._positions[idx][0], self._positions[idx][1], self._positions[idx][2], self._particleSpeed, self._motionCorrelatedness)

        # compute self birth (possible only for moving-particles view)
        for birth in self._mumotModel._agentProbabilities[EMPTYSET_SYMBOL]:
            birthRate = self._ratesDict[str(birth[1])] * self._timestepSize # scale the rate
            decimal = birthRate % 1
            birthsNum = int(birthRate - decimal)
            np.random.rand()
            if (np.random.rand() < decimal): birthsNum += 1
            #print ( "Birth rate " + str(birth[1]) + " triggers " + str(birthsNum) + " newborns")
            for _ in range(birthsNum):
                for newborn in birth[2]:
                    self._agents.append( newborn )
                    self._positions.append( (np.random.rand() * self._arena_width, np.random.rand() * self._arena_height, np.random.rand() * np.pi * 2.0) )
                    self._positionHistory.append( [] )
                    self._positionHistory[len(self._positions)-1].append( self._positions[len(self._positions)-1] )
        
        # Remove from lists (_agents, _positions, and _positionHistory) the 'dead' agents (possible only for moving-particles view)
        deads = [idx for idx, a in enumerate(self._agents) if a == EMPTYSET_SYMBOL]
#         print("Dead list is " + str(deads))
        for dead in reversed(deads):
            del self._agents[dead]
            del self._positions[dead]
            del self._positionHistory[dead]
            
        currentState = {state : self._agents.count(state) for state in self._initialState.keys()} #self._mumotModel._reactants | self._mumotModel._constantReactants}
        return currentState

    ## one timestep for one agent
    def _stepOneAgent(self, agent, neighs, activeNeighs):
        rnd = np.random.rand()
        lastVal = 0
        neighChanges = [None]*len(neighs)
        # counting how many neighbours for each state (to be uses for the interaction probabilities)
        neighCount = {x:neighs.count(x) for x in self._initialState.keys()} #self._mumotModel._reactants | self._mumotModel._constantReactants}
        for idx, neigh in enumerate(neighs):
            if not activeNeighs[idx]:
                neighCount[neigh] -= 1
#         print("Agent " + str(agent) + " with probSet=" + str(probSets))
#         print("nc:"+str(neighCount))
        for reaction in self._mumotModel._agentProbabilities[agent]:
            popScaling = 1
            rate = self._ratesDict[str(reaction[1])] * self._timestepSize # scaling the rate by the timeStep size
            if len(neighs) >= len(reaction[0]):
                j = 0
                for reagent in reaction[0]:
                    popScaling *= neighCount[reagent]/(len(neighs)-j) if neighCount[reagent] >= reaction[0].count(reagent) else 0
                    j += 1
            else:
                popScaling = 0
            val = popScaling * rate
            #print("For reaction: " + str(agent) + "+" + str(reaction[0]) + " the popScaling is " + str(popScaling))
            if (rnd < val + lastVal):
                # A state change happened!
                #print("Reaction: " + str(reaction[1]) + " by agent " + str(agent) + " with agent(s) " + str(reaction[0]) + " becomes " + str(reaction[2]) + " &others: " +str(reaction[3]))
                #print("Val was: " + str(val) + " lastVal: " + str(lastVal) + " and rand: " + str(rnd))
                
                # locking the other reagents involved in the reaction
                for idx_r, reagent in enumerate(reaction[0]):
                    for idx_n, neigh in enumerate(neighs):
                        if neigh == reagent and activeNeighs[idx_n] and neighChanges[idx_n] == None:
                            neighChanges[idx_n] = reaction[3][idx_r]
                            break
                
                return (reaction[2], neighChanges)
            else:
                lastVal += val
        # No state change happened
        return ([agent],neighChanges)
    
    
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

    ## return the (index) list of neighbours of 'agent'
    def _getNeighbours(self, agent, positions, distance_range):
        neighbour_list = []
        for neigh in np.arange(len(positions)):
            if (not neigh == agent) and (self._distance_on_torus(positions[agent][0], positions[agent][1], positions[neigh][0], positions[neigh][1]) < distance_range):
                neighbour_list.append(neigh)
        return neighbour_list
    
    ## returns the minimum distance calucalted on the torus given by periodic boundary conditions
    def _distance_on_torus( self, x_1, y_1, x_2, y_2 ):
        return np.sqrt(min(abs(x_1 - x_2), self._arena_width - abs(x_1 - x_2))**2 + 
                    min(abs(y_1 - y_2), self._arena_height - abs(y_1 - y_2))**2)
    
    ## updates the widgets related to the netType (it cannot be a MuMoTcontroller method because with multi-controller it needs to point to the right _controller)
    def _update_net_params(self, _=None):
        # oder of assignment is important (first, update the min and max, later, the value)
        self._controller._widgetsExtraParams['netParam'].unobserve_all()
        if (self._controller._widgetsExtraParams['netType'].value == NetworkType.FULLY_CONNECTED):
#             self._controller._widgetsExtraParams['netParam'].min = 0
#             self._controller._widgetsExtraParams['netParam'].max = 1
#             self._controller._widgetsExtraParams['netParam'].step = 1
#             self._controller._widgetsExtraParams['netParam'].value = 0
#             self._controller._widgetsExtraParams['netParam'].disabled = True
#             self._controller._widgetsExtraParams['netParam'].description = "None"
            self._controller._widgetsExtraParams['netParam'].layout.display = 'none'
        elif (self._controller._widgetsExtraParams['netType'].value == NetworkType.ERSOS_RENYI):
#             self._controller._widgetsExtraParams['netParam'].disabled = False
            self._controller._widgetsExtraParams['netParam'].layout.display = 'flex'
            self._controller._widgetsExtraParams['netParam'].min = 0.1
            self._controller._widgetsExtraParams['netParam'].max = 1
            self._controller._widgetsExtraParams['netParam'].step = 0.1
            self._controller._widgetsExtraParams['netParam'].value = 0.5
            self._controller._widgetsExtraParams['netParam'].description = "Network connectivity parameter (link probability)"
        elif (self._controller._widgetsExtraParams['netType'].value == NetworkType.BARABASI_ALBERT):
#             self._controller._widgetsExtraParams['netParam'].disabled = False
            self._controller._widgetsExtraParams['netParam'].layout.display = 'flex'
            maxVal = self._systemSize-1
            self._controller._widgetsExtraParams['netParam'].min = 1
            self._controller._widgetsExtraParams['netParam'].max = maxVal
            self._controller._widgetsExtraParams['netParam'].step = 1
            self._controller._widgetsExtraParams['netParam'].value = min(maxVal, 3)
            self._controller._widgetsExtraParams['netParam'].description = "Network connectivity parameter (new edges)"            
        elif (self._controller._widgetsExtraParams['netType'].value == NetworkType.SPACE):
            self._controller._widgetsExtraParams['netParam'].value = -1
         
        if (self._controller._widgetsExtraParams['netType'].value == NetworkType.DYNAMIC):
#             self._controller._widgetsExtraParams['netParam'].disabled = False
            self._controller._widgetsExtraParams['netParam'].layout.display = 'flex'
            self._controller._widgetsExtraParams['netParam'].min = 0.0
            self._controller._widgetsExtraParams['netParam'].max = 1
            self._controller._widgetsExtraParams['netParam'].step = 0.05
            self._controller._widgetsExtraParams['netParam'].value = 0.1
            self._controller._widgetsExtraParams['netParam'].description = "Interaction range"
#             self._controller._widgetsExtraParams['particleSpeed'].disabled = False
#             self._controller._widgetsExtraParams['motionCorrelatedness'].disabled = False
#             self._controller._widgetsPlotOnly['showTrace'].disabled = False
#             self._controller._widgetsPlotOnly['showInteractions'].disabled = False
            self._controller._widgetsExtraParams['particleSpeed'].layout.display = 'flex'
            self._controller._widgetsExtraParams['motionCorrelatedness'].layout.display = 'flex'
            self._controller._widgetsPlotOnly['showTrace'].layout.display = 'flex'
            self._controller._widgetsPlotOnly['showInteractions'].layout.display = 'flex'
        else:
            #self._controller._widgetsExtraParams['particleSpeed'].disabled = True
#             self._controller._widgetsExtraParams['motionCorrelatedness'].disabled = True
#             self._controller._widgetsPlotOnly['showTrace'].disabled = True
#             self._controller._widgetsPlotOnly['showInteractions'].disabled = True
            self._controller._widgetsExtraParams['particleSpeed'].layout.display = 'none'
            self._controller._widgetsExtraParams['motionCorrelatedness'].layout.display = 'none'
            self._controller._widgetsPlotOnly['showTrace'].layout.display = 'none'
            self._controller._widgetsPlotOnly['showInteractions'].layout.display = 'none'
        self._controller._widgetsExtraParams['netParam'].observe(self._controller._replotFunction, 'value')

## agent on networks view on model 
class MuMoTSSAView(MuMoTview): 
    _colors = None
    ## a matrix form of the left-handside of the rules
    _reactantsMatrix = None 
    ## the effect of each rule
    _ruleChanges = None
    ## total number of agents in the simulation
    _systemSize = None
    ## the system state at the start of the simulation (timestep zero) described as proportion of _systemSize
    _initialState = None
    ## dictionary of rates
    _ratesDict = None
    ## number of simulation timesteps
    _maxTime = None
    ## random seed
    _randomSeed = None
    ## visualisation type
    _visualisationType = None
    ## flag to plot proportions or full populations
    _plotProportions = None
    ## realtimePlot flag (TRUE = the plot is updated each timestep of the simulation; FALSE = it is updated once at the end of the simulation)
    _realtimePlot = None
    ## latest computed results
    _latestResults = None

    def __init__(self, model, controller, ssaParams, figure = None, params = None, **kwargs):
        super().__init__(model, controller, figure=figure, params=params, **kwargs)

        with io.capture_output() as log:
#         if True:
            if self._controller == None:
                # storing the rates for each rule
                ## @todo moving it to general method?
                self._ratesDict = {}
                rates_input_dict = dict( params )
#                 print("DIct is " + str(rates_input_dict) )
                for key in rates_input_dict.keys():
                    rates_input_dict[str(process_sympy(str(key)))] = rates_input_dict.pop(key) # replace the dictionary key with the str of the SymPy symbol
                #print("DIct is " + str(rates_input_dict) )
                #print("_mumotModel._rules is " + str(self._mumotModel._rules) )
                # create the self._ratesDict
                for rule in self._mumotModel._rules:
                    self._ratesDict[str(rule.rate)] = rates_input_dict[str(rule.rate)] 
                     
                
            self._initialState = {}
            for state,pop in ssaParams["initialState"].items():
                if isinstance(state, str):
                    self._initialState[process_sympy(state)] = pop # convert string into SymPy symbol
                else:
                    self._initialState[state] = pop
            self._maxTime = ssaParams["maxTime"]
            self._randomSeed = ssaParams["randomSeed"]
            self._visualisationType = ssaParams["visualisationType"]
            self._plotProportions = ssaParams["plotProportions"]
            self._realtimePlot = ssaParams.get('realtimePlot', False)
            
            # map colouts to each reactant
            #colors = cm.rainbow(np.linspace(0, 1, len(self._mumotModel._reactants) ))  # @UndefinedVariable
            self._colors = {}
            i = 0
            for state in sorted(self._initialState.keys(), key=str): #sorted(self._mumotModel._reactants, key=str):
                self._colors[state] = line_color_list[i] 
                i += 1 
        
        self._logs.append(log)
        if not self._silent:
            self._plot_timeEvolution()

    def _update_params(self):
        if self._controller != None:
            # getting the rates
            ## @todo moving it to general method?
#             freeParamDict = {}
#             for name, value in self._controller._widgetsFreeParams.items():
#                 freeParamDict[ Symbol(name) ] = value.value
            freeParamDict = self._get_argDict()
            self._ratesDict = {}
            for rule in self._mumotModel._rules:
                self._ratesDict[str(rule.rate)] = rule.rate.subs(freeParamDict) 
            self._systemSize = self._getSystemSize()

            # getting other parameters specific to SSA
            for state in self._initialState.keys():
                self._initialState[state] = self._controller._widgetsExtraParams['init'+str(state)].value
            self._randomSeed = self._controller._widgetsExtraParams['randomSeed'].value
            self._visualisationType = self._controller._widgetsPlotOnly['visualisationType'].value
            self._plotProportions = self._controller._widgetsPlotOnly['plotProportions'].value
            self._maxTime = self._controller._widgetsExtraParams['maxTime'].value
            self._realtimePlot = self._controller._widgetsExtraParams['realtimePlot'].value
    
    def _print_standalone_view_cmd(self):
        ssaParams = {}
        initState_str = {}
        for state,pop in self._initialState.items():
            initState_str[str(state)] = pop
        ssaParams["initialState"] = initState_str 
        ssaParams["maxTime"] = self._maxTime 
        ssaParams["randomSeed"] = self._randomSeed
        ssaParams["visualisationType"] = self._visualisationType
        ssaParams["plotProportions"] = self._plotProportions
        print( "mmt.MuMoTSSAView(model1, None, ssaParams = " + str(ssaParams) + ", params = " + str( list(self._ratesDict.items()) ) + " )")
    
    def _redrawOnly(self, _=None):
        self._update_params()
        self._initFigure()
        self._updateSSAFigure(self._latestResults, fullPlot=True)
    
    def _plot_timeEvolution(self, _=None):
        with io.capture_output() as log:
#         if True:
            self._update_params()
            self._log("Stochastic Simulation Algorithm (SSA)")
            self._print_standalone_view_cmd()

            # inti the random seed
            np.random.seed(self._randomSeed)
            
            # organise rates and reactants into strunctures to run the SSA
            #print("Reactants are " + str(sorted(self._mumotModel._reactants, key=str)))
            #print("Const Reactants are " + str(sorted(self._mumotModel._constantReactants, key=str)))
            #self._createSSAmatrix()
            
            # Clearing the plot and setting the axes
            self._initFigure()
           
            self._latestResults = self._runSSA()
            
            ## Final Plot
            if not self._realtimePlot:
                self._updateSSAFigure(self._latestResults, fullPlot=True)
           
        self._logs.append(log)
        
    def _runSSA(self):          
        ## @todo move the progress bar in the view
        if self._controller != None: self._controller._progressBar.max = self._maxTime 
         
        # initialise time
        t = 0
        # initialise populations by multiplying proportion with _systemSize
        #currentState = copy.deepcopy(self._initialState)
        #currentState = {s:p*self._systemSize for s,p in self._initialState.items()}
        currentState = {}
        leftOvers = {}
        for state,prop in self._initialState.items():
            pop = prop*self._systemSize
            if _almostEqual(pop, math.floor(pop)):
                leftOvers[state] = pop - math.floor(pop)
            currentState[state] = math.floor(pop)
        # if approximations resulted in one agent less, it is added randomly (with probability proportional to the rounding quantities)
#         if sum(currentState.values()) < self._systemSize:
#             rnd = np.random.rand() * sum(leftOvers.values())
#             bottom = 0.0
#             for state, prob in leftOvers.items():
#                 if rnd >= bottom and rnd < (bottom + prob):
#                     currentState[state] += 1
#                     break
#                 bottom += prob
            
        # Create logging structs
        evo = {}
        evo['time'] = [0]
        for state,pop in currentState.items():
            evo[state] = []
            evo[state].append(pop)
            
 
        while t < self._maxTime:
            # update progress bar
            if self._controller != None: self._controller._progressBar.value = t
            if self._controller != None: self._controller._progressBar.description = "Loading " + str(round(t/self._maxTime*100)) + "%:"
#             print("Time: " + str(t))
             
            timeInterval,currentState = self._stepSSA(currentState)
            if timeInterval == math.inf: timeInterval = self._maxTime-t
            # increment time
            t += timeInterval
             
            # log step
#             historyState.append( [t] + currentState )
#             for state,pop in zip(sorted(self._mumotModel._reactants|self._mumotModel._constantReactants, key=str), currentState):
            for state,pop in currentState.items():
                evo[state].append(pop)
            evo['time'].append(t)
            
#             print (evo)
            ## Plotting each timestep
            if self._realtimePlot:
                self._updateSSAFigure(evo, fullPlot=False)
         
        if self._controller != None: self._controller._progressBar.value = self._controller._progressBar.max
        if self._controller != None: self._controller._progressBar.description = "Completed 100%:"
#         print("State distribution each timestep: " + str(historyState))
#         print("Temporal evolution per state: " + str(evo))
        return evo
    
    def _initFigure(self):
        if not self._silent:
            plt.figure(self._figureNum)
            # Clearing the plot
            plt.clf()

        # Setting the axes
        if (self._visualisationType == 'evo'):
            plt.axes().set_aspect('auto')
            # create the frame
            #totAgents = self._systemSize
#                     self._plot.axis([0, self._maxTime, 0, totAgents])
            #plt.xlim((0, self._maxTime))
            #plt.ylim((0, totAgents))
#                 # make legend
#                 markers = [plt.Line2D([0,0],[0,0],color=color, marker='', linestyle='-') for color in self._colors.values()]
#                 self._plot.legend(markers, self._colors.keys(), bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)
            #if not self._silent: _fig_formatting_2D(self._figure, xlab="Time", ylab="Reactants", curve_replot=True)
            _fig_formatting_2D(self._figure, xlab="Time", ylab="Reactants", curve_replot=(not self._silent), choose_xrange=(0, self._maxTime), choose_yrange=(0, self._systemSize))
            #self._figure.show()
        elif (self._visualisationType == "final"):
#             plt.axes().set_aspect('equal') #for piechart
            plt.axes().set_aspect('auto') # for barchart
            if self._plotProportions:                
                plt.ylim((0, 1.0))
            else:
                plt.ylim((0, self._systemSize))
    
    def _updateSSAFigure(self, evo, fullPlot=True):
        if (self._visualisationType == "evo"):
            if not fullPlot and len(evo['time']) <= 2: # if incremental plot is requested, but it's the first item, we operate as fullPlot (to allow legend)
                fullPlot = True
            # (if fullPlot) plot all time-evolution
            if fullPlot:
                xdata = []
                ydata = []
                labels = []
                for state in sorted(self._initialState.keys(), key=str):
                    if (state == 'time'): continue
                    xdata.append( evo['time'] )
                    if self._plotProportions:
                        ytmp = [y/self._systemSize for y in evo[state]]
                        ydata.append(ytmp)
                        yrange = max(1.0, max(ytmp))
                    else:
                        ydata.append( evo[state] )
                        yrange = max(self._systemSize, max(evo[state]))
                    labels.append(state)
                #xdata=[list(np.arange(len(list(evo.values())[0])))]*len(evo.values()), ydata=list(evo.values()), curvelab=list(evo.keys())
                    #plt.plot(evo['time'], evo[state], color=self._colors[state]) #label=state,
                _fig_formatting_2D(xdata=xdata, ydata=ydata, curvelab=labels, curve_replot=False, choose_xrange=(0, self._maxTime), choose_yrange=(0, yrange))
            else: # If realtime-plot mode, draw only the last timestep rather than overlay all
                xdata = []
                ydata = []
                for state in sorted(self._initialState.keys(), key=str):
                    if (state == 'time'): continue
                    xdata.append( evo['time'] )
                    if self._plotProportions:
                        ytmp = [y/self._systemSize for y in evo[state]]
                        ydata.append(ytmp)
                        yrange = max(1.0, max(ytmp))
                    else:
                        ydata.append( evo[state] )
                        yrange = max(self._systemSize, max(evo[state]))
                    ## @todo replot only the last part (rather than all the line)
#                     xdata.append( [i-1,i] )
#                     pop = evo[state]
#                     ydata.append( pop[len(pop)-2:len(pop)] )
                _fig_formatting_2D(xdata=xdata, ydata=ydata, curve_replot=False, choose_xrange=(0, self._maxTime), choose_yrange=(0, yrange))
#                 plt.plot(evo['time'][len(pop)-2:len(pop)], pop[len(pop)-2:len(pop)], color=self._colors[state]) #label=state,
        elif (self._visualisationType == "final"):
            self._initFigure()

            finaldata = []
            labels = []
            colors = []
            for state in sorted(self._initialState.keys(), key=str):
                if (state == 'time'): continue
                if self._plotProportions:
                    finaldata.append( evo[state][len(evo[state])-1]/self._systemSize )
                else:
                    finaldata.append( evo[state][len(evo[state])-1] )
                labels.append(state)
                colors.append(self._colors[state])
#             
#             plt.pie(finaldata, labels=labels, autopct=_make_autopct(piedata), colors=colors) #shadow=True, startangle=90,
            xpos = np.arange(len( self._initialState.keys() ))  # the x locations for the bars
            width = 1       # the width of the bars
            
            plt.bar(xpos, finaldata, width, color=colors) #, yerr=stdev)
            plt.axes().set_xticks(xpos + width / 2)
            plt.axes().set_xticklabels(sorted(self._initialState.keys(), key=str))
        
        # update the figure
        if not self._silent:
            self._figure.canvas.draw()
            
    def _stepSSA(self, currentState): 
        # update transition probabilities accounting for the current state
        probabilitiesOfChange = {}
        for reaction_id, reaction in self._mumotModel._stoichiometry.items():
            prob = self._ratesDict[str(reaction["rate"])]
            numReagents = 0  
            for reactant, re_stoch in reaction.items():
                if reactant == 'rate': continue
                if re_stoch == 'const':
                    reactantOccurencies = 1
                else:
                    reactantOccurencies = re_stoch[0]
                if reactantOccurencies > 0:
                    prob *= currentState[reactant] * reactantOccurencies
                numReagents += reactantOccurencies
            if prob > 0 and numReagents > 1:
                prob /= sum(currentState.values())**( numReagents -1 ) 
            probabilitiesOfChange[reaction_id] = prob
#         for rule in self._reactantsMatrix:
#             prob = sum([a*b for a,b in zip(rule,currentState)])
#             numReagents = sum(x > 0 for x in rule)
#             if numReagents > 1:
#                 prob /= sum(currentState)**( numReagents -1 ) 
#             probabilitiesOfChange.append(prob)
        probSum = sum(probabilitiesOfChange.values())
        if probSum == 0: # no reaction are possible (the execution terminates with this population)
            return (math.inf, currentState)
        # computing when is happening next reaction
        timeInterval = np.random.exponential( 1/probSum );
        
        # Selecting the occurred reaction at random, with probability proportional to each reaction probabilities
        bottom = 0.0
        # Get a random between [0,1) (but we don't want 0!)
        reaction = 0.0
        while (reaction == 0.0):
            reaction = np.random.random_sample()
        # Normalising probOfChange in the range [0,1]
#         probabilitiesOfChange = [pc/probSum for pc in probabilitiesOfChange]
        probabilitiesOfChange = {r_id: pc/probSum for r_id, pc in probabilitiesOfChange.items()}
#         print("Prob of Change: " + str(probabilitiesOfChange))
#         index = -1
#         for i, prob in enumerate(probabilitiesOfChange):
#             if ( reaction >= bottom and reaction < (bottom + prob)):
#                 index = i
#                 break
#             bottom += prob
#         
#         if (index == -1):
#             print("ERROR! Transition not found. Error in the algorithm execution.")
#             sys.exit()
        reaction_id = -1
        for r_id, prob in probabilitiesOfChange.items():
            if reaction >= bottom and reaction < (bottom + prob):
                reaction_id = r_id
                break
            bottom += prob
        
        if (reaction_id == -1):
            print("ERROR! Transition not found. Error in the algorithm execution.")
            sys.exit()
#         print("current state: " + str(currentState))
#         print("triggered change: " + str(self._mumotModel._stoichiometry[reaction_id]))
        # apply the change
#         currentState = list(np.array(currentState) + np.array(self._ruleChanges[index]) )
        for reactant, re_stoch in self._mumotModel._stoichiometry[reaction_id].items():
            if reactant == 'rate' or re_stoch == 'const': continue
            currentState[reactant] += re_stoch[1] - re_stoch[0]
            if (currentState[reactant] < 0):
                print("ERROR! Population size became negative: " + str(currentState) + "; Error in the algorithm execution.")
                sys.exit()
        #print(currentState)
                
        return (timeInterval, currentState)
    
    ## @todo check if it is still running correctly after structural change for View independence from Controller (especially, if initialState is correctly initialised)
    def _singleRun(self, randomSeed):
        # set random seed
        np.random.seed(randomSeed)
        # Create logging structs
        historyState = []
        evo = {}
        evo['time'] = [0]
        for state,pop in self._initialState.items():
            evo[state] = []
            evo[state].append(pop)
         
        t = 0
        currentState = []
        for reactant in sorted(self._mumotModel._reactants|self._mumotModel._constantReactants, key=str):
            currentState.append(self._initialState[reactant])
        historyState.append( [t] + currentState )
 
        while t < self._maxTime:
            timeInterval,currentState = self._stepSSA(currentState)
            # increment time
            t += timeInterval

            # log step
            historyState.append( [t] + currentState )
            for state,pop in zip(sorted(self._mumotModel._reactants|self._mumotModel._constantReactants, key=str), currentState):
                evo[state].append(pop)
            evo['time'].append(t)
                      
#         print("State distribution each timestep: " + str(historyState))
#         print("Temporal evolution per state: " + str(evo))
        return (historyState,evo)
    

## create model from text description
def parseModel(modelDescription):
    ## @todo: add system size to model description
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
    constantReactants = set()
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
                constantReactant = False

                if state == 'A':
                    if token != "+" and token != "->" and token != ":":
                        state = 'B'
                        if '^' in token:
                            raise SyntaxError("Reactants cannot contain '^' :" + token + " in " + rule)
                        reactantCount += 1
                        if (token[0] == '(' and token[-1] == ')'):
                            constantReactant = True
                            token = token.replace('(','')
                            token = token.replace(')','')
                        if token == '\emptyset':
                            constantReactant = True
                            model._constantSystemSize = False
                            token = '1'
                        expr = process_sympy(token)
                        reactantAtoms = expr.atoms()
                        if len(reactantAtoms) != 1:
                            raise SyntaxError("Non-singleton symbol set in token " + token +" in rule " + rule)
                        for reactant in reactantAtoms:
                            pass # this loop extracts element from singleton set
                        if constantReactant:
                            if reactant not in constantReactants and token != '1':
                                constantReactants.add(reactant)                            
                        else:
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
                        if (token[0] == '(' and token[-1] == ')'):
                            constantReactant = True
                            token = token.replace('(','')
                            token = token.replace(')','')
                        if token == '\emptyset':
                            model._constantSystemSize = False
                            constantReactant = True
                            token = '1'
                        expr = process_sympy(token)
                        reactantAtoms = expr.atoms()
                        if len(reactantAtoms) != 1:
                            raise SyntaxError("Non-singleton symbol set in token " + token +" in rule " + rule)
                        for reactant in reactantAtoms:
                            pass # this loop extracts element from singleton set
                        if constantReactant:
                            if reactant not in constantReactants and token != '1':
                                constantReactants.add(reactant)                            
                        else:
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
    model._constantReactants = constantReactants
    # check intersection of reactants and constantReactants is empty
    intersect = model._reactants.intersection(model._constantReactants) 
    if len(intersect) != 0:
        raise SyntaxError("Following reactants defined as both constant and variable: " + str(intersect))
    model._rates = rates
    model._equations = _deriveODEsFromRules(model._reactants, model._rules)
    model._ratesLaTeX = {}
    rates = map(latex, list(model._rates))
    for (rate, latexStr) in zip(model._rates, rates):
        model._ratesLaTeX[repr(rate)] = latexStr
    constantReactants = map(latex, list(model._constantReactants))
    for (reactant, latexStr) in zip(model._constantReactants, constantReactants):
        model._ratesLaTeX[repr(reactant)] = '(' + latexStr + ')'    
    
    model._stoichiometry = _getStoichiometry(model._rules, model._constantReactants)
    
    return model

def _deriveODEsFromRules(reactants, rules):
    ## @todo: replace with principled derivation via Master Equation and van Kampen expansion
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

## produces dictionary with stoichiometry of all reactions with key ReactionNr
# ReactionNr represents another dictionary with reaction rate, reactants and their stoichiometry
## @todo: shall this become a model method?
def _getStoichiometry(rules, const_reactants):
    stoich = {}
    ReactionNr = numbered_symbols(prefix='Reaction ', cls=Symbol, start=1)
    for rule in rules:
        reactDict = {'rate': rule.rate}
        for reactant in rule.lhsReactants:
            if reactant != 1:
                if reactant in const_reactants:
                    reactDict[reactant] = 'const'
                else:
                    reactDict[reactant] = [rule.lhsReactants.count(reactant), rule.rhsReactants.count(reactant)]
        for reactant in rule.rhsReactants:
            if reactant != 1:
                if reactant not in rule.lhsReactants:
                    if reactant not in const_reactants:
                        reactDict[reactant] = [rule.lhsReactants.count(reactant), rule.rhsReactants.count(reactant)]
        stoich[ReactionNr.__next__()] = reactDict
        
    return stoich


## derivation of the Master equation, returns dictionary used in showMasterEquation
def _deriveMasterEquation(stoichiometry):
    substring = None
    P, E_op, x, y, v, w, t, m = symbols('P E_op x y v w t m')
    V = Symbol('V', real=True, constant=True)
    stoich = stoichiometry
    nvec = []
    for key1 in stoich:
        for key2 in stoich[key1]:
            if key2 != 'rate' and stoich[key1][key2] != 'const':
                if not key2 in nvec:
                    nvec.append(key2)
                if len(stoich[key1][key2]) == 3:
                    substring = stoich[key1][key2][2]
    nvec = sorted(nvec, key=default_sort_key)
    
    assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'
    
    rhs = 0
    sol_dict_rhs = {}
    f = lambdify((x(y, v-w)), x(y, v-w), modules='sympy')
    g = lambdify((x, y, v), (factorial(x)/factorial(x-y))/v**y, modules='sympy')
    for key1 in stoich:
        prod1 = 1
        prod2 = 1
        rate_fact = 1
        for key2 in stoich[key1]:
            if key2 != 'rate' and stoich[key1][key2] != 'const':
                prod1 *= f(E_op(key2, stoich[key1][key2][0]-stoich[key1][key2][1]))
                prod2 *= g(key2, stoich[key1][key2][0], V)
            if stoich[key1][key2] == 'const':
                rate_fact *= key2/V
        if len(nvec)==2:
            sol_dict_rhs[key1] = (prod1, simplify(prod2*V), P(nvec[0], nvec[1], t), stoich[key1]['rate']*rate_fact)
        elif len(nvec)==3:
            sol_dict_rhs[key1] = (prod1, simplify(prod2*V), P(nvec[0], nvec[1], nvec[2], t), stoich[key1]['rate']*rate_fact)
        else:
            sol_dict_rhs[key1] = (prod1, simplify(prod2*V), P(nvec[0], nvec[1], nvec[2], nvec[3], t), stoich[key1]['rate']*rate_fact)

    return sol_dict_rhs, substring


## Function returning the left-hand side and right-hand side of van Kampen expansion    
def _doVanKampenExpansion(rhs, stoich):
    P, E_op, x, y, v, w, t, m = symbols('P E_op x y v w t m')
    V = Symbol('V', real=True, constant=True)
    nvec = []
    nconstvec = []
    for key1 in stoich:
        for key2 in stoich[key1]:
            if key2 != 'rate' and stoich[key1][key2] != 'const':
                if not key2 in nvec:
                    nvec.append(key2)
            elif key2 != 'rate' and stoich[key1][key2] == 'const':
                if not key2 in nconstvec:
                    nconstvec.append(key2)
                    
    nvec = sorted(nvec, key=default_sort_key)
    assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'
    
    NoiseDict = {}
    PhiDict = {}
    PhiConstDict = {}
    
    for kk in range(len(nvec)):
        NoiseDict[nvec[kk]] = Symbol('eta_'+str(nvec[kk]))
        PhiDict[nvec[kk]] = Symbol('Phi_'+str(nvec[kk]))
        
    for kk in range(len(nconstvec)):
        PhiConstDict[nconstvec[kk]] = V*Symbol('Phi_'+str(nconstvec[kk]))

        
    rhs_dict, substring = rhs(stoich)
    rhs_vKE = 0
    
    if len(nvec)==2:
        lhs_vKE = (Derivative(P(nvec[0], nvec[1],t),t).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]]})
                  - sqrt(V)*Derivative(PhiDict[nvec[0]],t)*Derivative(P(nvec[0], nvec[1],t),nvec[0]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]]})
                  - sqrt(V)*Derivative(PhiDict[nvec[1]],t)*Derivative(P(nvec[0], nvec[1],t),nvec[1]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]]}))

        for key in rhs_dict:
            op = rhs_dict[key][0].subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]]})
            func1 = rhs_dict[key][1].subs({nvec[0]: V*PhiDict[nvec[0]]+sqrt(V)*NoiseDict[nvec[0]], nvec[1]: V*PhiDict[nvec[1]]+sqrt(V)*NoiseDict[nvec[1]]})
            func2 = rhs_dict[key][2].subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]]})
            func = func1*func2
            if len(op.args[0].args) ==0:
                term = (op*func).subs({op*func: func + op.args[1]/sqrt(V)*Derivative(func, op.args[0]) + op.args[1]**2/(2*V)*Derivative(func, op.args[0], op.args[0]) })
                
            else:
                term = (op.args[1]*func).subs({op.args[1]*func: func + op.args[1].args[1]/sqrt(V)*Derivative(func, op.args[1].args[0]) 
                                       + op.args[1].args[1]**2/(2*V)*Derivative(func, op.args[1].args[0], op.args[1].args[0])})
                term = (op.args[0]*term).subs({op.args[0]*term: term + op.args[0].args[1]/sqrt(V)*Derivative(term, op.args[0].args[0]) 
                                       + op.args[0].args[1]**2/(2*V)*Derivative(term, op.args[0].args[0], op.args[0].args[0])})
            #term_num, term_denom = term.as_numer_denom()
            rhs_vKE += rhs_dict[key][3].subs(PhiConstDict)*(term.doit() - func)
    elif len(nvec)==3:
        lhs_vKE = (Derivative(P(nvec[0], nvec[1], nvec[2], t), t).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]]})
                  - sqrt(V)*Derivative(PhiDict[nvec[0]],t)*Derivative(P(nvec[0], nvec[1], nvec[2], t), nvec[0]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]]})
                  - sqrt(V)*Derivative(PhiDict[nvec[1]],t)*Derivative(P(nvec[0], nvec[1], nvec[2], t), nvec[1]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]]})
                  - sqrt(V)*Derivative(PhiDict[nvec[2]],t)*Derivative(P(nvec[0], nvec[1], nvec[2], t), nvec[2]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]]}))
        rhs_dict, substring = rhs(stoich)
        rhs_vKE = 0
        for key in rhs_dict:
            op = rhs_dict[key][0].subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]]})
            func1 = rhs_dict[key][1].subs({nvec[0]: V*PhiDict[nvec[0]]+sqrt(V)*NoiseDict[nvec[0]], nvec[1]: V*PhiDict[nvec[1]]+sqrt(V)*NoiseDict[nvec[1]], nvec[2]: V*PhiDict[nvec[2]]+sqrt(V)*NoiseDict[nvec[2]]})
            func2 = rhs_dict[key][2].subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]]})
            func = func1*func2
            if len(op.args[0].args) ==0:
                term = (op*func).subs({op*func: func + op.args[1]/sqrt(V)*Derivative(func, op.args[0]) + op.args[1]**2/(2*V)*Derivative(func, op.args[0], op.args[0]) })
                
            elif len(op.args) ==2:
                term = (op.args[1]*func).subs({op.args[1]*func: func + op.args[1].args[1]/sqrt(V)*Derivative(func, op.args[1].args[0]) 
                                       + op.args[1].args[1]**2/(2*V)*Derivative(func, op.args[1].args[0], op.args[1].args[0])})
                term = (op.args[0]*term).subs({op.args[0]*term: term + op.args[0].args[1]/sqrt(V)*Derivative(term, op.args[0].args[0]) 
                                       + op.args[0].args[1]**2/(2*V)*Derivative(term, op.args[0].args[0], op.args[0].args[0])})
            elif len(op.args) ==3:
                term = (op.args[2]*func).subs({op.args[2]*func: func + op.args[2].args[1]/sqrt(V)*Derivative(func, op.args[2].args[0]) 
                                       + op.args[2].args[1]**2/(2*V)*Derivative(func, op.args[2].args[0], op.args[2].args[0])})
                term = (op.args[1]*term).subs({op.args[1]*term: term + op.args[1].args[1]/sqrt(V)*Derivative(term, op.args[1].args[0]) 
                                       + op.args[1].args[1]**2/(2*V)*Derivative(term, op.args[1].args[0], op.args[1].args[0])})
                term = (op.args[0]*term).subs({op.args[0]*term: term + op.args[0].args[1]/sqrt(V)*Derivative(term, op.args[0].args[0]) 
                                       + op.args[0].args[1]**2/(2*V)*Derivative(term, op.args[0].args[0], op.args[0].args[0])})
            else:
                print('Something went wrong!')
            rhs_vKE += rhs_dict[key][3].subs(PhiConstDict)*(term.doit() - func)    
    else:
        lhs_vKE = (Derivative(P(nvec[0], nvec[1], nvec[2], nvec[3], t), t).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]], nvec[3]: NoiseDict[nvec[3]]})
                  - sqrt(V)*Derivative(PhiDict[nvec[0]],t)*Derivative(P(nvec[0], nvec[1], nvec[2], nvec[3], t), nvec[0]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]], nvec[3]: NoiseDict[nvec[3]]})
                  - sqrt(V)*Derivative(PhiDict[nvec[1]],t)*Derivative(P(nvec[0], nvec[1], nvec[2], nvec[3], t), nvec[1]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]], nvec[3]: NoiseDict[nvec[3]]})
                  - sqrt(V)*Derivative(PhiDict[nvec[2]],t)*Derivative(P(nvec[0], nvec[1], nvec[2], nvec[3], t), nvec[2]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]], nvec[3]: NoiseDict[nvec[3]]})
                  - sqrt(V)*Derivative(PhiDict[nvec[3]],t)*Derivative(P(nvec[0], nvec[1], nvec[2], nvec[3], t), nvec[3]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]], nvec[3]: NoiseDict[nvec[3]]}))
        rhs_dict, substring = rhs(stoich)
        rhs_vKE = 0
        for key in rhs_dict:
            op = rhs_dict[key][0].subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]], nvec[3]: NoiseDict[nvec[3]]})
            func1 = rhs_dict[key][1].subs({nvec[0]: V*PhiDict[nvec[0]]+sqrt(V)*NoiseDict[nvec[0]], nvec[1]: V*PhiDict[nvec[1]]+sqrt(V)*NoiseDict[nvec[1]], nvec[2]: V*PhiDict[nvec[2]]+sqrt(V)*NoiseDict[nvec[2]], nvec[3]: V*PhiDict[nvec[3]]+sqrt(V)*NoiseDict[nvec[3]]})
            func2 = rhs_dict[key][2].subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]], nvec[3]: NoiseDict[nvec[3]]})
            func = func1*func2
            if len(op.args[0].args) ==0:
                term = (op*func).subs({op*func: func + op.args[1]/sqrt(V)*Derivative(func, op.args[0]) + op.args[1]**2/(2*V)*Derivative(func, op.args[0], op.args[0]) })
                
            elif len(op.args) ==2:
                term = (op.args[1]*func).subs({op.args[1]*func: func + op.args[1].args[1]/sqrt(V)*Derivative(func, op.args[1].args[0]) 
                                       + op.args[1].args[1]**2/(2*V)*Derivative(func, op.args[1].args[0], op.args[1].args[0])})
                term = (op.args[0]*term).subs({op.args[0]*term: term + op.args[0].args[1]/sqrt(V)*Derivative(term, op.args[0].args[0]) 
                                       + op.args[0].args[1]**2/(2*V)*Derivative(term, op.args[0].args[0], op.args[0].args[0])})
            elif len(op.args) ==3:
                term = (op.args[2]*func).subs({op.args[2]*func: func + op.args[2].args[1]/sqrt(V)*Derivative(func, op.args[2].args[0]) 
                                       + op.args[2].args[1]**2/(2*V)*Derivative(func, op.args[2].args[0], op.args[2].args[0])})
                term = (op.args[1]*term).subs({op.args[1]*term: term + op.args[1].args[1]/sqrt(V)*Derivative(term, op.args[1].args[0]) 
                                       + op.args[1].args[1]**2/(2*V)*Derivative(term, op.args[1].args[0], op.args[1].args[0])})
                term = (op.args[0]*term).subs({op.args[0]*term: term + op.args[0].args[1]/sqrt(V)*Derivative(term, op.args[0].args[0]) 
                                       + op.args[0].args[1]**2/(2*V)*Derivative(term, op.args[0].args[0], op.args[0].args[0])})
            elif len(op.args) ==4:
                term = (op.args[3]*func).subs({op.args[3]*func: func + op.args[3].args[1]/sqrt(V)*Derivative(func, op.args[3].args[0]) 
                                       + op.args[3].args[1]**2/(2*V)*Derivative(func, op.args[3].args[0], op.args[3].args[0])})
                term = (op.args[2]*term).subs({op.args[2]*term: term + op.args[2].args[1]/sqrt(V)*Derivative(term, op.args[2].args[0]) 
                                       + op.args[2].args[1]**2/(2*V)*Derivative(term, op.args[2].args[0], op.args[2].args[0])})
                term = (op.args[1]*term).subs({op.args[1]*term: term + op.args[1].args[1]/sqrt(V)*Derivative(term, op.args[1].args[0]) 
                                       + op.args[1].args[1]**2/(2*V)*Derivative(term, op.args[1].args[0], op.args[1].args[0])})
                term = (op.args[0]*term).subs({op.args[0]*term: term + op.args[0].args[1]/sqrt(V)*Derivative(term, op.args[0].args[0]) 
                                       + op.args[0].args[1]**2/(2*V)*Derivative(term, op.args[0].args[0], op.args[0].args[0])})
            else:
                print('Something went wrong!')
            rhs_vKE += rhs_dict[key][3].subs(PhiConstDict)*(term.doit() - func)
    
    return rhs_vKE.expand(), lhs_vKE, substring

## creates list of dictionaries where the key is the system size order
#def _get_orderedLists_vKE( _getStoichiometry,rules):
def _get_orderedLists_vKE(stoich):
    V = Symbol('V', real=True, constant=True)
    stoichiometry = stoich
    rhs_vke, lhs_vke, substring = _doVanKampenExpansion(_deriveMasterEquation, stoichiometry)
    Vlist_lhs=[]
    Vlist_rhs=[]
    for jj in range(len(rhs_vke.args)):
        try:
            Vlist_rhs.append(simplify(rhs_vke.args[jj]).collect(V, evaluate=False))
        except NotImplementedError:
            prod=1
            for nn in range(len(rhs_vke.args[jj].args)-1):
                prod*=rhs_vke.args[jj].args[nn]
            tempdict=prod.collect(V, evaluate=False)
            for key in tempdict:
                Vlist_rhs.append({key: prod/key*rhs_vke.args[jj].args[-1]})
    
    for jj in range(len(lhs_vke.args)):
        try:
            Vlist_lhs.append(simplify(lhs_vke.args[jj]).collect(V, evaluate=False))
        except NotImplementedError:
            prod=1
            for nn in range(len(lhs_vke.args[jj].args)-1):
                prod*=lhs_vke.args[jj].args[nn]
            tempdict=prod.collect(V, evaluate=False)
            for key in tempdict:
                Vlist_lhs.append({key: prod/key*lhs_vke.args[jj].args[-1]})
    return Vlist_lhs, Vlist_rhs, substring


## Function that returns the Fokker-Planck equation
def _getFokkerPlanckEquation(_get_orderedLists_vKE, stoich):
    P, t = symbols('P t')
    V = Symbol('V', real=True, constant=True)
    Vlist_lhs, Vlist_rhs, substring = _get_orderedLists_vKE(stoich)
    rhsFPE=0
    lhsFPE=0
    for kk in range(len(Vlist_rhs)):
        for key in Vlist_rhs[kk]:
            if key==1:
                rhsFPE += Vlist_rhs[kk][key]  
    for kk in range(len(Vlist_lhs)):
        for key in Vlist_lhs[kk]:
            if key==1:
                lhsFPE += Vlist_lhs[kk][key]            
        
    FPE = lhsFPE-rhsFPE
    
    nvec = []
    for key1 in stoich:
        for key2 in stoich[key1]:
            if key2 != 'rate' and stoich[key1][key2] != 'const':
                if not key2 in nvec:
                    nvec.append(key2)
    nvec = sorted(nvec, key=default_sort_key)
    assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'
    
    NoiseDict = {}
    for kk in range(len(nvec)):
        NoiseDict[nvec[kk]] = Symbol('eta_'+str(nvec[kk]))
    
    if len(Vlist_lhs)-1 == 2:
        SOL_FPE = solve(FPE, Derivative(P(NoiseDict[nvec[0]],NoiseDict[nvec[1]],t),t), dict=True)[0]
    elif len(Vlist_lhs)-1 == 3:
        SOL_FPE = solve(FPE, Derivative(P(NoiseDict[nvec[0]],NoiseDict[nvec[1]],NoiseDict[nvec[2]],t),t), dict=True)[0]
    elif len(Vlist_lhs)-1 == 4:
        SOL_FPE = solve(FPE, Derivative(P(NoiseDict[nvec[0]],NoiseDict[nvec[1]],NoiseDict[nvec[2]],NoiseDict[nvec[3]],t),t), dict=True)[0]
    else:
        print('Not implemented yet.')
           
    return SOL_FPE, substring


## calculates noise in the system
# returns equations of motion for noise
def _getNoiseEOM(_getFokkerPlanckEquation, _get_orderedLists_vKE, stoich):
    P, M_1, M_2, t = symbols('P M_1 M_2 t')
    #
    #A,B, alpha, beta, gamma = symbols('A B alpha beta gamma')
    #custom_stoich= {'reaction1': {'rate': alpha, A: [0,1]}, 'reaction2': {'rate': gamma, A: [2,0], B: [0,1]},
    #                 'reaction3': {'rate': beta, B: [1,0]}}
    #stoich = custom_stoich
    # 
    nvec = []
    for key1 in stoich:
        for key2 in stoich[key1]:
            if key2 != 'rate' and stoich[key1][key2] != 'const':
                if key2 not in nvec:
                    nvec.append(key2)
    nvec = sorted(nvec, key=default_sort_key)
    assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'

    NoiseDict = {}
    for kk in range(len(nvec)):
        NoiseDict[nvec[kk]] = Symbol('eta_'+str(nvec[kk]))
    FPEdict, substring = _getFokkerPlanckEquation(_get_orderedLists_vKE, stoich)
    
    NoiseSub1stOrder = {}
    NoiseSub2ndOrder = {}

    if len(NoiseDict)==2:
        Pdim = P(NoiseDict[nvec[0]],NoiseDict[nvec[1]],t)
    elif len(NoiseDict)==3:
        Pdim = P(NoiseDict[nvec[0]],NoiseDict[nvec[1]],NoiseDict[nvec[2]],t)
    else:
        Pdim = P(NoiseDict[nvec[0]],NoiseDict[nvec[1]],NoiseDict[nvec[2]],NoiseDict[nvec[3]],t)
        
    for noise1 in NoiseDict:
        NoiseSub1stOrder[NoiseDict[noise1]*Pdim] = M_1(NoiseDict[noise1])
        for noise2 in NoiseDict:
            for noise3 in NoiseDict:
                key = NoiseDict[noise1]*NoiseDict[noise2]*Derivative(Pdim,NoiseDict[noise3])
                if key not in NoiseSub1stOrder:
                    if NoiseDict[noise1] == NoiseDict[noise2] and NoiseDict[noise3] == NoiseDict[noise1]:
                        NoiseSub1stOrder[key] = -2*M_1(NoiseDict[noise1])
                    elif NoiseDict[noise1] != NoiseDict[noise2] and NoiseDict[noise3] == NoiseDict[noise1]:
                        NoiseSub1stOrder[key] = -M_1(NoiseDict[noise2])
                    elif NoiseDict[noise1] != NoiseDict[noise2] and NoiseDict[noise3] == NoiseDict[noise2]:
                        NoiseSub1stOrder[key] = -M_1(NoiseDict[noise1])
                    elif NoiseDict[noise1] != NoiseDict[noise3] and NoiseDict[noise3] != NoiseDict[noise2]:
                        NoiseSub1stOrder[key] = 0
                    else:
                        NoiseSub1stOrder[key] = 0 
                key2 = NoiseDict[noise1]*Derivative(Pdim,NoiseDict[noise2],NoiseDict[noise3])
                if key2 not in NoiseSub1stOrder:
                    NoiseSub1stOrder[key2] = 0   
    
    for noise1 in NoiseDict:
        for noise2 in NoiseDict:
            key = NoiseDict[noise1]*NoiseDict[noise2]*Pdim
            if key not in NoiseSub2ndOrder:
                NoiseSub2ndOrder[key] = M_2(NoiseDict[noise1]*NoiseDict[noise2])
            for noise3 in NoiseDict:
                for noise4 in NoiseDict:
                    key2 = NoiseDict[noise1]*NoiseDict[noise2]*NoiseDict[noise3]*Derivative(Pdim,NoiseDict[noise4])
                    if key2 not in NoiseSub2ndOrder:
                        if noise1 == noise2 and noise2 == noise3 and noise3 == noise4:
                            NoiseSub2ndOrder[key2] = -3*M_2(NoiseDict[noise1]*NoiseDict[noise1])
                        elif noise1 == noise2 and noise2 != noise3 and noise1 == noise4:
                            NoiseSub2ndOrder[key2] = -2*M_2(NoiseDict[noise1]*NoiseDict[noise3])
                        elif noise1 == noise2 and noise2 != noise3 and noise3 == noise4:
                            NoiseSub2ndOrder[key2] = -M_2(NoiseDict[noise1]*NoiseDict[noise2])
                        elif noise1 != noise2 and noise2 == noise3 and noise1 == noise4:
                            NoiseSub2ndOrder[key2] = -M_2(NoiseDict[noise2]*NoiseDict[noise3])
                        elif noise1 != noise2 and noise2 == noise3 and noise3 == noise4:
                            NoiseSub2ndOrder[key2] = -2*M_2(NoiseDict[noise1]*NoiseDict[noise2])
                        elif noise1 != noise2 and noise2 != noise3 and noise1 != noise3:
                            if noise1 == noise4:
                                NoiseSub2ndOrder[key2] = -M_2(NoiseDict[noise2]*NoiseDict[noise3])
                            elif noise2 == noise4:
                                NoiseSub2ndOrder[key2] = -M_2(NoiseDict[noise1]*NoiseDict[noise3])
                            elif noise3 == noise4:
                                NoiseSub2ndOrder[key2] = -M_2(NoiseDict[noise1]*NoiseDict[noise2])
                            else:
                                NoiseSub2ndOrder[key2] = 0
                        else:
                            NoiseSub2ndOrder[key2] = 0
                    key3 = NoiseDict[noise1]*NoiseDict[noise2]*Derivative(Pdim,NoiseDict[noise3],NoiseDict[noise4])
                    if key3 not in NoiseSub2ndOrder:
                        if noise1 == noise3 and noise2 == noise4:
                            if noise1 == noise2:
                                NoiseSub2ndOrder[key3] = 2
                            else:
                                 NoiseSub2ndOrder[key3] = 1
                        elif noise1 == noise4 and noise2 == noise3:
                            if noise1 == noise2:
                                NoiseSub2ndOrder[key3] = 2
                            else:
                                NoiseSub2ndOrder[key3] = 1
                        else:
                            NoiseSub2ndOrder[key3] = 0
    NoiseSubs1stOrder = {}                   
    EQsys1stOrdMom = []
    EOM_1stOrderMom = {}
    for fpe_lhs in FPEdict: 
        for noise in NoiseDict:
            eq1stOrderMoment = (NoiseDict[noise]*FPEdict[fpe_lhs]).expand() 
            eq1stOrderMoment = eq1stOrderMoment.subs(NoiseSub1stOrder)
            if len(NoiseDict)==2:
                eq1stOrderMoment = collect(eq1stOrderMoment, M_1(NoiseDict[nvec[0]]))
                eq1stOrderMoment = collect(eq1stOrderMoment, M_1(NoiseDict[nvec[1]]))
            elif len(NoiseDict)==3:
                eq1stOrderMoment = collect(eq1stOrderMoment, M_1(NoiseDict[nvec[0]]))
                eq1stOrderMoment = collect(eq1stOrderMoment, M_1(NoiseDict[nvec[1]]))
                eq1stOrderMoment = collect(eq1stOrderMoment, M_1(NoiseDict[nvec[2]]))
            else:
                eq1stOrderMoment = collect(eq1stOrderMoment, M_1(NoiseDict[nvec[0]]))
                eq1stOrderMoment = collect(eq1stOrderMoment, M_1(NoiseDict[nvec[1]]))
                eq1stOrderMoment = collect(eq1stOrderMoment, M_1(NoiseDict[nvec[2]]))
                eq1stOrderMoment = collect(eq1stOrderMoment, M_1(NoiseDict[nvec[3]]))
            EQsys1stOrdMom.append(eq1stOrderMoment)
            if M_1(NoiseDict[noise]) not in EOM_1stOrderMom:
                EOM_1stOrderMom[M_1(NoiseDict[noise])] = eq1stOrderMoment
                NoiseSubs1stOrder[M_1(NoiseDict[noise])] = r'\left< \vphantom{Dg}\right.' +latex(NoiseDict[noise]) + r'\left. \vphantom{Dg}\right>'
    
    NoiseSubs2ndOrder = {}
    EQsys2ndOrdMom = []
    EOM_2ndOrderMom = {}
    for fpe_lhs in FPEdict: 
        for noise1 in NoiseDict:
            for noise2 in NoiseDict:
                eq2ndOrderMoment = (NoiseDict[noise1]*NoiseDict[noise2]*FPEdict[fpe_lhs]).expand() 
                eq2ndOrderMoment = eq2ndOrderMoment.subs(NoiseSub2ndOrder)
                eq2ndOrderMoment = eq2ndOrderMoment.subs(NoiseSub1stOrder)
                #eq2ndOrderMoment = eq2ndOrderMoment.subs(SOL_1stOrderMom[0])
                if len(NoiseDict)==2:
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[1]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]]))
                elif len(NoiseDict)==3:
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[1]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[2]]*NoiseDict[nvec[2]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[2]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[2]]))
                else:
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[1]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[2]]*NoiseDict[nvec[2]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[3]]*NoiseDict[nvec[3]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[2]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[3]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[2]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[3]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[2]]*NoiseDict[nvec[3]]))
                    
                eq2ndOrderMoment = eq2ndOrderMoment.simplify()
                if eq2ndOrderMoment not in EQsys2ndOrdMom:
                    EQsys2ndOrdMom.append(eq2ndOrderMoment)
                if M_2(NoiseDict[noise1]*NoiseDict[noise2]) not in EOM_2ndOrderMom:
                    EOM_2ndOrderMom[M_2(NoiseDict[noise1]*NoiseDict[noise2])] = eq2ndOrderMoment
                    NoiseSubs2ndOrder[M_2(NoiseDict[noise1]*NoiseDict[noise2])] = r'\left< \vphantom{Dg}\right.' +latex(NoiseDict[noise1]*NoiseDict[noise2]) + r'\left. \vphantom{Dg}\right>' 
      
    return EQsys1stOrdMom, EOM_1stOrderMom, NoiseSubs1stOrder, EQsys2ndOrdMom, EOM_2ndOrderMom, NoiseSubs2ndOrder 


## calculates noise in the system
# returns analytical solution for stationary noise
def _getNoiseStationarySol(_getNoiseEOM, _getFokkerPlanckEquation, _get_orderedLists_vKE, stoich):
    P, M_1, M_2, t = symbols('P M_1 M_2 t')
    
    EQsys1stOrdMom, EOM_1stOrderMom, NoiseSubs1stOrder, EQsys2ndOrdMom, EOM_2ndOrderMom, NoiseSubs2ndOrder = _getNoiseEOM(_getFokkerPlanckEquation, _get_orderedLists_vKE, stoich)
    
    nvec = []
    for key1 in stoich:
        for key2 in stoich[key1]:
            if key2 != 'rate' and stoich[key1][key2] != 'const':
                if not key2 in nvec:
                    nvec.append(key2)
    nvec = sorted(nvec, key=default_sort_key)
    assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'

    NoiseDict = {}
    for kk in range(len(nvec)):
        NoiseDict[nvec[kk]] = Symbol('eta_'+str(nvec[kk]))
            
    if len(NoiseDict)==2:
        SOL_1stOrderMom = solve(EQsys1stOrdMom, [M_1(NoiseDict[nvec[0]]),M_1(NoiseDict[nvec[1]])], dict=True)
    elif len(NoiseDict)==3:
        SOL_1stOrderMom = solve(EQsys1stOrdMom, [M_1(NoiseDict[nvec[0]]),M_1(NoiseDict[nvec[1]]),M_1(NoiseDict[nvec[2]])], dict=True)
    else:
        SOL_1stOrderMom = solve(EQsys1stOrdMom, [M_1(NoiseDict[nvec[0]]),M_1(NoiseDict[nvec[1]]),M_1(NoiseDict[nvec[2]]),M_1(NoiseDict[nvec[3]])], dict=True)
    
    if len(SOL_1stOrderMom[0]) != len(NoiseDict):
        print('Solution for 1st order noise moments NOT unique!')
        return None, None, None, None
                    
    SOL_2ndOrdMomDict = {} 
    if len(NoiseDict)==2:
        SOL_2ndOrderMom = list(linsolve(EQsys2ndOrdMom, [M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]]), 
                                                         M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[1]]), 
                                                         M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]])]))[0] #only one set of solutions (if any) in linear system of equations
        
        if M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]]) in SOL_2ndOrderMom:
            print('Solution for 2nd order noise moments NOT unique!')
            return None, None, None, None
        
#             ZsubDict = {M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]]): 0}
#             SOL_2ndOrderMomMod = []
#             for nn in range(len(SOL_2ndOrderMom)):
#                 SOL_2ndOrderMomMod.append(SOL_2ndOrderMom[nn].subs(ZsubDict))
#             SOL_2ndOrderMom = SOL_2ndOrderMomMod
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]])] = SOL_2ndOrderMom[0]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[1]])] = SOL_2ndOrderMom[1]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]])] = SOL_2ndOrderMom[2]
    
    elif len(NoiseDict)==3:
        SOL_2ndOrderMom = list(linsolve(EQsys2ndOrdMom, [M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]]), 
                                                         M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[1]]), 
                                                         M_2(NoiseDict[nvec[2]]*NoiseDict[nvec[2]]), 
                                                         M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]]), 
                                                         M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[2]]), 
                                                         M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[2]])]))[0] #only one set of solutions (if any) in linear system of equations; hence index [0]
#         ZsubDict = {}
        for noise1 in NoiseDict:
            for noise2 in NoiseDict:
                if M_2(NoiseDict[noise1]*NoiseDict[noise2]) in SOL_2ndOrderMom:
                    print('Solution for 2nd order noise moments NOT unique!')
                    return None, None, None, None
#                     ZsubDict[M_2(NoiseDict[noise1]*NoiseDict[noise2])] = 0
#         if len(ZsubDict) > 0:
#             SOL_2ndOrderMomMod = []
#             for nn in range(len(SOL_2ndOrderMom)):
#                 SOL_2ndOrderMomMod.append(SOL_2ndOrderMom[nn].subs(ZsubDict))
#         SOL_2ndOrderMom = SOL_2ndOrderMomMod
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]])] = SOL_2ndOrderMom[0]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[1]])] = SOL_2ndOrderMom[1]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[2]]*NoiseDict[nvec[2]])] = SOL_2ndOrderMom[2]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]])] = SOL_2ndOrderMom[3]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[2]])] = SOL_2ndOrderMom[4]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[2]])] = SOL_2ndOrderMom[5]
        
    else:
        SOL_2ndOrderMom = list(linsolve(EQsys2ndOrdMom, [M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]]), 
                                                         M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[1]]), 
                                                         M_2(NoiseDict[nvec[2]]*NoiseDict[nvec[2]]),
                                                         M_2(NoiseDict[nvec[3]]*NoiseDict[nvec[3]]), 
                                                         M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]]), 
                                                         M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[2]]), 
                                                         M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[3]]),
                                                         M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[2]]),
                                                         M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[3]]),
                                                         M_2(NoiseDict[nvec[2]]*NoiseDict[nvec[3]])]))[0] #only one set of solutions (if any) in linear system of equations; hence index [0]
#         ZsubDict = {}
        for noise1 in NoiseDict:
            for noise2 in NoiseDict:
                if M_2(NoiseDict[noise1]*NoiseDict[noise2]) in SOL_2ndOrderMom:
                    print('Solution for 2nd order noise moments NOT unique!')
                    return None, None, None, None
#                     ZsubDict[M_2(NoiseDict[noise1]*NoiseDict[noise2])] = 0
#         if len(ZsubDict) > 0:
#             SOL_2ndOrderMomMod = []
#             for nn in range(len(SOL_2ndOrderMom)):
#                 SOL_2ndOrderMomMod.append(SOL_2ndOrderMom[nn].subs(ZsubDict))
#         SOL_2ndOrderMom = SOL_2ndOrderMomMod
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]])] = SOL_2ndOrderMom[0]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[1]])] = SOL_2ndOrderMom[1]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[2]]*NoiseDict[nvec[2]])] = SOL_2ndOrderMom[2]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[3]]*NoiseDict[nvec[3]])] = SOL_2ndOrderMom[3]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]])] = SOL_2ndOrderMom[4]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[2]])] = SOL_2ndOrderMom[5]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[3]])] = SOL_2ndOrderMom[6]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[2]])] = SOL_2ndOrderMom[7]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[3]])] = SOL_2ndOrderMom[8]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[2]]*NoiseDict[nvec[3]])] = SOL_2ndOrderMom[9]    
      
    return SOL_1stOrderMom[0], NoiseSubs1stOrder, SOL_2ndOrdMomDict, NoiseSubs2ndOrder 
 

# def _getNoise(_getFokkerPlanckEquation, _get_orderedLists_vKE, stoich):
#     P, M_1, M_2, t = symbols('P M_1 M_2 t')
#     
#     nvec = []
#     for key1 in stoich:
#         for key2 in stoich[key1]:
#             if not key2 == 'rate':
#                 if not key2 in nvec:
#                     nvec.append(key2)
#     nvec = sorted(nvec, key=default_sort_key)
#     assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'
# 
#     NoiseDict = {}
#     for kk in range(len(nvec)):
#         NoiseDict[nvec[kk]] = Symbol('eta_'+str(nvec[kk]))
#     FPEdict, substring = _getFokkerPlanckEquation(_get_orderedLists_vKE, stoich)
#     print(len(NoiseDict))
#     
#     NoiseSub1stOrder = {}
#     NoiseSub2ndOrder = {}
#     
#     if len(NoiseDict)==2:
#         Pdim = P(NoiseDict[nvec[0]],NoiseDict[nvec[1]],t)
#     elif len(NoiseDict)==3:
#         Pdim = P(NoiseDict[nvec[0]],NoiseDict[nvec[1]],NoiseDict[nvec[2]],t)
#     else:
#         Pdim = P(NoiseDict[nvec[0]],NoiseDict[nvec[1]],NoiseDict[nvec[2]],NoiseDict[nvec[3]],t)
#         
#     for noise1 in NoiseDict:
#         NoiseSub1stOrder[NoiseDict[noise1]*Pdim] = M_1(NoiseDict[noise1])
#         for noise2 in NoiseDict:
#             for noise3 in NoiseDict:
#                 key = NoiseDict[noise1]*NoiseDict[noise2]*Derivative(Pdim,NoiseDict[noise3])
#                 if key not in NoiseSub1stOrder:
#                     if NoiseDict[noise1] == NoiseDict[noise2] and NoiseDict[noise3] == NoiseDict[noise1]:
#                         NoiseSub1stOrder[key] = -2*M_1(NoiseDict[noise1])
#                     elif NoiseDict[noise1] != NoiseDict[noise2] and NoiseDict[noise3] == NoiseDict[noise1]:
#                         NoiseSub1stOrder[key] = -M_1(NoiseDict[noise2])
#                     elif NoiseDict[noise1] != NoiseDict[noise2] and NoiseDict[noise3] == NoiseDict[noise2]:
#                         NoiseSub1stOrder[key] = -M_1(NoiseDict[noise1])
#                     elif NoiseDict[noise1] != NoiseDict[noise3] and NoiseDict[noise3] != NoiseDict[noise2]:
#                         NoiseSub1stOrder[key] = 0
#                     else:
#                         NoiseSub1stOrder[key] = 0 
#                 key2 = NoiseDict[noise1]*Derivative(Pdim,NoiseDict[noise2],NoiseDict[noise3])
#                 if key2 not in NoiseSub1stOrder:
#                     NoiseSub1stOrder[key2] = 0   
#     
#     for noise1 in NoiseDict:
#         for noise2 in NoiseDict:
#             key = NoiseDict[noise1]*NoiseDict[noise2]*Pdim
#             if key not in NoiseSub2ndOrder:
#                 NoiseSub2ndOrder[key] = M_2(NoiseDict[noise1]*NoiseDict[noise2])
#             for noise3 in NoiseDict:
#                 for noise4 in NoiseDict:
#                     key2 = NoiseDict[noise1]*NoiseDict[noise2]*NoiseDict[noise3]*Derivative(Pdim,NoiseDict[noise4])
#                     if key2 not in NoiseSub2ndOrder:
#                         if noise1 == noise2 and noise2 == noise3 and noise3 == noise4:
#                             NoiseSub2ndOrder[key2] = -3*M_2(NoiseDict[noise1]*NoiseDict[noise1])
#                         elif noise1 == noise2 and noise2 != noise3 and noise1 == noise4:
#                             NoiseSub2ndOrder[key2] = -2*M_2(NoiseDict[noise1]*NoiseDict[noise3])
#                         elif noise1 == noise2 and noise2 != noise3 and noise3 == noise4:
#                             NoiseSub2ndOrder[key2] = -M_2(NoiseDict[noise1]*NoiseDict[noise2])
#                         elif noise1 != noise2 and noise2 == noise3 and noise1 == noise4:
#                             NoiseSub2ndOrder[key2] = -M_2(NoiseDict[noise2]*NoiseDict[noise3])
#                         elif noise1 != noise2 and noise2 == noise3 and noise3 == noise4:
#                             NoiseSub2ndOrder[key2] = -2*M_2(NoiseDict[noise1]*NoiseDict[noise2])
#                         elif noise1 != noise2 and noise2 != noise3 and noise1 != noise3:
#                             if noise1 == noise4:
#                                 NoiseSub2ndOrder[key2] = -M_2(NoiseDict[noise2]*NoiseDict[noise3])
#                             elif noise2 == noise4:
#                                 NoiseSub2ndOrder[key2] = -M_2(NoiseDict[noise1]*NoiseDict[noise3])
#                             elif noise3 == noise4:
#                                 NoiseSub2ndOrder[key2] = -M_2(NoiseDict[noise1]*NoiseDict[noise2])
#                             else:
#                                 NoiseSub2ndOrder[key2] = 0
#                         else:
#                             NoiseSub2ndOrder[key2] = 0
#                     key3 = NoiseDict[noise1]*NoiseDict[noise2]*Derivative(Pdim,NoiseDict[noise3],NoiseDict[noise4])
#                     if key3 not in NoiseSub2ndOrder:
#                         if noise1 == noise3 and noise2 == noise4:
#                             if noise1 == noise2:
#                                 NoiseSub2ndOrder[key3] = 2
#                             else:
#                                  NoiseSub2ndOrder[key3] = 1
#                         elif noise1 == noise4 and noise2 == noise3:
#                             if noise1 == noise2:
#                                 NoiseSub2ndOrder[key3] = 2
#                             else:
#                                 NoiseSub2ndOrder[key3] = 1
#                         else:
#                             NoiseSub2ndOrder[key3] = 0
#     NoiseSubs1stOrder = {}                   
#     EQsys1stOrdMom = []
#     EOM_1stOrderMom = {}
#     for fpe_lhs in FPEdict: 
#         for noise in NoiseDict:
#             eq1stOrderMoment = (NoiseDict[noise]*FPEdict[fpe_lhs]).expand() 
#             eq1stOrderMoment = eq1stOrderMoment.subs(NoiseSub1stOrder)
#             EQsys1stOrdMom.append(eq1stOrderMoment)
#             if M_1(NoiseDict[noise]) not in EOM_1stOrderMom:
#                 EOM_1stOrderMom[M_1(NoiseDict[noise])] = eq1stOrderMoment
#                 NoiseSubs1stOrder[M_1(NoiseDict[noise])] = r'<'+latex(NoiseDict[noise])+'>'
#     print(EOM_1stOrderMom)
#             
#     if len(NoiseDict)==2:
#         SOL_1stOrderMom = solve(EQsys1stOrdMom, [M_1(NoiseDict[nvec[0]]),M_1(NoiseDict[nvec[1]])], dict=True)
#     elif len(NoiseDict)==3:
#         SOL_1stOrderMom = solve(EQsys1stOrdMom, [M_1(NoiseDict[nvec[0]]),M_1(NoiseDict[nvec[1]]),M_1(NoiseDict[nvec[2]])], dict=True)
#     else:
#         SOL_1stOrderMom = solve(EQsys1stOrdMom, [M_1(NoiseDict[nvec[0]]),M_1(NoiseDict[nvec[1]]),M_1(NoiseDict[nvec[2]]),M_1(NoiseDict[nvec[3]])], dict=True)
#     print(SOL_1stOrderMom)
#     
#     NoiseSubs2ndOrder = {}
#     EQsys2ndOrdMom = []
#     EOM_2ndOrderMom = {}
#     for fpe_lhs in FPEdict: 
#         for noise1 in NoiseDict:
#             for noise2 in NoiseDict:
#                 eq2ndOrderMoment = (NoiseDict[noise1]*NoiseDict[noise2]*FPEdict[fpe_lhs]).expand() 
#                 eq2ndOrderMoment = eq2ndOrderMoment.subs(NoiseSub2ndOrder)
#                 eq2ndOrderMoment = eq2ndOrderMoment.subs(NoiseSub1stOrder)
#                 eq2ndOrderMoment = eq2ndOrderMoment.subs(SOL_1stOrderMom[0])
#                 if len(NoiseDict)==2:
#                     eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]]))
#                     eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[1]]))
#                     eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]]))
#                 elif len(NoiseDict)==3:
#                     eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]]))
#                     eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[1]]))
#                     eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[2]]*NoiseDict[nvec[2]]))
#                     eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]]))
#                     eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[2]]))
#                     eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[2]]))
#                 else:
#                     eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]]))
#                     eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[1]]))
#                     eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]]))
#                 eq2ndOrderMoment = eq2ndOrderMoment.simplify()
#                 if eq2ndOrderMoment not in EQsys2ndOrdMom:
#                     EQsys2ndOrdMom.append(eq2ndOrderMoment)
#                 if M_2(NoiseDict[noise1]*NoiseDict[noise2]) not in EOM_2ndOrderMom:
#                     EOM_2ndOrderMom[M_2(NoiseDict[noise1]*NoiseDict[noise2])] = eq2ndOrderMoment
#                     NoiseSubs2ndOrder[M_2(NoiseDict[noise1]*NoiseDict[noise2])] = r'<'+latex(NoiseDict[noise1]*NoiseDict[noise2])+'>'
#     print(EOM_2ndOrderMom)
#                     
#     SOL_2ndOrdMomDict = {} 
#     if len(NoiseDict)==2:
#         SOL_2ndOrderMom = list(linsolve(EQsys2ndOrdMom, [M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]]), M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[1]]), M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]])]))[0] #only one set of solutions (if any) in linear system of equations
#         
#         if M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]]) in SOL_2ndOrderMom:
#             ZsubDict = {M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]]): 0}
#             SOL_2ndOrderMomMod = []
#             for nn in range(len(SOL_2ndOrderMom)):
#                 SOL_2ndOrderMomMod.append(SOL_2ndOrderMom[nn].subs(ZsubDict))
#             SOL_2ndOrderMom = SOL_2ndOrderMomMod
#         SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]])] = SOL_2ndOrderMom[0]
#         SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[1]])] = SOL_2ndOrderMom[1]
#         SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]])] = SOL_2ndOrderMom[2]
#     
#     elif len(NoiseDict)==3:
#         SOL_2ndOrderMom = list(linsolve(EQsys2ndOrdMom, [M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]]), 
#                                                          M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[1]]), 
#                                                          M_2(NoiseDict[nvec[2]]*NoiseDict[nvec[2]]), 
#                                                          M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]]), 
#                                                          M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[2]]), 
#                                                          M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[2]])]))[0] #only one set of solutions (if any) in linear system of equations; hence index [0]
#         ZsubDict = {}
#         for noise1 in NoiseDict:
#             for noise2 in NoiseDict:
#                 if M_2(NoiseDict[noise1]*NoiseDict[noise2]) in SOL_2ndOrderMom:
#                     ZsubDict[M_2(NoiseDict[noise1]*NoiseDict[noise2])] = 0
#         if len(ZsubDict) > 0:
#             SOL_2ndOrderMomMod = []
#             for nn in range(len(SOL_2ndOrderMom)):
#                 SOL_2ndOrderMomMod.append(SOL_2ndOrderMom[nn].subs(ZsubDict))
#         SOL_2ndOrderMom = SOL_2ndOrderMomMod
#         SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]])] = SOL_2ndOrderMom[0]
#         SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[1]])] = SOL_2ndOrderMom[1]
#         SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[2]]*NoiseDict[nvec[2]])] = SOL_2ndOrderMom[2]
#         SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]])] = SOL_2ndOrderMom[3]
#         SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[2]])] = SOL_2ndOrderMom[4]
#         SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[2]])] = SOL_2ndOrderMom[5]
#     print(SOL_2ndOrdMomDict)    
#       
#     return EOM_1stOrderMom, SOL_1stOrderMom[0], NoiseSubs1stOrder, EOM_2ndOrderMom, SOL_2ndOrdMomDict, NoiseSubs2ndOrder 
    

## Function that returns the ODE system deerived from Master equation
def _getODEs_vKE(_get_orderedLists_vKE, stoich):
    P, t = symbols('P t')
    V = Symbol('V', real=True, constant=True)
    Vlist_lhs, Vlist_rhs, substring = _get_orderedLists_vKE(stoich)
    rhsODE=0
    lhsODE=0
    for kk in range(len(Vlist_rhs)):
        for key in Vlist_rhs[kk]:
            if key==sqrt(V):
                rhsODE += Vlist_rhs[kk][key]            
    for kk in range(len(Vlist_lhs)):
        for key in Vlist_lhs[kk]:
            if key==sqrt(V):
                lhsODE += Vlist_lhs[kk][key]  
        
    ODE = lhsODE-rhsODE
    
    nvec = []
    for key1 in stoich:
        for key2 in stoich[key1]:
            if key2 != 'rate' and stoich[key1][key2] != 'const':
                if not key2 in nvec:
                    nvec.append(key2)
    #for reactant in reactants:
    #    nvec.append(reactant)
    nvec = sorted(nvec, key=default_sort_key)
    assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'
    
    PhiDict = {}
    NoiseDict = {}
    for kk in range(len(nvec)):
        NoiseDict[nvec[kk]] = Symbol('eta_'+str(nvec[kk]))
        PhiDict[nvec[kk]] = Symbol('Phi_'+str(nvec[kk]))
        
    PhiSubDict = None    
    if not substring == None:
        PhiSubDict = {}
        for sub in substring:
            PhiSubSym = Symbol('Phi_'+str(sub))
            PhiSubDict[PhiSubSym] = substring[sub]
        for key in PhiSubDict:
            for sym in PhiSubDict[key].atoms(Symbol):
                phisub = Symbol('Phi_'+str(sym))
                if sym in nvec:
                    symSub = phisub
                    PhiSubDict[key] = PhiSubDict[key].subs({sym: symSub})
                else:
                    PhiSubDict[key] = PhiSubDict[key].subs({sym: 1}) #here we assume that if a reactant in the substitution string is not a time-dependent reactant it can only be the total number of reactants which is constant, i.e. 1=N/N
    
    
    if len(Vlist_lhs)-1 == 2:
        ode1 = 0
        ode2 = 0
        for kk in range(len(ODE.args)):
            prod=1
            for nn in range(len(ODE.args[kk].args)-1):
                prod *= ODE.args[kk].args[nn]
            if ODE.args[kk].args[-1] == Derivative(P(NoiseDict[nvec[0]],NoiseDict[nvec[1]],t), NoiseDict[nvec[0]]):
                ode1 += prod
            elif ODE.args[kk].args[-1] == Derivative(P(NoiseDict[nvec[0]],NoiseDict[nvec[1]],t), NoiseDict[nvec[1]]):
                ode2 += prod
            else:
                print('Check ODE.args!')
                
        if PhiSubDict:
            ode1 = ode1.subs(PhiSubDict)
            ode2 = ode2.subs(PhiSubDict) 
            
            for key in PhiSubDict:
                if key == PhiDict[nvec[0]]:
                    ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]] , t), dict=True)
                    ODEsys = {**ODE_2[0]}
                elif key == PhiDict[nvec[1]]:
                    ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]] , t), dict=True)
                    ODEsys = {**ODE_1[0]}
                else:
                    ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]] , t), dict=True)
                    ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]] , t), dict=True)
                    ODEsys = {**ODE_1[0], **ODE_2[0]}            
        else:        
            ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]] , t), dict=True)
            ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]] , t), dict=True)
            ODEsys = {**ODE_1[0], **ODE_2[0]}
                
    elif len(Vlist_lhs)-1 == 3:
        ode1 = 0
        ode2 = 0
        ode3 = 0
        for kk in range(len(ODE.args)):
            prod=1
            for nn in range(len(ODE.args[kk].args)-1):
                prod *= ODE.args[kk].args[nn]
            if ODE.args[kk].args[-1] == Derivative(P(NoiseDict[nvec[0]],NoiseDict[nvec[1]],NoiseDict[nvec[2]],t), NoiseDict[nvec[0]]):
                ode1 += prod
            elif ODE.args[kk].args[-1] == Derivative(P(NoiseDict[nvec[0]],NoiseDict[nvec[1]],NoiseDict[nvec[2]],t), NoiseDict[nvec[1]]):
                ode2 += prod
            else:
                ode3 += prod
        
        if PhiSubDict:
            ode1 = ode1.subs(PhiSubDict)
            ode2 = ode2.subs(PhiSubDict)
            ode3 = ode3.subs(PhiSubDict)
            
            for key in PhiSubDict:
                if key == PhiDict[nvec[0]]:
                    ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]] , t), dict=True)
                    ODE_3 = solve(ode3, Derivative(PhiDict[nvec[2]] , t), dict=True)
                    ODEsys = {**ODE_2[0], **ODE_3[0]}
                elif key == PhiDict[nvec[1]]:
                    ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]] , t), dict=True)
                    ODE_3 = solve(ode3, Derivative(PhiDict[nvec[2]] , t), dict=True)
                    ODEsys = {**ODE_1[0], **ODE_3[0]}
                elif key == PhiDict[nvec[2]]:
                    ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]] , t), dict=True)
                    ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]] , t), dict=True)
                    ODEsys = {**ODE_1[0], **ODE_2[0]}
                else:
                    ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]] , t), dict=True)
                    ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]] , t), dict=True)
                    ODE_3 = solve(ode3, Derivative(PhiDict[nvec[2]] , t), dict=True)
                    ODEsys = {**ODE_1[0], **ODE_2[0], **ODE_3[0]} 
        
        else:
            ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]] , t), dict=True)
            ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]] , t), dict=True)
            ODE_3 = solve(ode3, Derivative(PhiDict[nvec[2]] , t), dict=True)
            ODEsys = {**ODE_1[0], **ODE_2[0], **ODE_3[0]}
            
    elif len(Vlist_lhs)-1 == 4:
        ode1 = 0
        ode2 = 0
        ode3 = 0
        ode4 = 0
        for kk in range(len(ODE.args)):
            prod=1
            for nn in range(len(ODE.args[kk].args)-1):
                prod *= ODE.args[kk].args[nn]
            if ODE.args[kk].args[-1] == Derivative(P(NoiseDict[nvec[0]],NoiseDict[nvec[1]],NoiseDict[nvec[2]],NoiseDict[nvec[3]],t), NoiseDict[nvec[0]]):
                ode1 += prod
            elif ODE.args[kk].args[-1] == Derivative(P(NoiseDict[nvec[0]],NoiseDict[nvec[1]],NoiseDict[nvec[2]],NoiseDict[nvec[3]],t), NoiseDict[nvec[1]]):
                ode2 += prod
            elif ODE.args[kk].args[-1] == Derivative(P(NoiseDict[nvec[0]],NoiseDict[nvec[1]],NoiseDict[nvec[2]],NoiseDict[nvec[3]],t), NoiseDict[nvec[2]]):
                ode3 += prod
            else:
                ode4 += prod
        
        if PhiSubDict:
            ode1 = ode1.subs(PhiSubDict)
            ode2 = ode2.subs(PhiSubDict)
            ode3 = ode3.subs(PhiSubDict)
            ode4 = ode4.subs(PhiSubDict)
            
            for key in PhiSubDict:
                if key == PhiDict[nvec[0]]:
                    ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]] , t), dict=True)
                    ODE_3 = solve(ode3, Derivative(PhiDict[nvec[2]] , t), dict=True)
                    ODE_4 = solve(ode4, Derivative(PhiDict[nvec[3]] , t), dict=True)
                    ODEsys = {**ODE_2[0], **ODE_3[0], **ODE_4[0]}
                elif key == PhiDict[nvec[1]]:
                    ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]] , t), dict=True)
                    ODE_3 = solve(ode3, Derivative(PhiDict[nvec[2]] , t), dict=True)
                    ODE_4 = solve(ode4, Derivative(PhiDict[nvec[3]] , t), dict=True)
                    ODEsys = {**ODE_1[0], **ODE_3[0], **ODE_4[0]}
                elif key == PhiDict[nvec[2]]:
                    ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]] , t), dict=True)
                    ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]] , t), dict=True)
                    ODE_4 = solve(ode4, Derivative(PhiDict[nvec[3]] , t), dict=True)
                    ODEsys = {**ODE_1[0], **ODE_2[0], **ODE_4[0]}
                elif key == PhiDict[nvec[3]]:
                    ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]] , t), dict=True)
                    ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]] , t), dict=True)
                    ODE_3 = solve(ode3, Derivative(PhiDict[nvec[2]] , t), dict=True)
                    ODEsys = {**ODE_1[0], **ODE_2[0], **ODE_3[0]}
                else:
                    ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]] , t), dict=True)
                    ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]] , t), dict=True)
                    ODE_3 = solve(ode3, Derivative(PhiDict[nvec[2]] , t), dict=True)
                    ODE_4 = solve(ode4, Derivative(PhiDict[nvec[3]] , t), dict=True)
                    ODEsys = {**ODE_1[0], **ODE_2[0], **ODE_3[0], **ODE_4[0]} 
        
        else:
            ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]] , t), dict=True)
            ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]] , t), dict=True)
            ODE_3 = solve(ode3, Derivative(PhiDict[nvec[2]] , t), dict=True)
            ODE_4 = solve(ode4, Derivative(PhiDict[nvec[3]] , t), dict=True)
            ODEsys = {**ODE_1[0], **ODE_2[0], **ODE_3[0], **ODE_4[0]}
            
    else:
        print('Not implemented yet.')
        
    return ODEsys
 

def _process_params(params):
    paramsRet = []
    paramNames, paramValues = zip(*params)
    for name in paramNames:
#                self._paramNames.append(name.replace('\\','')) ## @todo: have to rationalise how LaTeX characters are handled
        if name == 'plotLimits' or name == 'systemSize':
            paramsRet.append(name)
        else:
            expr = process_sympy(name.replace('\\\\','\\'))
            atoms = expr.atoms()
            if len(atoms) > 1:
                raise SyntaxError("Non-singleton parameter name in parameter " + name)
            for atom in atoms:
                # parameter name should contain a single atom
                pass
            paramsRet.append(atom)
            
    return (paramsRet, paramValues)

    
def _raiseModelError(expected, read, rule):
    raise SyntaxError("Expected " + expected + " but read '" + read + "' in rule: " + rule)


## generic method for constructing figures in MuMoTview and MuMoTmultiController classes
def _buildFig(object, figure = None):
    global figureCounter
    object._figureNum = figureCounter
    figureCounter += 1
    plt.ion()
    with warnings.catch_warnings(): # ignore warnings when plt.hold has been deprecated in installed libraries - still need to try plt.hold(True) in case older libraries in use
        warnings.filterwarnings("ignore",category=MatplotlibDeprecationWarning)
        warnings.filterwarnings("ignore",category=UserWarning)
        plt.hold(True)  
    if figure == None:
        object._figure = plt.figure(object._figureNum) 
    else:
        object._figure = figure

## used for determining significant digits for axes formatting in plots MuMoTstreamView and MuMoTbifurcationView 
def round_to_1(x):
    if x == 0: return 1
    return round(x, -int(floor(log10(abs(x)))))

## Function for editing properties of 3D plots. 
#
#This function is used in MuMoTvectorView.
def _fig_formatting_3D(figure, xlab=None, ylab=None, zlab=None, ax_reformat=False, 
                       specialPoints=None, showFixedPoints=False, **kwargs):
    
    fig = plt.gcf()
    #fig.set_size_inches(10,8) 
    ax = fig.gca(projection='3d')
    
    if kwargs.get('showPlane', False) == True:
        #pointsMesh = np.linspace(0, 1, 11)
        #Xdat, Ydat = np.meshgrid(pointsMesh, pointsMesh)
        #Zdat = 1 - Xdat - Ydat
        #Zdat[Zdat<0] = 0
        #ax.plot_surface(Xdat, Ydat, Zdat, rstride=20, cstride=20, color='grey', alpha=0.25)
        #ax.plot_wireframe(Xdat, Ydat, Zdat, rstride=1, cstride=1, color='grey', alpha=0.5)
        ax.plot([1,0,0,1], [0,1,0,0], [0,0,1,0], linewidth=2, c='k')
        
    if xlab==None:
        try:
            xlabelstr = ax.xaxis.get_label_text()
            if len(ax.xaxis.get_label_text())==0:
                xlabelstr = 'choose x-label'
        except:
            xlabelstr = 'choose x-label'
    else:
        xlabelstr = xlab
        
    if ylab==None:
        try:
            ylabelstr = ax.yaxis.get_label_text()
            if len(ax.yaxis.get_label_text())==0:
                ylabelstr = 'choose y-label'
        except:
            ylabelstr = 'choose y-label'
    else:
        ylabelstr = ylab
        
    if zlab==None:
        try:
            zlabelstr = ax.yaxis.get_label_text()
            if len(ax.zaxis.get_label_text())==0:
                zlabelstr = 'choose z-label'
        except:
            zlabelstr = 'choose z-label'
    else:
        zlabelstr = zlab
    
    x_lim_left = ax.get_xbound()[0]#ax.xaxis.get_data_interval()[0]
    x_lim_right = ax.get_xbound()[1]#ax.xaxis.get_data_interval()[1]
    y_lim_bot = ax.get_ybound()[0]#ax.yaxis.get_data_interval()[0]
    y_lim_top = ax.get_ybound()[1]
    z_lim_bot = ax.get_zbound()[0]#ax.zaxis.get_data_interval()[0]
    z_lim_top = ax.get_zbound()[1]
    if ax_reformat==False:
        xmajortickslocs = ax.xaxis.get_majorticklocs()
        #xminortickslocs = ax.xaxis.get_minorticklocs()
        ymajortickslocs = ax.yaxis.get_majorticklocs()
        #yminortickslocs = ax.yaxis.get_minorticklocs()
        zmajortickslocs = ax.zaxis.get_majorticklocs()
        #zminortickslocs = ax.zaxis.get_minorticklocs()
        #plt.cla()
        ax.set_xticks(xmajortickslocs)
        #ax.set_xticks(xminortickslocs, minor = True)
        ax.set_yticks(ymajortickslocs)
        #ax.set_yticks(yminortickslocs, minor = True)
        ax.set_zticks(zmajortickslocs)
        #ax.set_zticks(zminortickslocs, minor = True)
    else:
        max_xrange = x_lim_right - x_lim_left
        max_yrange = y_lim_top - y_lim_bot
        if kwargs.get('showPlane', False) == True:
            max_zrange = z_lim_top
        else:
            max_zrange = z_lim_top - z_lim_bot
            
        if max_xrange < 1.0:
            xMLocator_major = round_to_1(max_xrange/4)
        else:
            xMLocator_major = round_to_1(max_xrange/6)
        #xMLocator_minor = xMLocator_major/2
        if max_yrange < 1.0:
            yMLocator_major = round_to_1(max_yrange/4)
        else:
            yMLocator_major = round_to_1(max_yrange/6)
        #yMLocator_minor = yMLocator_major/2
        if max_zrange < 1.0:
            zMLocator_major = round_to_1(max_zrange/4)
        else:
            zMLocator_major = round_to_1(max_zrange/6)
        #zMLocator_minor = yMLocator_major/2
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xMLocator_major))
        #ax.xaxis.set_minor_locator(ticker.MultipleLocator(xMLocator_minor))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(yMLocator_major))
        #ax.yaxis.set_minor_locator(ticker.MultipleLocator(yMLocator_minor))
        ax.zaxis.set_major_locator(ticker.MultipleLocator(zMLocator_major))
        #ax.zaxis.set_minor_locator(ticker.MultipleLocator(zMLocator_minor))
        
    ax.set_xlim3d(x_lim_left,x_lim_right)
    ax.set_ylim3d(y_lim_bot,y_lim_top)
    if kwargs.get('showPlane', False) == True:
        ax.set_zlim3d(0,z_lim_top)
    else:
        ax.set_zlim3d(z_lim_bot,z_lim_top)
                    
    if showFixedPoints==True:
        if not specialPoints[0] == []:
            for jj in range(len(specialPoints[0])):
                try:
                    len(specialPoints[3][jj]) == 3
                    lam1 = specialPoints[3][jj][0]
                    lam2 = specialPoints[3][jj][1]
                    lam3 = specialPoints[3][jj][2]
                    if (re(lam1) < 0 and re(lam2) < 0 and re(lam3) < 0):
                        FPcolor = 'g'
                        FPmarker = 'o'
                    else:
                        FPcolor = 'r'  
                        FPmarker = '>'     
                except:
                    print('Check input!')
                    FPcolor='k'  
                     
                ax.scatter([specialPoints[0][jj]], [specialPoints[1][jj]], [specialPoints[2][jj]], 
                           marker=FPmarker, s=300, c=FPcolor)
    
    if 'fontsize' in kwargs:
        if not kwargs['fontsize']==None:
            chooseFontSize = kwargs['fontsize']
    elif len(xlabelstr) > 40 or len(ylabelstr) > 40 or len(zlabelstr) > 40:
        chooseFontSize = 16
    elif 31 <= len(xlabelstr) <= 40 or 31 <= len(ylabelstr) <= 40 or 31 <= len(zlabelstr) <= 40:
        chooseFontSize = 20
    elif 26 <= len(xlabelstr) <= 30 or 26 <= len(ylabelstr) <= 30 or 26 <= len(zlabelstr) <= 30:
        chooseFontSize = 26
    else:
        chooseFontSize = 30
    
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = 20
    ax.set_xlabel(r''+str(xlabelstr), fontsize = chooseFontSize)
    ax.set_ylabel(r''+str(ylabelstr), fontsize = chooseFontSize)
    if len(str(zlabelstr))>1:
        ax.set_zlabel(r''+str(zlabelstr), fontsize = chooseFontSize, rotation=90)
    else:
        ax.set_zlabel(r''+str(zlabelstr), fontsize = chooseFontSize)
        
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax.zaxis.get_major_ticks():
        tick.set_pad(8)
        tick.label.set_fontsize(18)      
        
    plt.tight_layout(pad=4)
 

## Function for formatting 2D plots. 
#
#This function is used in MuMoTvectorView, MuMoTstreamView and MuMoTbifurcationView    
def _fig_formatting_2D(figure=None, xdata=None, ydata=None, choose_xrange=None, choose_yrange=None, eigenvalues=None, 
                       curve_replot=False, ax_reformat=False, showFixedPoints=False, specialPoints=None,
                       xlab=None, ylab=None, curvelab=None, **kwargs):
    #print(kwargs)
    
    linestyle_list = ['solid','dashed', 'dashdot', 'dotted', 'solid','dashed', 'dashdot', 'dotted', 'solid']
    
    if xdata and ydata:
        if len(xdata) == len(ydata):
            #plt.figure(figsize=(8,6), dpi=80)
            ax = plt.axes()
            data_x=xdata
            data_y=ydata
            
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
    #print(data_x)
    
    if xlab==None:
        try:
            xlabelstr = ax.xaxis.get_label_text()
            if len(ax.xaxis.get_label_text())==0:
                xlabelstr = 'choose x-label'
        except:
            xlabelstr = 'choose x-label'
    else:
        xlabelstr = xlab
        
    if ylab==None:
        try:
            ylabelstr = ax.yaxis.get_label_text()
            if len(ax.yaxis.get_label_text())==0:
                ylabelstr = 'choose y-label'
        except:
            ylabelstr = 'choose y-label'
    else:
        ylabelstr = ylab
    
    if ax_reformat==False and figure!=None:
        xmajortickslocs = ax.xaxis.get_majorticklocs()
        xminortickslocs = ax.xaxis.get_minorticklocs()
        ymajortickslocs = ax.yaxis.get_majorticklocs()
        yminortickslocs = ax.yaxis.get_minorticklocs()
        x_lim_left = ax.get_xbound()[0]#ax.xaxis.get_data_interval()[0]
        x_lim_right = ax.get_xbound()[1]#ax.xaxis.get_data_interval()[1]
        y_lim_bot = ax.get_ybound()[0]#ax.yaxis.get_data_interval()[0]
        y_lim_top = ax.get_ybound()[1]#ax.yaxis.get_data_interval()[1]
        #print(ax.yaxis.get_data_interval())
        
    if curve_replot==True:
        plt.cla()
    
    if ax_reformat==False and figure!=None:
        ax.set_xticks(xmajortickslocs)
        ax.set_xticks(xminortickslocs, minor = True)
        ax.set_yticks(ymajortickslocs)
        ax.set_yticks(yminortickslocs, minor = True)
        ax.tick_params(axis = 'both', which = 'major', length=5, width=2)
        ax.tick_params(axis = 'both', which = 'minor', length=3, width=1)
        plt.xlim(x_lim_left,x_lim_right)
        plt.ylim(y_lim_bot,y_lim_top)
        
    if figure==None or curve_replot==True:
        if 'LineThickness' in kwargs:
            LineThickness = kwargs['LineThickness']
        else:
            LineThickness = 4
        
        if eigenvalues:
            round_digit = 10
#             solX_dict={} #bifurcation parameter
#             solY_dict={} #state variable 1
#             solX_dict['solX_unst']=[] 
#             solY_dict['solY_unst']=[]
#             solX_dict['solX_stab']=[]
#             solY_dict['solY_stab']=[]
#             solX_dict['solX_saddle']=[]
#             solY_dict['solY_saddle']=[]
#             
#             nr_sol_unst=0
#             nr_sol_saddle=0
#             nr_sol_stab=0
#             data_x_tmp=[]
#             data_y_tmp=[]
#             #print(specialPoints)
            for nn in range(len(data_x)):
                solX_dict={} #bifurcation parameter
                solY_dict={} #state variable 1
                solX_dict['solX_unst']=[] 
                solY_dict['solY_unst']=[]
                solX_dict['solX_stab']=[]
                solY_dict['solY_stab']=[]
                solX_dict['solX_saddle']=[]
                solY_dict['solY_saddle']=[]
                
                nr_sol_unst=0
                nr_sol_saddle=0
                nr_sol_stab=0
                data_x_tmp=[]
                data_y_tmp=[]
                #sign_change=0
                for kk in range(len(eigenvalues[nn])):
                    if kk > 0:
                        if (np.sign(np.round(np.real(eigenvalues[nn][kk][0]), round_digit))*np.sign(np.round(np.real(eigenvalues[nn][kk-1][0]), round_digit)) <= 0
                            or np.sign(np.round(np.real(eigenvalues[nn][kk][1]), round_digit))*np.sign(np.round(np.real(eigenvalues[nn][kk-1][1]), round_digit)) <= 0):
                            #print('sign change')
                            #sign_change+=1
                            #print(sign_change)
                            #if specialPoints !=None and specialPoints[0]!=[]:
                            #    data_x_tmp.append(specialPoints[0][sign_change-1])
                            #    data_y_tmp.append(specialPoints[1][sign_change-1])
                            
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
                            
                            data_x_tmp_first=data_x_tmp[-1]
                            data_y_tmp_first=data_y_tmp[-1]
                            nr_sol_stab=0
                            nr_sol_saddle=0
                            nr_sol_unst=0
                            data_x_tmp=[]
                            data_y_tmp=[]
                            data_x_tmp.append(data_x_tmp_first)
                            data_y_tmp.append(data_y_tmp_first)
                            #if specialPoints !=None and specialPoints[0]!=[]:
                            #    data_x_tmp.append(specialPoints[0][sign_change-1])
                            #    data_y_tmp.append(specialPoints[1][sign_change-1])
                    if np.sign(np.round(np.real(eigenvalues[nn][kk][0]), round_digit)) == -1 and np.sign(np.round(np.real(eigenvalues[nn][kk][1]), round_digit)) == -1:  
                        nr_sol_stab=1
                    elif np.sign(np.round(np.real(eigenvalues[nn][kk][0]), round_digit)) in [0,1] and np.sign(np.round(np.real(eigenvalues[nn][kk][1]), round_digit)) == -1:
                        nr_sol_saddle=1
                    elif np.sign(np.round(np.real(eigenvalues[nn][kk][0]), round_digit)) == -1 and np.sign(np.round(np.real(eigenvalues[nn][kk][1]), round_digit)) in [0,1]:
                        nr_sol_saddle=1
                    else:
                        nr_sol_unst=1                
#                     if np.real(eigenvalues[nn][kk][0]) < 0 and np.real(eigenvalues[nn][kk][1]) < 0:  
#                         nr_sol_stab=1
#                     elif np.real(eigenvalues[nn][kk][0]) >= 0 and np.real(eigenvalues[nn][kk][1]) < 0:
#                         nr_sol_saddle=1
#                     elif np.real(eigenvalues[nn][kk][0]) < 0 and np.real(eigenvalues[nn][kk][1]) >= 0:
#                         nr_sol_saddle=1
#                     else:
#                         nr_sol_unst=1
                         
                    data_x_tmp.append(data_x[nn][kk])
                    data_y_tmp.append(data_y[nn][kk])
                
                    if kk == len(eigenvalues[nn])-1:
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
                        plt.plot(solX_dict['solX_unst'][jj], 
                                 solY_dict['solY_unst'][jj], 
                                 c = line_color_list[2], 
                                 ls = linestyle_list[3], lw = LineThickness, label = r'unstable')
                if not solX_dict['solX_stab'] == []:            
                    for jj in range(len(solX_dict['solX_stab'])):
                        plt.plot(solX_dict['solX_stab'][jj], 
                                 solY_dict['solY_stab'][jj], 
                                 c = line_color_list[1], 
                                 ls = linestyle_list[0], lw = LineThickness, label = r'stable')
                if not solX_dict['solX_saddle'] == []:            
                    for jj in range(len(solX_dict['solX_saddle'])):
                        plt.plot(solX_dict['solX_saddle'][jj], 
                                 solY_dict['solY_saddle'][jj], 
                                 c = line_color_list[0], 
                                 ls = linestyle_list[1], lw = LineThickness, label = r'saddle')
                
                
                
#             if not solX_dict['solX_unst'] == []:            
#                 for jj in range(len(solX_dict['solX_unst'])):
#                     plt.plot(solX_dict['solX_unst'][jj], 
#                              solY_dict['solY_unst'][jj], 
#                              c = line_color_list[2], 
#                              ls = linestyle_list[3], lw = LineThickness, label = r'unstable')
#             if not solX_dict['solX_stab'] == []:            
#                 for jj in range(len(solX_dict['solX_stab'])):
#                     plt.plot(solX_dict['solX_stab'][jj], 
#                              solY_dict['solY_stab'][jj], 
#                              c = line_color_list[1], 
#                              ls = linestyle_list[0], lw = LineThickness, label = r'stable')
#             if not solX_dict['solX_saddle'] == []:            
#                 for jj in range(len(solX_dict['solX_saddle'])):
#                     plt.plot(solX_dict['solX_saddle'][jj], 
#                              solY_dict['solY_saddle'][jj], 
#                              c = line_color_list[0], 
#                              ls = linestyle_list[1], lw = LineThickness, label = r'saddle')
                                    
                            
        else:
            for nn in range(len(data_x)):
                try:
                    plt.plot(data_x[nn], data_y[nn], c = line_color_list[nn], 
                             ls = linestyle_list[nn], lw = LineThickness, label = r''+str(curvelab[nn]))
                except:
                    plt.plot(data_x[nn], data_y[nn], c = line_color_list[nn], 
                             ls = linestyle_list[nn], lw = LineThickness)
        
        
    if len(xlabelstr) > 40 or len(ylabelstr) > 40:
        chooseFontSize = 16
    elif 31 <= len(xlabelstr) <= 40 or 31 <= len(ylabelstr) <= 40:
        chooseFontSize = 20
    elif 26 <= len(xlabelstr) <= 30 or 26 <= len(ylabelstr) <= 30:
        chooseFontSize = 26
    else:
        chooseFontSize = 30
        
    if 'fontsize' in kwargs:
        if not kwargs['fontsize']==None:
            chooseFontSize = kwargs['fontsize']

    plt.xlabel(r''+str(xlabelstr), fontsize = chooseFontSize)
    plt.ylabel(r''+str(ylabelstr), fontsize = chooseFontSize)
    #ax.set_xlabel(r''+str(xlabelstr), fontsize = chooseFontSize)
    #ax.set_ylabel(r''+str(ylabelstr), fontsize = chooseFontSize)
     
    if figure==None or ax_reformat==True:
        if choose_xrange:
            max_xrange = choose_xrange[1]-choose_xrange[0]
        else:
            #xrange = [np.max(data_x[kk]) - np.min(data_x[kk]) for kk in range(len(data_x))]
            XaxisMax = np.max([np.max(data_x[kk]) for kk in range(len(data_x))])
            XaxisMin = np.min([np.min(data_x[kk]) for kk in range(len(data_x))])
            max_xrange = XaxisMax - XaxisMin  # max(xrange)
        
        if choose_yrange:
            max_yrange = choose_yrange[1]-choose_yrange[0]
        else:
            #yrange = [np.max(data_y[kk]) - np.min(data_y[kk]) for kk in range(len(data_y))]
            #max_yrange = np.max(data_y) - np.min(data_y) #max(yrange)
            YaxisMax = np.max([np.max(data_y[kk]) for kk in range(len(data_y))])
            YaxisMin = np.min([np.min(data_y[kk]) for kk in range(len(data_y))])
            max_yrange = YaxisMax - YaxisMin
        
        if max_xrange < 1.0:
            xMLocator_major = round_to_1(max_xrange/5)
        else:
            xMLocator_major = round_to_1(max_xrange/10)
        xMLocator_minor = xMLocator_major/2
        if max_yrange < 1.0:
            yMLocator_major = round_to_1(max_yrange/5)
        else:
            yMLocator_major = round_to_1(max_yrange/10)
        yMLocator_minor = yMLocator_major/2
        
        if choose_xrange:
            plt.xlim(choose_xrange[0]-xMLocator_minor, choose_xrange[1]+xMLocator_minor)
        else:
            plt.xlim(XaxisMin-xMLocator_minor, XaxisMax+xMLocator_minor)
        if choose_yrange:
            plt.ylim(choose_yrange[0]-yMLocator_minor, choose_yrange[1]+yMLocator_minor)
        else:
            plt.ylim(YaxisMin-yMLocator_minor, YaxisMax+yMLocator_minor)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(xMLocator_major))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(xMLocator_minor))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(yMLocator_major))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(yMLocator_minor))
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        
        ax.tick_params('both', length=5, width=2, which='major')
        ax.tick_params('both', length=3, width=1, which='minor')
    
    if eigenvalues:
        if specialPoints != []:
            if specialPoints[0] != []:
                for jj in range(len(specialPoints[0])):
                    plt.plot([specialPoints[0][jj]], [specialPoints[1][jj]], marker='o', markersize=15, 
                             c=line_color_list[-1])    
                for a,b,c in zip(specialPoints[0], specialPoints[1], specialPoints[2]): 
                    if a > plt.xlim()[0]+(plt.xlim()[1]-plt.xlim()[0])/2:
                        x_offset = -(plt.xlim()[1]-plt.xlim()[0])*0.02
                    else:
                        x_offset = (plt.xlim()[1]-plt.xlim()[0])*0.02
                    if b > plt.ylim()[0]+(plt.ylim()[1]-plt.ylim()[0])/2:
                        y_offset = -(plt.ylim()[1]-plt.ylim()[0])*0.05
                    else:
                        y_offset = (plt.ylim()[1]-plt.ylim()[0])*0.05
                    plt.text(a+x_offset, b+y_offset, c, fontsize=18)
    
    if showFixedPoints==True:
        if not specialPoints[0] == []:
            for jj in range(len(specialPoints[0])):
                try:
                    len(specialPoints[2][jj]) == 2
                    lam1 = specialPoints[2][jj][0]
                    lam2 = specialPoints[2][jj][1]
                    if re(lam1) < 0 and re(lam2) < 0:
                        FPcolor = line_color_list[1]
                        FPfill = 'full'
                    elif re(lam1) > 0 and re(lam2) > 0:
                        FPcolor = line_color_list[2]
                        FPfill = 'none'
                    else:
                        FPcolor = line_color_list[0]
                        FPfill = 'none'
                        
                except:
                    print('Check input!')
                    FPcolor=line_color_list[-1]  
                    FPfill = 'none'
                     
                plt.plot([specialPoints[0][jj]], [specialPoints[1][jj]], marker='o', markersize=12, 
                         c=FPcolor, fillstyle=FPfill, mew=4, mec=FPcolor)
    if kwargs.get('grid', False) == True:
        plt.grid()
        
    if curvelab != None:
        #if 'legend_loc' in kwargs:
        #    legend_loc = kwargs['legend_loc']
        #else:
        #    legend_loc = 'upper left'
        legend_fontsize = kwargs.get('legend_fontsize', 20)
        legend_loc = kwargs.get('legend_loc', 'upper left')
        plt.legend(loc=str(legend_loc), fontsize=legend_fontsize, ncol=2)
        
    for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(18) 
    for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(18)
    plt.tight_layout() 
    
def _decodeNetworkTypeFromString(netTypeStr):
    # init the network type
    admissibleNetTypes = {'full': NetworkType.FULLY_CONNECTED, 
                          'erdos-renyi': NetworkType.ERSOS_RENYI, 
                          'barabasi-albert': NetworkType.BARABASI_ALBERT, 
                          'dynamic': NetworkType.DYNAMIC}
    
    if netTypeStr not in admissibleNetTypes:
        print("ERROR! Invalid network type argument! Valid strings are: " + str(admissibleNetTypes) )
    return admissibleNetTypes.get(netTypeStr, None)

def _encodeNetworkTypeToString(netType):
    # init the network type
    netTypeEncoding = {NetworkType.FULLY_CONNECTED: 'full', 
                          NetworkType.ERSOS_RENYI: 'erdos-renyi', 
                          NetworkType.BARABASI_ALBERT: 'barabasi-albert', 
                          NetworkType.DYNAMIC: 'dynamic'}
    
    if netType not in netTypeEncoding:
        print("ERROR! Invalid netTypeEncoding table! Tryed to encode network type: " + str(netType) )
    return netTypeEncoding.get(netType, 'none')

def _make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

def _almostEqual(a,b):
    epsilon = 0.0000001
    return abs(a-b) < epsilon 
