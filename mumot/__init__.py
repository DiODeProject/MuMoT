"""Multiscale Modelling Tool (MuMoT)

For documentation and version information use about()

Contributors:
James A. R. Marshall, Andreagiovanni Reina, Thomas Bose

Packaging, Documentation and Deployment:
Will Furnass

Windows Compatibility:
Renato Pagliara Vasquez
"""

import copy
import datetime
import math
import numbers
import os
import re
import sys
import tempfile
import warnings
from bisect import bisect_left
from enum import Enum
from math import floor, log10
import base64
from ipywidgets import HTML

#if operating system is macOS
#use non-default matplotlib backend
#otherwise rendering of images might not be correct (e.g. superfluous figures when sliders are moved)
#automated testing using tox could be affected as well if default matplotlib backend is used 
if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')

import ipywidgets.widgets as widgets
import matplotlib.patches as mpatch
import matplotlib.ticker as ticker
import networkx as nx  # @UnresolvedImport
import numpy as np
import PyDSTool as dst
import sympy
from graphviz import Digraph
from IPython.display import Javascript, Math, display
from IPython.utils import io
from matplotlib import pyplot as plt
from matplotlib.cbook import MatplotlibDeprecationWarning
from mpl_toolkits.mplot3d import axes3d  # @UnresolvedImport
from pyexpat import model  # @UnresolvedImport
from scipy.integrate import odeint
from sympy import (Derivative, Matrix, Symbol, collect, default_sort_key,
                   expand, factorial, lambdify, latex, linsolve, nan,
                   numbered_symbols, preview, simplify, solve, symbols)

from mumot.process_latex.process_latex import process_sympy
from ._version import __version__

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
except NameError as e:
    # There is no currently-running IPython instance
    pass


class MuMoTWarning(Warning):
    """Class to report MuMoT-specific warnings.
    """
    pass


class MuMoTError(Exception):
    """Class to report MuMoT-specific errors.
    """
    pass


class MuMoTValueError(MuMoTError):
    """Class to report MuMoT-specific errors arising from incorrect input.
    """
    pass


class MuMoTSyntaxError(MuMoTError):
    """Class to report MuMoT-specific errors arising from incorrectly-structured input.
    """
    pass


figureCounter = 1  # global figure counter for model views

MAX_RANDOM_SEED = 2147483647
INITIAL_RATE_VALUE = 0.5
RATE_BOUND = 10.0
RATE_STEP = 0.1
MULTIPLOT_COLUMNS = 2
EMPTYSET_SYMBOL = process_sympy('1')

INITIAL_COND_INIT_VAL = 0.0
INITIAL_COND_INIT_BOUND = 1.0

LINE_COLOR_LIST = ['b', 'g', 'r', 'c', 'm', 'y', 'grey', 'orange', 'k']

GREEK_LETT_LIST_1 = ['alpha', 'beta', 'gamma', 'Gamma', 'delta', 'Delta', 'epsilon',
                    'zeta', 'theta', 'Theta', 'iota', 'kappa', 'lambda', 'Lambda', 
                    'mu', 'xi', 'Xi', 'pi', 'Pi', 'rho', 'sigma', 'Sigma', 'tau', 
                    'upsilon', 'Upsilon', 'phi', 'Phi', 'chi', 'psi', 'Psi', 'omega', 'Omega', 'varrho', 'vartheta', 'varepsilon', 'varphi']
GREEK_LETT_LIST_2 = ['\\' + GreekLett for GreekLett in GREEK_LETT_LIST_1]
GREEK_LETT_RESERVED_LIST = ['\\eta', '\\nu', '\\Phi', '(\\eta)', '(\\nu)', '(\\Phi)']
GREEK_LETT_RESERVED_LIST_PRINT = ['eta', 'nu', 'Phi']
# 
# GREEK_LETT_LIST_2=['\\alpha', '\\beta', '\\gamma', '\\Gamma', '\\delta', '\\Delta', '\\epsilon',
#                     '\\zeta', '\\theta', '\\Theta', '\\iota', '\\kappa', '\\lambda', '\\Lambda', 
#                     '\\mu', '\\nu', '\\xi', '\\Xi', '\\pi', '\\Pi', '\\rho', '\\sigma', '\\Sigma', '\\tau', 
#                     '\\upsilon', '\\Upsilon', '\\phi', '\\Phi', '\\chi', '\\psi', '\\Psi', '\\omega', '\\Omega']


class NetworkType(Enum):
    """Enumeration of possible network types."""

    FULLY_CONNECTED = 0
    ERSOS_RENYI = 1
    BARABASI_ALBERT = 2
    SPACE = 3
    DYNAMIC = 4


class MuMoTdefault:
    """Store default parameters."""

    _initialRateValue = 2  # @todo: was 1 (choose initial values sensibly)
    _rateLimits = (0.0, 20.0)  # @todo: choose limit values sensibly
    _rateStep = 0.1  # @todo: choose rate step sensibly

    @staticmethod
    def setRateDefaults(initRate=_initialRateValue, limits=_rateLimits, step=_rateStep):
        MuMoTdefault._initialRateValue = initRate
        MuMoTdefault._rateLimits = limits
        MuMoTdefault._rateStep = step
    
    _maxTime = 3
    _timeLimits = (0, 10)
    _timeStep = 0.1
    #_maxTime = 5
    #_timeLimits = (0, 50)
    #_timeStep = 0.5

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
        
    _systemSize = 10
    _systemSizeLimits = (5, 100)
    _systemSizeStep = 1

    @staticmethod
    def setSystemSizeDefaults(initSysSize=_systemSize, limits=_systemSizeLimits, step=_systemSizeStep):
        MuMoTdefault._systemSize = initSysSize
        MuMoTdefault._systemSizeLimits = limits
        MuMoTdefault._systemSizeStep = step
        
    _plotLimits = 1.0
    _plotLimitsLimits = (0.1, 5.0)
    _plotLimitsStep = 0.1

    @staticmethod
    def setPlotLimitsDefaults(initPlotLimits=_plotLimits, limits=_plotLimitsLimits, step=_plotLimitsStep):
        MuMoTdefault._plotLimits = initPlotLimits
        MuMoTdefault._plotLimitsLimits = limits
        MuMoTdefault._plotLimitsStep = step
    

class MuMoTmodel:
    """Model class."""
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

    def substitute(self, subsString):
        """Create a new model with variable substitutions.

        Parameters
        ----------
        subsString: str
            Comma-separated string of assignments.

        Returns
        -------
        :class:`MuMoTmodel`
            A new model.

        """
        sizeSet = False
        sizeSetrhs = []
        sizeSetExpr = None
        sizeSetKosher = False
        subs = []
        subsStrings = subsString.split(',')
        for subString in subsStrings:
            if '=' not in subString:
                raise MuMoTSyntaxError("No '=' in assignment " + subString)
            assignment = process_sympy(subString)
            subs.append((assignment.lhs, assignment.rhs))
        newModel = MuMoTmodel()
        newModel._constantSystemSize = self._constantSystemSize
        newModel._rules = copy.deepcopy(self._rules)
        newModel._reactants = copy.deepcopy(self._reactants)
        newModel._constantReactants = copy.deepcopy(self._constantReactants)
        newModel._equations = copy.deepcopy(self._equations)
        newModel._stoichiometry = copy.deepcopy(self._stoichiometry)

        for sub in subs:
            if sub[0] in newModel._reactants and len(sub[1].atoms(Symbol)) == 1:
                raise MuMoTSyntaxError("Using substitute to rename reactants not supported: " + str(sub[0]) + " = " + str(sub[1]))
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
            if sub[0] in newModel._reactants or (sub[0] * -1) in newModel._reactants:
                for atom in sub[1].atoms(Symbol):
                    if atom not in newModel._reactants and atom != self._systemSize:
                        if newModel._systemSize is None:
                            newModel._systemSize = atom
                            sizeSet = True
                            sizeSetExpr = sub[1] - sub[0]
                            if sub[0] in newModel._reactants:
                                sizeSetKosher = True
                        else:
                            raise MuMoTSyntaxError("More than one unknown reactant encountered when trying to set system size: " + str(sub[0]) + " = " + str(sub[1]))
                    else:
                        sizeSetrhs.append(atom)
                if newModel._systemSize is None:
                    raise MuMoTSyntaxError("Expected to find system size parameter but failed: " + str(sub[0]) + " = " + str(sub[1]))
                ## @todo: more thorough error checking for valid system size expression
                newModel._reactants.discard(sub[0])
                if sizeSetKosher:
                    del newModel._equations[sub[0]]
        if newModel._systemSize is None:
            newModel._systemSize = self._systemSize
        for reactant in newModel._equations:
            rhs = newModel._equations[reactant]
            for symbol in rhs.atoms(Symbol):
                if symbol not in newModel._reactants and symbol not in newModel._constantReactants and symbol != newModel._systemSize:
                    newModel._rates.add(symbol)
        newModel._ratesLaTeX = {}
        rates = map(latex, list(newModel._rates))
        for (rate, latexStr) in zip(newModel._rates, rates):
            newModel._ratesLaTeX[repr(rate)] = latexStr
        constantReactants = map(latex, list(newModel._constantReactants))
        for (reactant, latexStr) in zip(newModel._constantReactants, constantReactants):
#            newModel._ratesLaTeX[repr(reactant)] = '(' + latexStr + ')'
            newModel._ratesLaTeX[repr(reactant)] = '\Phi_{' + latexStr + '}'
        if sizeSet:
            # need to check setting system size was done correctly
            candidateExpr = newModel._systemSize
            for reactant in self._equations:
                # check all reactants in original model present in substitution string (first check, to help users explicitly realise they must inlude all reactants)
                if sizeSetKosher and reactant != sub[0] and reactant not in sizeSetrhs:
                    raise MuMoTSyntaxError("Expected to find reactant " + str(reactant) + " but failed: " + str(sub[0]) + " = " + str(sub[1]))
                candidateExpr = candidateExpr - reactant
#                if reactant not in sizeSetrhs:
            # check substitution format is correct
            diffExpr = candidateExpr - sizeSetExpr
            if diffExpr != 0:
                raise MuMoTSyntaxError("System size not set by expression of form <reactant> = <system size> - <reactants>: difference = " + str(diffExpr))
                    
        ## @todo: what else should be copied to new model?

        return newModel

    def visualise(self):
        """Build a graphical representation of the model.

        Returns
        -------
        :class:`graphviz.Digraph`
            Graphical representation of model.

        Notes
        -----
        If result cannot be plotted check for installation of ``libltdl``
        e.g on macOS see if XQuartz needs updating or do: ::

            brew install libtool --universal
            brew link libtool

        """
        errorShown = False
        if self._dot is None:
            dot = Digraph(comment="Model", engine='circo')
            if not self._constantSystemSize:
                dot.node(str('1'), " ", image=self._localLaTeXimageFile(Symbol('\\emptyset')))  # @todo: only display if used: for now, guess it is used if system size is non-constant                
            for reactant in self._reactants:
                dot.node(str(reactant), " ", image=self._localLaTeXimageFile(reactant))
            for reactant in self._constantReactants:
                latexrep = '(' + self._ratesLaTeX[repr(reactant)].replace('\Phi_{', '').replace('}', '') + ')'
                dot.node(str(reactant), " ", image=self._localLaTeXimageFile(Symbol(latexrep)))                
            for rule in self._rules:
                # render LaTeX representation of rule
                latexrep = '$$' + _doubleUnderscorify(_greekPrependify(self._ratesLaTeX.get(repr(rule.rate), repr(rule.rate)))) + '$$'
#                latexrep = latexrep.replace('\\','\\\\')
                localfilename = self._localLaTeXimageFile(latexrep)
                htmlLabel = r'<<TABLE BORDER="0"><TR><TD><IMG SRC="' + localfilename + r'"/></TD></TR></TABLE>>'
                if len(rule.lhsReactants) == 1:
                    dot.edge(str(rule.lhsReactants[0]), str(rule.rhsReactants[0]), label=htmlLabel)
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
                        for i in range(0, 2):
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

                    if source is None:
                        # 'reciprocal inhibition' motif A + B -> C + C/D
                        source = str(rule.lhsReactants[0])
                        target = str(rule.lhsReactants[1])
                        head = 'dot'
                        tail = 'dot'

                    if source is not None:
                        dot.edge(source, target, label=htmlLabel, arrowhead=head, arrowtail=tail, dir='both')
                else:
                    if not errorShown:
                        errorShown = True
                        print("Model contains rules with three or more reactants; only displaying unary and binary rules")
            self._dot = dot
                
        return self._dot

    def showConstantReactants(self):
        """Show a sorted LaTeX representation of the model's constant reactants.

        Displays the LaTeX representation in the Jupyter Notebook if there are
        constant reactants in the model, otherwise prints an error.
        
        Returns
        -------
            `None`

        """
        if len(self._constantReactants) == 0:
            print('No constant reactants in the model!')
        else:
            for reactant in self._constantReactants:
                out = self._ratesLaTeX[repr(reactant)]
                out = _doubleUnderscorify(_greekPrependify(out))
                display(Math(out))

    def showReactants(self):
        """Show a sorted LaTeX representation of the model's reactants.

        Displays rendered LaTeX in the Jupyter Notebook.
        
        Returns
        -------
            `None`

        """
        if self._reactantsLaTeX is None:
            self._reactantsLaTeX = []
            reactants = map(latex, list(self._reactants))
            for reactant in reactants:
                self._reactantsLaTeX.append(reactant)
            self._reactantsLaTeX.sort()
        for reactant in self._reactantsLaTeX:
            out = _doubleUnderscorify(_greekPrependify(reactant))
            display(Math(out))

    def showRates(self):
        """Show a sorted LaTeX representation of the model's rate parameters.

        Displays rendered LaTeX in the Jupyter Notebook.
        
        Returns
        -------
            `None`

        """
        for reaction in self._stoichiometry:
            out = latex(self._stoichiometry[reaction]['rate']) + "\; (" + latex(reaction) + ")"
            out = _doubleUnderscorify(_greekPrependify(out))
            display(Math(out))


    def showSingleAgentRules(self):
        """Show the probabilistic transitions of the agents in each possible reactant-state.

        Returns
        -------
            `None`
        """
        if not self._agentProbabilities:
            self._getSingleAgentRules()
        for agent, probs in self._agentProbabilities.items():
            if agent == EMPTYSET_SYMBOL:
                print("Spontaneous birth from EMPTYSET", end=' ')
            else:
                print("Agent " + str(agent), end=' ')
            if probs:
                print("reacts")
                for prob in probs:
                    print("  at rate " + str(prob[1]), end=' ')
                    if prob[0]:
                        print("when encounters " + str(prob[0]), end=' ')
                    else:
                        print("alone", end=' ') 
                    print("and becomes " + str(prob[2]), end=', ')
                    if prob[0]:
                        print("while", end=' ')
                        for i in np.arange(len(prob[0])):
                            print("reagent " + str(prob[0][i]) + " becomes " + str(prob[3][i]), end=' ')
                    print("")
            else: 
                print("does not initiate any reaction.")

    def getODEs(self, method = 'massAction'):
        """Get symbolic equations for the model system of ODEs.


        Parameters
        ----------
        method : str, optional
            Can be ``'massAction'`` (default) or ``'vanKampen'``.

        Returns
        -------
        :class:`dict`
            Dictionary of ODE right hand sides with reactant (left hand side) as key

        """

        if method == 'massAction':
            return self._equations
        elif method == 'vanKampen':
            return _getODEs_vKE(_get_orderedLists_vKE, self._stoichiometry)
        else:
            print('Invalid input for method. Choose either method = \'massAction\' or method = \'vanKampen\'. Default is \'massAction\'.')


    def showODEs(self, method = 'massAction'):
        """Show a LaTeX representation of the model system of ODEs.

        Displays rendered LaTeX in the Jupyter Notebook.

        Parameters
        ----------
        method : str, optional
            Can be ``'massAction'`` (default) or ``'vanKampen'``.
            
        Returns
        -------
            `None`

        """

        if method == 'massAction':
            for reactant in self._reactants:
                out = "\\displaystyle \\frac{\\textrm{d}" + latex(reactant) + "}{\\textrm{d}t} := " + latex(self._equations[reactant])
                out = _doubleUnderscorify(_greekPrependify(out))
                display(Math(out))
        elif method == 'vanKampen':
            ODEdict = _getODEs_vKE(_get_orderedLists_vKE, self._stoichiometry)
            for ode in ODEdict:
                out = latex(ode) + " := " + latex(ODEdict[ode])
                out = _doubleUnderscorify(_greekPrependify(out))
                display(Math(out))
        else:
            print('Invalid input for method. Choose either method = \'massAction\' or method = \'vanKampen\'. Default is \'massAction\'.')


    def getStoichiometry(self):
        """Get stoichiometry as a dictionary

        Returns
        -------
        :class:`dict`
            Dictionary  with key ReactionNr; ReactionNr represents another dictionary with reaction rate, reactants
            and corresponding stoichiometry.

        """

        return self._stoichiometry

    def showStoichiometry(self):
        """Display stoichiometry as a dictionary with keys ReactionNr,
        ReactionNr represents another dictionary with reaction rate, reactants
        and corresponding stoichiometry.

        Displays rendered LaTeX in the Jupyter Notebook.
        
        Returns
        -------
            `None`

        """
        out = latex(self._stoichiometry)
        out = _doubleUnderscorify(_greekPrependify(out))
        display(Math(out))

    def getMasterEquation(self):
        """Gets Master Equation expressed with step operators, and substitutions.

        Returns
        -------
        :class:`dict`, :class:`dict`
            Dictionary showing all terms of the right hand side of the Master Equation
            Dictionary of substitutions used, this defaults to `None` if no substitutions were made

        """

        P, t = symbols('P t')
        out_rhs = ""
        stoich = self._stoichiometry
        nvec = []
        for key1 in stoich:
            for key2 in stoich[key1]:
                if key2 != 'rate' and stoich[key1][key2] != 'const':
                    if not key2 in nvec:
                        nvec.append(key2)
        nvec = sorted(nvec, key=default_sort_key)
        
        if len(nvec) < 1 or len(nvec) > 4:
            print("Derivation of Master Equation works for 1, 2, 3 or 4 different reactants only")
            
            return
#        assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'
        rhs_dict, substring = _deriveMasterEquation(stoich)

        return rhs_dict, substring


    def showMasterEquation(self):
        """Displays Master equation expressed with step operators.

        Displays rendered LaTeX in the Jupyter Notebook.
        
        Returns
        -------
            `None`

        """

        P, t = symbols('P t')
        out_rhs = ""
        stoich = self._stoichiometry
        nvec = []
        for key1 in stoich:
            for key2 in stoich[key1]:
                if key2 != 'rate' and stoich[key1][key2] != 'const':
                    if not key2 in nvec:
                        nvec.append(key2)
        nvec = sorted(nvec, key=default_sort_key)
        
        if len(nvec) < 1 or len(nvec) > 4:
            print("Derivation of Master Equation works for 1, 2, 3 or 4 different reactants only")
            
            return
#        assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'
        rhs_dict, substring = _deriveMasterEquation(stoich)

        #rhs_ME = 0
        term_count = 0
        for key in rhs_dict:
            #rhs_ME += terms_gcd(key*(rhs_dict[key][0]-1)*rhs_dict[key][1]*rhs_dict[key][2], deep=True)
            if term_count == 0:
                rhs_plus = ""
            else:
                rhs_plus = " + "
            out_rhs += rhs_plus + latex(rhs_dict[key][3]) + " ( " + latex((rhs_dict[key][0]-1)) + " ) " + latex(rhs_dict[key][1]) + " " + latex(rhs_dict[key][2])
            term_count += 1
        
        if len(nvec) == 1:
            lhs_ME = Derivative(P(nvec[0], t), t)
        elif len(nvec) == 2:
            lhs_ME = Derivative(P(nvec[0], nvec[1], t), t)
        elif len(nvec) == 3:
            lhs_ME = Derivative(P(nvec[0], nvec[1], nvec[2], t), t)
        else:
            lhs_ME = Derivative(P(nvec[0], nvec[1], nvec[2], nvec[3], t), t)     
        
        #return {lhs_ME: rhs_ME}
        out = latex(lhs_ME) + ":= " + out_rhs
        out = _doubleUnderscorify(out)
        out = _greekPrependify(out)
        display(Math(out))
        #substring is a dictionary
        if not substring is None:
            for subKey, subVal in substring.items():
                subK = _greekPrependify(_doubleUnderscorify(str(subKey)))
                subV = _greekPrependify(_doubleUnderscorify(str(subVal)))
                display(Math("With \; substitution:\;" + latex(subK) + ":= " + latex(subV)))


    def getVanKampenExpansion(self):
        """Get van Kampen expansion when the operators are expanded up to
        second order.

        Returns
        -------
        :class:`Add`, :class:`Add`, :class:`dict`
            van Kampen expansion left hand side
            van Kampen expansion right hand side
            Dictionary of substitutions used, this defaults to `None` if no substitutions were made
        """

        rhs_vke, lhs_vke, substring = _doVanKampenExpansion(_deriveMasterEquation, self._stoichiometry)

        return lhs_vke, rhs_vke, substring


    def showVanKampenExpansion(self):
        """Show van Kampen expansion when the operators are expanded up to
        second order.

        Displays rendered LaTeX in the Jupyter Notebook.
        
        Returns
        -------
            `None`

        """
        rhs_vke, lhs_vke, substring = _doVanKampenExpansion(_deriveMasterEquation, self._stoichiometry)
        out = latex(lhs_vke) + " := \n" + latex(rhs_vke)
        out = _doubleUnderscorify(_greekPrependify(out))
        display(Math(out))
        #substring is a dictionary
        if not substring is None:
            for subKey, subVal in substring.items():
                subK = _greekPrependify(_doubleUnderscorify(str(subKey)))
                subV = _greekPrependify(_doubleUnderscorify(str(subVal)))
                display(Math("With \; substitution:\;" + latex(subK) + ":= " + latex(subV)))


    def getFokkerPlanckEquation(self):
        """Get Fokker-Planck equation derived from term ~ O(1) in van Kampen
        expansion (linear noise approximation).

        Returns
        -------
        :class:`dict`, :class:`dict`
            Dictionary of Fokker-Planck right hand sides
            Dictionary of substitutions used, this defaults to `None` if no substitutions were made

        """

        return _getFokkerPlanckEquation(_get_orderedLists_vKE, self._stoichiometry)


    def showFokkerPlanckEquation(self):
        """Show Fokker-Planck equation derived from term ~ O(1) in van Kampen
        expansion (linear noise approximation).

        Displays rendered LaTeX in the Jupyter Notebook.
        
        Returns
        -------
            `None`

        """
        FPEdict, substring = _getFokkerPlanckEquation(_get_orderedLists_vKE, self._stoichiometry)
        for fpe in FPEdict:
            out = latex(fpe) + " := " + latex(FPEdict[fpe])
            out = _doubleUnderscorify(_greekPrependify(out))
            display(Math(out))
        #substring is a dictionary
        if not substring is None:
            for subKey, subVal in substring.items():
                subK = _greekPrependify(_doubleUnderscorify(str(subKey)))
                subV = _greekPrependify(_doubleUnderscorify(str(subVal)))
                display(Math("With \; substitution:\;" + latex(subK) + ":= " + latex(subV)))


    def getNoiseEquations(self):
        """Get equations of motion of first and second order moments of noise.

        Returns
        -------
        :class:`dict`, :class:`dict`, :class:`dict`, :class:`dict`
            Dictionary of first order moments equations of motion right hand sides (derived using Fokker-Planck equation)
            Dictionary of substitutions used for first order moments equations
            Dictionary of second order moments equations of motion right hand sides (derived using Fokker-Planck equation)
            Dictionary of substitutions used for second order moments equations

        """
        EQsys1stOrdMom, EOM_1stOrderMom, NoiseSubs1stOrder, EQsys2ndOrdMom, EOM_2ndOrderMom, NoiseSubs2ndOrder = _getNoiseEOM(_getFokkerPlanckEquation, _get_orderedLists_vKE, self._stoichiometry)

        return EOM_1stOrderMom, NoiseSubs1stOrder, EOM_2ndOrderMom, NoiseSubs2ndOrder


    def showNoiseEquations(self):
        """Display equations of motion of first and second order moments of noise.

        Displays rendered LaTeX in the Jupyter Notebook.
        
        Returns
        -------
            `None`

        """
        EQsys1stOrdMom, EOM_1stOrderMom, NoiseSubs1stOrder, EQsys2ndOrdMom, EOM_2ndOrderMom, NoiseSubs2ndOrder = _getNoiseEOM(_getFokkerPlanckEquation, _get_orderedLists_vKE, self._stoichiometry)
        for eom1 in EOM_1stOrderMom:
            out = "\\displaystyle \\frac{\\textrm{d}" + latex(eom1.subs(NoiseSubs1stOrder)) + "}{\\textrm{d}t} := " + latex(EOM_1stOrderMom[eom1].subs(NoiseSubs1stOrder))
            out = _doubleUnderscorify(out)
            out = _greekPrependify(out)
            display(Math(out))
        for eom2 in EOM_2ndOrderMom:
            out = "\\displaystyle \\frac{\\textrm{d}" + latex(eom2.subs(NoiseSubs2ndOrder)) + "}{\\textrm{d}t} := " + latex(EOM_2ndOrderMom[eom2].subs(NoiseSubs2ndOrder))
            out = _doubleUnderscorify(out)
            out = _greekPrependify(out)
            display(Math(out))


    def getNoiseSolutions(self):
        """Gets noise in the stationary state.

        Returns
        -------
        :class:`dict`, :class:`dict`, :class:`dict`, :class:`dict`
            Dictionary of first order moments noise solution right hand sides
            Dictionary of substitutions used for first order moments solutions
            Dictionary of second order moments noise solution right hand sides
            Dictionary of substitutions used for second order moments solutions
        """
        return _getNoiseStationarySol(_getNoiseEOM, _getFokkerPlanckEquation, _get_orderedLists_vKE, self._stoichiometry)


    def showNoiseSolutions(self):
        """Display noise in the stationary state.

        Displays rendered LaTeX in the Jupyter Notebook.
        
        Returns
        -------
            `None`
            
        """
        
        SOL_1stOrderMom, NoiseSubs1stOrder, SOL_2ndOrdMomDict, NoiseSubs2ndOrder = _getNoiseStationarySol(_getNoiseEOM, _getFokkerPlanckEquation, _get_orderedLists_vKE, self._stoichiometry)
        print('Stationary solutions of first and second order moments of noise:')
        if SOL_1stOrderMom is None:
            print('Noise 1st-order moments could not be calculated analytically.')
            return None
        else:
            for sol1 in SOL_1stOrderMom:
                out = latex(sol1.subs(NoiseSubs1stOrder)) + latex(r'(t \to \infty)') + ":= " + latex(SOL_1stOrderMom[sol1].subs(NoiseSubs1stOrder))
                out = _doubleUnderscorify(_greekPrependify(out))
                display(Math(out))
        if SOL_2ndOrdMomDict is None:
            print('Noise 2nd-order moments could not be calculated analytically.')
            return None
        else:
            for sol2 in SOL_2ndOrdMomDict:
                out = latex(sol2.subs(NoiseSubs2ndOrder)) + latex(r'(t \to \infty)') + " := " + latex(SOL_2ndOrdMomDict[sol2].subs(NoiseSubs2ndOrder))
                out = _doubleUnderscorify(_greekPrependify(out))
                display(Math(out))


    def show(self):
        """Show a LaTeX representation of the model.

        Display all rules in the model as rendered LaTeX in the Jupyter
        Notebook.
        
        Returns
        -------
            `None`

        Notes
        -----
        If rules are rendered in the Notebook with extraneous ``|`` characters
        on the right-hand-side then try:

         * Switching web browser and/or
         * Updating the ``notebook`` package:
           ``pip install --upgrade --no-deps notebook``

        """
        for rule in self._rules:
            out = ""
            for reactant in rule.lhsReactants:
                if reactant == EMPTYSET_SYMBOL:
                    reactant = Symbol('\emptyset')
                if reactant in self._constantReactants:
                    out += "("
                out += latex(reactant)
                if reactant in self._constantReactants:
                    out += ")"                 
                out += " + "
            out = out[0:len(out) - 2]  # delete the last ' + '
            out += " \\xrightarrow{" + latex(rule.rate) + "}"
            for reactant in rule.rhsReactants:
                if reactant == EMPTYSET_SYMBOL:
                    reactant = Symbol('\emptyset')
                if reactant in self._constantReactants:
                    out += "(" 
                out += latex(reactant)
                if reactant in self._constantReactants:
                    out += ")"                 
                out += " + "
            out = out[0:len(out) - 2]  # delete the last ' + '
            out = _doubleUnderscorify(_greekPrependify(out))
            display(Math(out))

    def integrate(self, showStateVars=None, initWidgets=None, **kwargs):
        """Construct interactive time evolution plot for state variables.

        Parameters
        ----------
        showStateVars : optional
            Select reactants (state variables) to be shown in plot, if not
            specified all reactants are plotted.  Type?
        initWidgets : dict, optional
            Dictionary where keys are the free-parameter or any other specific
            parameter, and values are four values, e.g.
            ``'parameter':[initial-value, min-value, max-value, step-size]``

        Keywords
        --------
        maxTime : float, optional
            Simulation time for noise correlations.  Must be strictly positive.
            Defaults to 3.0.
        tstep : float, optional
            Time step of numerical integration of reactants.  Defaults to 0.01.
        plotProportions : bool, optional
            Flag to plot proportions or full populations.  Defaults to False
        initialState : dict, optional
            Initial proportions of the reactants (type: float in range [0,1]),
            can also be set via ``initWidgets`` argument.  Defaults to an empty
            dictionary.
        conserved : bool, optional
            Specify if a system is conserved to make proportions of state
            variables (at time t=0) sum up to 1, or not. Defaults to False.
        legend_fontsize: int, optional
            Specify fontsize of legend.  Defaults to 14.
        legend_loc : str
            Specify legend location: combinations like 'upper left' (default),
            'lower right', or 'center center' are allowed (9 options in total).
        fontsize : integer, optional
            Specify fontsize for axis-labels.  If not specified, the fontsize
            is automatically derived from the length of axis label.
        xlab : str, optional
            Specify label on x-axis.   Defaults to 'time t'.
        ylab : str, optional
            Specify label on y-axis.   Defaults to 'reactants'.
        choose_xrange : list of float, optional
            Specify range plotted on x-axis as a two-element iterable of the
            form [xmin, xmax]. If not given uses data values to set axis
            limits.
        choose_trange : list of float, optional
            Specify range plotted on y-axis as a two-element iterable of the
            form [ymin, ymax]. If not given uses data values to set axis
            limits.
        silent : bool, optional
            Switch on/off widgets and plot. Important for use with multi
            controllers. Defaults to False.

        Returns
        -------
            :class:`MuMoTtimeEvolutionController`

        Notes
        -----
        Plotting keywords are also described in the `user manual`_.

        .. _user manual: https://mumot.readthedocs.io/en/latest/getting_started.html

        """
        if initWidgets is None:
            initWidgets = {}
        
        if self._systemSize is not None:
            kwargs['conserved'] = True
        
        paramValuesDict = self._create_free_param_dictionary_for_controller(inputParams=kwargs.get('params', []), initWidgets=initWidgets, showSystemSize=True, showPlotLimits=False)
        
        IntParams = {}
        # read input parameters
        IntParams['initialState'] = _format_advanced_option(optionName='initialState', inputValue=kwargs.get('initialState'), initValues=initWidgets.get('initialState'), extraParam=self._getAllReactants())
        IntParams['maxTime'] = _format_advanced_option(optionName='maxTime', inputValue=kwargs.get('maxTime'), initValues=initWidgets.get('maxTime'))
        IntParams['plotProportions'] = _format_advanced_option(optionName='plotProportions', inputValue=kwargs.get('plotProportions'), initValues=initWidgets.get('plotProportions'))
        IntParams['conserved'] = [kwargs.get('conserved', False), True]
        IntParams['substitutedReactant'] = [ [react for react in self._getAllReactants()[0] if react not in self._reactants][0] if self._systemSize is not None else None, True]
        
        # construct controller
        viewController = MuMoTtimeEvolutionController(paramValuesDict=paramValuesDict, paramLabelDict=self._ratesLaTeX, continuousReplot=False, advancedOpts=IntParams, showSystemSize=True, **kwargs)
        
        #if showStateVars:
        #    showStateVars = [r'' + showStateVars[kk] for kk in range(len(showStateVars))]
        
        modelView = MuMoTintegrateView(self, viewController, IntParams, showStateVars, **kwargs)
        viewController._setView(modelView)
        
        viewController._setReplotFunction(modelView._plot_NumSolODE, modelView._redrawOnly)
        
        return viewController
    
        
    def noiseCorrelations(self, initWidgets=None, **kwargs):
        """Construct interactive time evolution plot for noise correlations
        around fixed points.

        Parameters
        ----------
        initWidgets : dict, optional
            Dictionary where keys are the free-parameter or any other specific
            parameter, and values are four values, e.g.
            ``'parameter': [initial-value, min-value, max-value, step-size]``.

        Keywords
        --------
        maxTime : float, optional
            Simulation time for noise correlations.  Must be strictly positive.
            Defaults to 3.0.
        tstep : float, optional
            Time step of numerical integration of noise correlations.  Defaults
            to 0.01.
        maxTimeDS : float, optional
            Simulation time for ODE system.  Must be strictly positive.
            Defaults to 50.
        tstepDS : float, optional
            Time step of numerical integration of ODE system.  Defaults to
            0.01.
        initialState : dict, optional
            Initial proportions of the reactants.  Must be in range [0, 1].
            Can also be set via ``initWidgets`` argument.
            Defaults to an empty dictionary.
        conserved : bool, optional
            Whether a system is conserved to make proportions of state
            variables (at time t=0) sum up to 1.  Defaults to False.
        legend_fontsize : int, optional
            Font size of legend.  Defaults to 14.
        legend_loc : str, optional
            Legend location.  Combinations like ``'upper left'``, ``'lower
            right'``, or ``'center center'`` are allowed (9 options in total).
            Defaults to ``'upper left'``.
        fontsize : int, optional
            Font size for axis labels  If not given, font size is automatically
            derived from length of axis label.
        xlab = str, optional
            Label on x-axis.  Defaults to ``'time t'``.
        ylab : str, optional
            Label on y-axis.  Defaults to ``'noise correlations'``.
        choose_xrange : list of float, optional
            Range to be plotted on x-axis.  Specified as ``[xmin, xmax]``.  If
            not given uses data values to set axis limits.
        choose_yrange : list of float, optional
            Range to be plotted on y-axis.  Specified as ``[ymin, ymax]``.  If
            not given uses data values to set axis limits.
        silent : bool, optional
            Switch on/off widgets and plot.  Important for use with multi
            controllers.  Defaults to False.

        Returns
        -------
            :class:`MuMoTtimeEvolutionController`

        Notes
        -----
        Plotting keywords are also described in the `user manual`_.

        .. _user manual: https://mumot.readthedocs.io/en/latest/getting_started.html

        """
        if initWidgets is None:
            initWidgets = {}

        if self._systemSize is not None:
            kwargs['conserved'] = True
        
        paramValuesDict = self._create_free_param_dictionary_for_controller(inputParams=kwargs.get('params', []), initWidgets=initWidgets, showSystemSize=True, showPlotLimits=False)
        
        NCParams = {}
        # read input parameters
        NCParams['initialState'] = _format_advanced_option(optionName='initialState', inputValue=kwargs.get('initialState'), initValues=initWidgets.get('initialState'), extraParam=self._getAllReactants())
        NCParams['maxTime'] = _format_advanced_option(optionName='maxTime', inputValue=kwargs.get('maxTime'), initValues=initWidgets.get('maxTime'))
        NCParams['conserved'] = [kwargs.get('conserved', False), True]
        NCParams['substitutedReactant'] = [ [react for react in self._getAllReactants()[0] if react not in self._reactants][0] if self._systemSize is not None else None, True]
        
        EQsys1stOrdMom, EOM_1stOrderMom, NoiseSubs1stOrder, EQsys2ndOrdMom, EOM_2ndOrderMom, NoiseSubs2ndOrder = _getNoiseEOM(_getFokkerPlanckEquation, _get_orderedLists_vKE, self._stoichiometry)
        
        # construct controller
        viewController = MuMoTtimeEvolutionController(paramValuesDict=paramValuesDict, paramLabelDict=self._ratesLaTeX, continuousReplot=False, advancedOpts=NCParams, showSystemSize=True, **kwargs)
        
        modelView = MuMoTnoiseCorrelationsView(self, viewController, NCParams, EOM_1stOrderMom, EOM_2ndOrderMom, **kwargs)
        
        viewController._setView(modelView)
        
        viewController._setReplotFunction(modelView._plot_NumSolODE)
        
        return viewController

    
    def _check2ndOrderMom(self, showNoise=False):
        """Check if 2nd Order moments of noise-noise correlations can be calculated via Master equation and Fokker-Planck equation"""
        
        if showNoise == True:
            substitutions = False
            for reaction in self._stoichiometry:
                for key in self._stoichiometry[reaction]:
                    if key != 'rate':
                        if self._stoichiometry[reaction][key] != 'const':
                            if len(self._stoichiometry[reaction][key]) > 2:
                                substitutions = True
            
            if substitutions == False:
                SOL_1stOrderMomDict, NoiseSubs1stOrder, SOL_2ndOrdMomDict, NoiseSubs2ndOrder = _getNoiseStationarySol(_getNoiseEOM, _getFokkerPlanckEquation, _get_orderedLists_vKE, self._stoichiometry)
                # SOL_2ndOrdMomDict is second order solution and will be used by MuMoTnoiseView
            else:
                SOL_2ndOrdMomDict = None
            
#                 if SOL_2ndOrdMomDict is None:
#                     if substitutions == True:
#                         print('Noise in stream plots is only available for systems with exactly two time dependent reactants. Stream plot only works WITHOUT noise in case of reducing number of state variables from 3 to 2 via substitute() - method.')
#                     print('Warning: Noise in the system could not be calculated: \'showNoise\' automatically disabled.')
#                     kwargs['showNoise'] = False
        else:
            SOL_2ndOrdMomDict = None
            
        return SOL_2ndOrdMomDict

    # construct interactive stream plot with the option to show noise around
    # fixed points
    def stream(self, stateVariable1, stateVariable2, stateVariable3=None,
               params=None, initWidgets=None, **kwargs):
        """Display interactive stream plot of ``stateVariable1`` (x-axis),
        ``stateVariable2`` (y-axis), and optionally ``stateVariable3`` (z-axis;
        not currently supported - see below).

        Parameters
        ----------
        stateVariable1
            State variable to be plotted on the x-axis. Type?
        stateVariable2
            State variable to be plotted on the y-axis. Type?
        stateVariable3 : optional
            State variable to be plotted on the z-axis.  Not currently
            supported; use `vector` instead for 3-dimensional systems.  Type?
        params : optional
            Parameter list (see 'Partial controllers' in the `user manual`_).
            Type?
        initWidgets : dict, optional
             Keys are the free parameter or any other specific parameter, and
             values each a list of ``[initial-value, min-value, max-value,
             step-size]``.  

        Keywords
        --------
        showFixedPoints : bool, optional
             Plot fixed points.  Defaults to False.
        showNoise : bool, optional
             Plot noise around fixed points.  Defaults to False.
        runs : int, optional
           Number of simulation runs to be executed. Must be strictly positive. Defaults to 1.
        aggregateResults : bool, optional
           Flag to aggregate or not the results from several runs. Defaults to True.
        fontsize : int, optional
             Font size for axis-labels.
        xlab : str, optional
             Label for x-axis.  Defaults to ``'stateVariable1'``.
        ylab : str, optional
             Label for y-axis.  Defaults to ``'stateVariable2'``.
        choose_xrange : list of float, optional
             Range plotted on x-axis.  Specify as ``[xmin, xmax]``.  If not
             given uses data values to set axis limits
        choose_yrange : list of float, optional
             Range plotted on y-axis.  Specify as ``[ymin, ymax]``.  If not
             given uses data values to set axis limits
        silent : bool, optional
             Switch on/off widgets and plot.  Important for use with multi
             controllers.  Defaults to False.

        Returns
        -------
        :class:`MuMoTcontroller`
            A MuMoT controller object

        Notes
        -----
        Plotting keywords are also described in the `user manual`_.

        .. _user manual: https://mumot.readthedocs.io/en/latest/getting_started.html

        """
        if initWidgets is None:
            initWidgets = {}

        if self._systemSize is None and self._constantSystemSize == True:  # duplicate check in view and controller required for speedy error reporting, plus flexibility to instantiate view independent of controller
            print("Cannot construct field-based plot until system size is set, using substitute()")
            return None

        if self._check_state_variables(stateVariable1, stateVariable2, stateVariable3):
            if stateVariable3 is None:
                SOL_2ndOrdMomDict = self._check2ndOrderMom(showNoise=kwargs.get('showNoise', False))
            else:
                print('3D stream plot not yet implemented.')
                #SOL_2ndOrdMomDict = None
                return None
            
            continuous_update = not (kwargs.get('showNoise', False) or kwargs.get('showFixedPoints', False))
            showNoise = kwargs.get('showNoise', False)                 
            showSystemSize = showNoise 
            plotLimitsSlider = not(self._constantSystemSize)
            paramValuesDict = self._create_free_param_dictionary_for_controller(inputParams=params if params is not None else [], initWidgets=initWidgets, showSystemSize=showSystemSize, showPlotLimits=plotLimitsSlider)
            
            advancedOpts = {} 
            # read input parameters
            advancedOpts['maxTime'] = _format_advanced_option(optionName='maxTime', inputValue=kwargs.get('maxTime'), initValues=initWidgets.get('maxTime'), extraParam="asNoise")
            advancedOpts['randomSeed'] = _format_advanced_option(optionName='randomSeed', inputValue=kwargs.get('randomSeed'), initValues=initWidgets.get('randomSeed'))
            # next line to address issue #283
            #advancedOpts['plotProportions'] = _format_advanced_option(optionName='plotProportions', inputValue=kwargs.get('plotProportions'), initValues=initWidgets.get('plotProportions'))
            # next lines useful to address issue #95
            #advancedOpts['final_x'] = _format_advanced_option(optionName='final_x', inputValue=kwargs.get('final_x'), initValues=initWidgets.get('final_x'), extraParam=self._getAllReactants()[0])
            #advancedOpts['final_y'] = _format_advanced_option(optionName='final_y', inputValue=kwargs.get('final_y'), initValues=initWidgets.get('final_y'), extraParam=self._getAllReactants()[0])
            advancedOpts['runs'] = _format_advanced_option(optionName='runs', inputValue=kwargs.get('runs'), initValues=initWidgets.get('runs'), extraParam="asNoise")
            advancedOpts['aggregateResults'] = _format_advanced_option(optionName='aggregateResults', inputValue=kwargs.get('aggregateResults'), initValues=initWidgets.get('aggregateResults'))
            
            # construct controller
            viewController = MuMoTfieldController(paramValuesDict=paramValuesDict, paramLabelDict=self._ratesLaTeX, continuousReplot=continuous_update, showPlotLimits=plotLimitsSlider, advancedOpts=advancedOpts, showSystemSize=showSystemSize, **kwargs)

            # construct view
            modelView = MuMoTstreamView(self, viewController, advancedOpts, SOL_2ndOrdMomDict, stateVariable1, stateVariable2, stateVariable3, params=params, **kwargs)

            viewController._setView(modelView)
            viewController._setReplotFunction(modelView._plot_field)

            return viewController
        else:
            return None

    # construct interactive vector plot with the option to show noise around
    # fixed points
    def vector(self, stateVariable1, stateVariable2, stateVariable3=None,
               params=None, initWidgets=None, **kwargs):
        """Display interactive stream plot of ``stateVariable1`` (x-axis),
        ``stateVariable2`` (y-axis), and optionally ``stateVariable3`` (z-axis;
        not currently supported - see below)

        Parameters
        ----------
        stateVariable1
            State variable to be plotted on the x-axis. Type?
        stateVariable2
            State variable to be plotted on the y-axis. Type?
        stateVariable3 : optional
            State variable to be plotted on the z-axis.  Type?
        params : optional
            Parameter list (see 'Partial controllers' in the `user manual`_).
            Type?
        initWidgets : dict, optional
             Keys are the free parameter or any other specific parameter, and
             values each a list of ``[initial-value, min-value, max-value,
             step-size]``.  

        Keywords
        --------
        showFixedPoints : bool, optional
             Plot fixed points.  Defaults to False.
        showNoise : bool, optional
             Plot noise around fixed points.  Defaults to False.
        runs : int, optional
           Number of simulation runs to be executed. Must be strictly positive. Defaults to 1.
        aggregateResults : bool, optional
           Flag to aggregate or not the results from several runs. Defaults to True.
        fontsize : int, optional
             Font size for axis-labels.
        xlab : str, optional
             Label for x-axis.  Defaults to ``'stateVariable1'``.
        ylab : str, optional
             Label for y-axis.  Defaults to ``'stateVariable2'``.
        zlab : str, optional
             Label for z-axis (3D plots only).  Defaults to ``'stateVariable3'``.
        choose_xrange : list of float, optional
             Range plotted on x-axis.  Specify as ``[xmin, xmax]``.  If not
             given uses data values to set axis limits
        choose_yrange : list of float, optional
             Range plotted on y-axis.  Specify as ``[ymin, ymax]``.  If not
             given uses data values to set axis limits
        silent : bool, optional
             Switch on/off widgets and plot.  Important for use with multi
             controllers.  Defaults to False.

        Returns
        -------
        :class:`MuMoTcontroller`
            A MuMoT controller object
        
        Notes
        -----
        Plotting keywords are also described in the `user manual`_.

        .. _user manual: https://mumot.readthedocs.io/en/latest/getting_started.html

        """
        if initWidgets is None:
            initWidgets = {}

        if self._systemSize is None and self._constantSystemSize == True:  # duplicate check in view and controller required for speedy error reporting, plus flexibility to instantiate view independent of controller
            print("Cannot construct field-based plot until system size is set, using substitute()")
            return None

        if self._check_state_variables(stateVariable1, stateVariable2, stateVariable3):
            if stateVariable3 is None:
                SOL_2ndOrdMomDict = self._check2ndOrderMom(showNoise=kwargs.get('showNoise', False))
            else:
                SOL_2ndOrdMomDict = None
                    
            continuous_update = not (kwargs.get('showNoise', False) or kwargs.get('showFixedPoints', False))
            showNoise = kwargs.get('showNoise', False)                 
            showSystemSize = showNoise 
            plotLimitsSlider = not(self._constantSystemSize)
            paramValuesDict = self._create_free_param_dictionary_for_controller(inputParams=params if params is not None else [], initWidgets=initWidgets, showSystemSize=showSystemSize, showPlotLimits=plotLimitsSlider)
            
            advancedOpts = {} 
            # read input parameters
            advancedOpts['maxTime'] = _format_advanced_option(optionName='maxTime', inputValue=kwargs.get('maxTime'), initValues=initWidgets.get('maxTime'), extraParam="asNoise")
            advancedOpts['randomSeed'] = _format_advanced_option(optionName='randomSeed', inputValue=kwargs.get('randomSeed'), initValues=initWidgets.get('randomSeed'))
            # next line to address issue #283
            #advancedOpts['plotProportions'] = _format_advanced_option(optionName='plotProportions', inputValue=kwargs.get('plotProportions'), initValues=initWidgets.get('plotProportions'))
            # next lines useful to address issue #95
            #advancedOpts['final_x'] = _format_advanced_option(optionName='final_x', inputValue=kwargs.get('final_x'), initValues=initWidgets.get('final_x'), extraParam=self._getAllReactants()[0])
            #advancedOpts['final_y'] = _format_advanced_option(optionName='final_y', inputValue=kwargs.get('final_y'), initValues=initWidgets.get('final_y'), extraParam=self._getAllReactants()[0])
            advancedOpts['runs'] = _format_advanced_option(optionName='runs', inputValue=kwargs.get('runs'), initValues=initWidgets.get('runs'), extraParam="asNoise")
            advancedOpts['aggregateResults'] = _format_advanced_option(optionName='aggregateResults', inputValue=kwargs.get('aggregateResults'), initValues=initWidgets.get('aggregateResults'))
            
            # construct controller
            viewController = MuMoTfieldController(paramValuesDict=paramValuesDict, paramLabelDict=self._ratesLaTeX, continuousReplot=continuous_update, showPlotLimits=plotLimitsSlider, advancedOpts=advancedOpts, showSystemSize=showSystemSize, **kwargs)
            
            # construct view
            modelView = MuMoTvectorView(self, viewController, advancedOpts, SOL_2ndOrdMomDict, stateVariable1, stateVariable2, stateVariable3, params=params, **kwargs)
                    
            viewController._setView(modelView)
            viewController._setReplotFunction(modelView._plot_field)         
            
            return viewController
        else:
            return None

    def bifurcation(self, bifurcationParameter, stateVariable1,
                    stateVariable2=None, initWidgets=None, **kwargs):
        """Construct and display bifurcation plot of ``stateVariable1``
        (y-axis), or ``stateVariable1`` +/- ``stateVariable2`` (y-axis),
        depending on ``bifurcationParameter`` (x-axis).

        1D and 2D systems are currently supported.  Only limit points and
        branch points can be detected.

        Parameters
        ----------
        bifurcationParameter
            Critical parameter plotted on x-axis.  Type?
        stateVariable1
            State variable expression to be plotted on the y-axis; allowed are:
            reactant1, reactant1-reactant2, or reactant1+reactant2.  Type?
        stateVariable2 : optional
            State variable if system is larger than 2D (not currently
            supported).  Type?
        initWidgets : dict, optional
            Keys are the free-parameter or any other specific parameter, and
            values are four values, e.g. ``'parameter': [initial-value,
            min-value, max-value, step-size]``.  

        Keywords
        --------
        initialState : dict, optional
            Initial proportions of the reactants. State variables are the keys the
            values of which must be in range [0, 1].
            Can also be set via ``initWidgets`` argument.
            Defaults to an empty dictionary.
        initBifParam : float, optional
            Initial value of bifurcation parameter.  Can also be set via
            ``initWidgets`` argument.  Defaults to 2.0.
        conserved : bool, optional
            Whether a system is conserved to make proportions of state
            variables (at time t=0) sum up to 1, or not.  Defaults to False.
        contMaxNumPoints: int, optional
            Maximum number of continuation points.  Defaults to 100.
        fontsize : int, optional
            Font size for axis labels.  If not given, font size is
            automatically derived from length of axis label.
        xlab : str, optional
            Label for x-axis. If not given uses symbol for
            ``bifurcationParameter`` in arguments as default label.
        ylab : str, optional
            Label for y-axis.  If not given, defaults to
            ``'\Phi_{stateVariable1}'``, where ``stateVariable1`` is another
            argument to this method.  If ``stateVariable1`` is a sum/difference
            of the form 'Reactant1 +/- Reactant2' the default label is
            ``'\Phi_{Reactant1} - \Phi_{Reactant2}'``.
        choose_xrange : list of float, optional
            Range plotted on x-axis as ``[xmin, xmax]``.  If not given,
            uses data values to set axis limits.
        choose_yrange : list of float, optional
            Range plotted on y-axis as ``[ymin, ymax]``.  If not given,
            uses data values to set axis limits.
        silent : bool, optional
            Switch on/off widgets and plot. Important for use with multi
            controllers.

        Returns
        -------
            :class:`MuMoTbifurcationController`

        Notes
        -----
        Plotting keywords are also described in the `user manual`_.

        .. _user manual: https://mumot.readthedocs.io/en/latest/getting_started.html

        """
        if initWidgets is None:
            initWidgets = {}
        stateVariableList = []
        for reactant in self._reactants:
            if reactant not in self._constantReactants:
                stateVariableList.append(reactant)
        if len(stateVariableList) > 2:
            print('Sorry, bifurcation diagrams are currently only supported for 1D and 2D systems (1 or 2 time-dependent variables in the ODE system)!')
            return None
        
        conserved = kwargs.get('conserved', False)
        #check for substitutions of state variables in conserved systems
        stoich = self._stoichiometry
        for key1 in stoich:
            for key2 in stoich[key1]:
                if key2 != 'rate' and stoich[key1][key2] != 'const':
                    if len(stoich[key1][key2]) == 3:
                        conserved = True
        
        #if bifurcationParameter[0]=='\\':
        #        bifPar = bifurcationParameter[1:]
        #else:
        #    bifPar = bifurcationParameter
        bifPar = bifurcationParameter
        #if self._systemSize:
        #    kwargs['conserved'] = True
                    
        paramValuesDict = self._create_free_param_dictionary_for_controller(inputParams=kwargs.get('params', []), initWidgets=initWidgets, showSystemSize=False, showPlotLimits=False)
        
        if str(process_sympy(bifPar)) in paramValuesDict: 
            del paramValuesDict[str(process_sympy(bifPar))]
        
        BfcParams = {}
        # read input parameters
        BfcParams['initBifParam'] = _format_advanced_option(optionName='initBifParam', inputValue=kwargs.get('initBifParam'), initValues=initWidgets.get('initBifParam'))
        BfcParams['initialState'] = _format_advanced_option(optionName='initialState', inputValue=kwargs.get('initialState'), initValues=initWidgets.get('initialState'), extraParam=self._getAllReactants())
        BfcParams['bifurcationParameter'] = [bifPar, True]
        BfcParams['conserved'] = [conserved, True]
        BfcParams['substitutedReactant'] = [ [react for react in self._getAllReactants()[0] if react not in self._reactants][0] if self._systemSize is not None else None, True]
        
        # construct controller
        viewController = MuMoTbifurcationController(paramValuesDict=paramValuesDict, paramLabelDict=self._ratesLaTeX, continuousReplot=False, advancedOpts=BfcParams, showSystemSize=False, **kwargs)
        
        #if showStateVars:
        #    showStateVars = [r'' + showStateVars[kk] for kk in range(len(showStateVars))]
        
        modelView = MuMoTbifurcationView(self, viewController, BfcParams, bifurcationParameter, stateVariable1, stateVariable2, **kwargs)
        viewController._setView(modelView)
        
        viewController._setReplotFunction(modelView._plot_bifurcation)
        
        return viewController
         
    
    def multiagent(self, initWidgets=None, **kwargs):
        """Construct interactive multiagent plot (simulation of agents locally interacting with each other).
        
        Parameters
        ----------
        initWidgets : dict {str,list}, optional
           Keys are the free-parameter or any other specific parameter, and values are four values as [initial-value, min-value, max-value, step-size]

        Keywords
        --------
        params: list [(str,num)], optional
            List of parameters defined as pairs ('parameter name', value). See 'Partial controllers' in the docs/MuMoTuserManual.ipynb. Rates defaults to mumot.MuMoTdefault._initialRateValue. System size defaults to mumot.MuMoTdefault._systemSize. 
        initialState : dictionary {str:float}, optional
           Initial proportions of the reactants (type: dictionary with reactants as keys and floats in range [0,1] as values).
           See the bookmark of and example. Defaults to a dictionary with the (alphabetically) first reactant to 1 and the rest to 0.
        maxTime : float, optional
           Simulation time. Must be strictly positive. Defaults to mumot.MuMoTdefault._maxTime.
        randomSeed : int, optional
           Random seed. Must be strictly positive in range [0, mumot.MAX_RANDOM_SEED]). Defaults to a random number.
        plotProportions : bool, optional
           Flag to plot proportions or full populations. Defaults to False.
        realtimePlot : bool, optional
           Flag to plot results in realtime (True = the plot is updated each timestep of the simulation; False = the plot is updated once at the end of the simulation). Defaults to False.
        visualisationType : str, optional
            Type of visualisation (``'evo'``,``'graph'``,``'final'`` or ``'barplot'``). See docs/MuMoTuserManual.ipynb for more details. Defaults to 'evo'.
        final_x : object, optional
           Which reactant is shown on x-axis when visualisation type is 'final'. Defaults to the alphabetically first reactant.
        final_y : object, optional
           Which reactant is shown on y-axis when visualisation type is 'final'. Defaults to the alphabetically second reactant.
        runs : int, optional
           Number of simulation runs to be executed. Must be strictly positive. Defaults to 1.
        aggregateResults : bool, optional
           Flag to aggregate or not the results from several runs. Defaults to True.
        netType : str, optional
           Type of network (``'full'``, ``'erdos-renyi'``, ``'barabasi-albert'`` or ``'dynamic'``. See docs/MuMoTuserManual.ipynb for more details. Defaults to 'full'.
        netParam : float, optional
           Property of the network ralated to connectivity. The precise meaning and range of this parameter varies depending on the netType. See docs/MuMoTuserManual.ipynb for more details and defaults.
        motionCorrelatedness : float, optional
           (active only for netType='dynamic') level of inertia in the random walk, with 0 the reactants do a completely uncorrelated random walk and with 1 they move on straight trajectories. Must be in range [0, 1]. Defaults to 0.5.
        particleSpeed : float, optional
           (active only for netType='dynamic') speed of the moving particle, i.e. displacement in one timestep. Must be in range [0,1]. Defaults to 0.01.
        timestepSize : float, optional
           Length of one timestep, the maximum size is automatically determined by the rates. Must be strictly positive. Defaults to the maximum value.
        showTrace : bool, optional
           (active only for netType='dynamic') flag to plot the trajectory of each reactant. Defaults to False.
        showInteractions : bool, optional
           (active only for netType='dynamic') flag to plot the interaction range between particles. Defaults to False.
        silent : bool, optional
            Switch on/off widgets and plot. Important for use with multicontrollers. Defaults to False.
            
        Returns
        -------
        :class:`MuMoTmultiagentController`
            A MuMoT controller object
        """
        if initWidgets is None:
            initWidgets = dict()

        paramValuesDict = self._create_free_param_dictionary_for_controller(inputParams=kwargs.get('params', []), initWidgets=initWidgets, showSystemSize=True, showPlotLimits=False)
        
        MAParams = {} 
        # read input parameters
        MAParams['initialState'] = _format_advanced_option(optionName='initialState', inputValue=kwargs.get('initialState'), initValues=initWidgets.get('initialState'), extraParam=self._getAllReactants())
        MAParams['substitutedReactant'] = [ [react for react in self._getAllReactants()[0] if react not in self._reactants][0] if self._systemSize is not None else None, True]
        MAParams['maxTime'] = _format_advanced_option(optionName='maxTime', inputValue=kwargs.get('maxTime'), initValues=initWidgets.get('maxTime'))
        MAParams['randomSeed'] = _format_advanced_option(optionName='randomSeed', inputValue=kwargs.get('randomSeed'), initValues=initWidgets.get('randomSeed'))
        MAParams['motionCorrelatedness'] = _format_advanced_option(optionName='motionCorrelatedness', inputValue=kwargs.get('motionCorrelatedness'), initValues=initWidgets.get('motionCorrelatedness'))
        MAParams['particleSpeed'] = _format_advanced_option(optionName='particleSpeed', inputValue=kwargs.get('particleSpeed'), initValues=initWidgets.get('particleSpeed'))
        MAParams['timestepSize'] = _format_advanced_option(optionName='timestepSize', inputValue=kwargs.get('timestepSize'), initValues=initWidgets.get('timestepSize'))
        MAParams['netType'] = _format_advanced_option(optionName='netType', inputValue=kwargs.get('netType'), initValues=initWidgets.get('netType'))
        systemSize = paramValuesDict["systemSize"][0]
        MAParams['netParam'] = _format_advanced_option(optionName='netParam', inputValue=kwargs.get('netParam'), initValues=initWidgets.get('netParam'), extraParam=MAParams['netType'], extraParam2=systemSize)
        MAParams['plotProportions'] = _format_advanced_option(optionName='plotProportions', inputValue=kwargs.get('plotProportions'), initValues=initWidgets.get('plotProportions'))
        MAParams['realtimePlot'] = _format_advanced_option(optionName='realtimePlot', inputValue=kwargs.get('realtimePlot'), initValues=initWidgets.get('realtimePlot'))
        MAParams['showTrace'] = _format_advanced_option(optionName='showTrace', inputValue=kwargs.get('showTrace'), initValues=initWidgets.get('showTrace', MAParams['netType'] == NetworkType.DYNAMIC))
        MAParams['showInteractions'] = _format_advanced_option(optionName='showInteractions', inputValue=kwargs.get('showInteractions'), initValues=initWidgets.get('showInteractions'))
        MAParams['visualisationType'] = _format_advanced_option(optionName='visualisationType', inputValue=kwargs.get('visualisationType'), initValues=initWidgets.get('visualisationType'), extraParam="multiagent")
        MAParams['final_x'] = _format_advanced_option(optionName='final_x', inputValue=kwargs.get('final_x'), initValues=initWidgets.get('final_x'), extraParam=self._getAllReactants()[0])
        MAParams['final_y'] = _format_advanced_option(optionName='final_y', inputValue=kwargs.get('final_y'), initValues=initWidgets.get('final_y'), extraParam=self._getAllReactants()[0])
        MAParams['runs'] = _format_advanced_option(optionName='runs', inputValue=kwargs.get('runs'), initValues=initWidgets.get('runs'))
        MAParams['aggregateResults'] = _format_advanced_option(optionName='aggregateResults', inputValue=kwargs.get('aggregateResults'), initValues=initWidgets.get('aggregateResults'))
        
        # if the netType is a fixed-param and its value is not 'DYNAMIC', all useless parameter become fixed (and widgets are never displayed)
        if MAParams['netType'][-1]:
            decodedNetType = _decodeNetworkTypeFromString(MAParams['netType'][0])
            if decodedNetType != NetworkType.DYNAMIC:
                MAParams['motionCorrelatedness'][-1] = True
                MAParams['particleSpeed'][-1] = True
                MAParams['showTrace'][-1] = True
                MAParams['showInteractions'][-1] = True
                if decodedNetType == NetworkType.FULLY_CONNECTED:
                    MAParams['netParam'][-1] = True
        
        # construct controller
        viewController = MuMoTmultiagentController(paramValuesDict=paramValuesDict, paramLabelDict=self._ratesLaTeX, continuousReplot=False, advancedOpts=MAParams, showSystemSize=True, **kwargs)
        # Get the default network values assigned from the controller
        modelView = MuMoTmultiagentView(self, viewController, MAParams, **kwargs)
        viewController._setView(modelView)
#         viewController._setReplotFunction(modelView._computeAndPlotSimulation(self._reactants, self._rules))
        viewController._setReplotFunction(modelView._computeAndPlotSimulation, modelView._redrawOnly)
        #viewController._widgetsExtraParams['netType'].value.observe(modelView._update_net_params, 'value') #netType is special

        return viewController

    def SSA(self, initWidgets=None, **kwargs):
        """Construct interactive Stochastic Simulation Algorithm (SSA) plot (simulation run of the Gillespie algorithm to approximate the Master Equation solution).

        Parameters
        ----------
        initWidgets : dict {str,list}, optional
           Keys are the free-parameter or any other specific parameter, and values are four values as [initial-value, min-value, max-value, step-size]

        Keywords
        --------
        params: list [(str,num)], optional
            List of parameters defined as pairs ('parameter name', value). See 'Partial controllers' in the docs/MuMoTuserManual.ipynb. Rates defaults to mumot.MuMoTdefault._initialRateValue. System size defaults to mumot.MuMoTdefault._systemSize. 
        initialState : dictionary {str:float}, optional
           Initial proportions of the reactants (type: dictionary with reactants as keys and floats in range [0,1] as values).
           See the bookmark of and example. Defaults to a dictionary with the (alphabetically) first reactant to 1 and the rest to 0.
        maxTime : float, optional
           Simulation time. Must be strictly positive. Defaults to mumot.MuMoTdefault._maxTime.
        randomSeed : int, optional
           Random seed. Must be strictly positive in range [0, mumot.MAX_RANDOM_SEED]). Defaults to a random number.
        plotProportions : bool, optional
           Flag to plot proportions or full populations. Defaults to False.
        realtimePlot : bool, optional
           Flag to plot results in realtime (True = the plot is updated each timestep of the simulation; False = the plot is updated once at the end of the simulation). Defaults to False.
        visualisationType : str, optional
            Type of visualisation (``'evo'``,``'final'`` or ``'barplot'``). See docs/MuMoTuserManual.ipynb for more details. Defaults to 'evo'.
        final_x : object, optional
           Which reactant is shown on x-axis when visualisation type is 'final'. Defaults to the alphabetically first reactant.
        final_y : object, optional
           Which reactant is shown on y-axis when visualisation type is 'final'. Defaults to the alphabetically second reactant.
        runs : int, optional
           Number of simulation runs to be executed. Must be strictly positive. Defaults to 1.
        aggregateResults : bool, optional
           Flag to aggregate or not the results from several runs. Defaults to True.
        silent : bool, optional
            Switch on/off widgets and plot. Important for use with multicontrollers. Defaults to False.
            
        Returns
        -------
        :class:`MuMoTstochasticSimulationController`
            A MuMoT controller object
        """
        if initWidgets is None:
            initWidgets = {}

        paramValuesDict = self._create_free_param_dictionary_for_controller(
            inputParams=kwargs.get('params', []),
            initWidgets=initWidgets,
            showSystemSize=True, 
            showPlotLimits=False)

        ssaParams = {}
        # read input parameters
        ssaParams['initialState'] = _format_advanced_option(optionName='initialState', inputValue=kwargs.get('initialState'), initValues=initWidgets.get('initialState'), extraParam=self._getAllReactants())
        ssaParams['substitutedReactant'] = [ [react for react in self._getAllReactants()[0] if react not in self._reactants][0] if self._systemSize is not None else None, True]
        ssaParams['maxTime'] = _format_advanced_option(optionName='maxTime', inputValue=kwargs.get('maxTime'), initValues=initWidgets.get('maxTime'))
        ssaParams['randomSeed'] = _format_advanced_option(optionName='randomSeed', inputValue=kwargs.get('randomSeed'), initValues=initWidgets.get('randomSeed'))
        ssaParams['plotProportions'] = _format_advanced_option(optionName='plotProportions', inputValue=kwargs.get('plotProportions'), initValues=initWidgets.get('plotProportions'))
        ssaParams['realtimePlot'] = _format_advanced_option(optionName='realtimePlot', inputValue=kwargs.get('realtimePlot'), initValues=initWidgets.get('realtimePlot'))
        ssaParams['visualisationType'] = _format_advanced_option(optionName='visualisationType', inputValue=kwargs.get('visualisationType'), initValues=initWidgets.get('visualisationType'), extraParam="SSA")
        ssaParams['final_x'] = _format_advanced_option(optionName='final_x', inputValue=kwargs.get('final_x'), initValues=initWidgets.get('final_x'), extraParam=self._getAllReactants()[0])
        ssaParams['final_y'] = _format_advanced_option(optionName='final_y', inputValue=kwargs.get('final_y'), initValues=initWidgets.get('final_y'), extraParam=self._getAllReactants()[0])
        ssaParams['runs'] = _format_advanced_option(optionName='runs', inputValue=kwargs.get('runs'), initValues=initWidgets.get('runs'))
        ssaParams['aggregateResults'] = _format_advanced_option(optionName='aggregateResults', inputValue=kwargs.get('aggregateResults'), initValues=initWidgets.get('aggregateResults'))
        
        # construct controller
        viewController = MuMoTstochasticSimulationController(paramValuesDict=paramValuesDict, paramLabelDict=self._ratesLaTeX, continuousReplot=False, advancedOpts=ssaParams, showSystemSize=True, **kwargs)

        modelView = MuMoTSSAView(self, viewController, ssaParams, **kwargs)
        viewController._setView(modelView)
        
        viewController._setReplotFunction(modelView._computeAndPlotSimulation, modelView._redrawOnly)
        
        return viewController


    ## get the pair of set (reactants, constantReactants). This method is necessary to have all reactants (to set the system size) also after a substitution has occurred
    def _getAllReactants(self):
        allReactants = set()
        allConstantReactants = set()
        for reaction in self._stoichiometry.values():
            for reactant, info in reaction.items():
                if (not reactant == 'rate') and (reactant not in allReactants) and (reactant not in allConstantReactants):
                    if info == 'const':
                        allConstantReactants.add(reactant)
                    else:
                        allReactants.add(reactant)
        return (allReactants, allConstantReactants)

    def _get_solutions(self):
        if self._solutions is None:
            self._solutions = solve(iter(self._equations.values()), self._reactants, force=False, positive=False, set=False)
        return self._solutions


    def _create_free_param_dictionary_for_controller(self, inputParams, initWidgets=None, showSystemSize=False, showPlotLimits=False):
        initWidgetsSympy = {process_sympy(paramName): paramValue for paramName, paramValue in initWidgets.items()} if initWidgets is not None else {}

        paramValuesDict = {}
        for freeParam in self._rates.union(self._constantReactants):
            paramValuesDict[str(freeParam)] = _parse_input_keyword_for_numeric_widgets(inputValue=_get_item_from_params_list(inputParams, str(freeParam)),
                                    defaultValueRangeStep=[MuMoTdefault._initialRateValue, MuMoTdefault._rateLimits[0], MuMoTdefault._rateLimits[1], MuMoTdefault._rateStep], 
                                    initValueRangeStep=initWidgetsSympy.get(freeParam), 
                                    validRange=(-float("inf"), float("inf")))
            
        if showSystemSize:
            paramValuesDict['systemSize'] = _parse_input_keyword_for_numeric_widgets(inputValue=_get_item_from_params_list(inputParams, 'systemSize'),
                                    defaultValueRangeStep=[MuMoTdefault._systemSize, MuMoTdefault._systemSizeLimits[0], MuMoTdefault._systemSizeLimits[1], MuMoTdefault._systemSizeStep], 
                                    initValueRangeStep=initWidgets.get('systemSize'), 
                                    validRange=(1, float("inf"))) 
        if showPlotLimits:
            paramValuesDict['plotLimits'] = _parse_input_keyword_for_numeric_widgets(inputValue=_get_item_from_params_list(inputParams, 'plotLimits'),
                                    defaultValueRangeStep=[MuMoTdefault._plotLimits, MuMoTdefault._plotLimitsLimits[0], MuMoTdefault._plotLimitsLimits[1], MuMoTdefault._plotLimitsStep], 
                                    initValueRangeStep=initWidgets.get('plotLimits'), 
                                    validRange=(-float("inf"), float("inf"))) 
        
        return paramValuesDict


    def _getSingleAgentRules(self):
        """derive the single-agent rules (which are used in the multiagent simulation) from the reaction rules"""
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
#             for idx, reactant in enumerate(rule.lhsReactants):
#                 if reactant in allConstantReactants: #self._constantReactants:
#                     if not (rule.rhsReactants[idx] == reactant or rule.rhsReactants[idx] == EMPTYSET_SYMBOL):
#                         errorMsg = 'In rule ' + str(rule.lhsReactants) + ' -> '  + str(rule.rhsReactants) + ' constant reactant are not properly used. ' \
#                                     'Constant reactants must either match the same constant reactant or the EMPTYSET on the right-handside. \n' \
#                                     'NOTE THAT ORDER MATTERS: MuMoT assumes that first reactant on left-handside becomes first reactant on right-handside and so on for sencond and third...'
#                         print(errorMsg)
#                         raise MuMoTValueError(errorMsg)
#                 elif rule.rhsReactants[idx] in allConstantReactants:
#                     errorMsg = 'In rule ' + str(rule.lhsReactants) + ' -> '  + str(rule.rhsReactants) + ' constant reactant are not properly used.' \
#                                     'Constant reactants appears on the right-handside and not on the left-handside. \n' \
#                                     'NOTE THAT ORDER MATTERS: MuMoT assumes that first reactant on left-handside becomes first reactant on right-handside and so on for sencond and third...'
#                     print(errorMsg)
#                     raise MuMoTValueError(errorMsg)
#                 
#                 if reactant == EMPTYSET_SYMBOL:
#                     targetReact.append(rule.rhsReactants[idx])
            
            for reactant in rule.rhsReactants:
                if reactant in allConstantReactants:
                    warningMsg = 'WARNING! Constant reactants appearing on the right-handside are ignored. Every constant reactant on the left-handside (implicitly) corresponds to the same constant reactant on the right-handside.\n'\
                                'E.g., in rule ' + str(rule.lhsReactants) + ' -> ' + str(rule.rhsReactants) + ' constant reactants should not appear on the right-handside.'
                    print(warningMsg)
                    break  # print maximum one warning
            
            # add to the target of the first non-empty item the new born coming from empty-set or constant reactants 
            for idx, reactant in enumerate(rule.lhsReactants):
                if reactant == EMPTYSET_SYMBOL or reactant in allConstantReactants:
                    # constant reactants on the right-handside are ignored
                    if rule.rhsReactants[idx] not in allConstantReactants:
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
                # extend the other-reactants list
                otherReact.append(react2)
                # create list of targets for the other reactants (if the reactant is constant the target is itself)
                if react2 in allConstantReactants:
                    otherTargets.append(react2)
                else:
                    otherTargets.append(rule.rhsReactants[idx2])
            
            # the target reactant for the first non-empty reactant is the reactant with the same index (on the rhs) plus all the reactants coming from empty-sets (filled through initial loop)
            if reactant in allConstantReactants:
                targetReact.append(reactant)
            elif not reactant == EMPTYSET_SYMBOL:  # if empty it's not added because it has been already added in the initial loop
                targetReact.append(rule.rhsReactants[idx])
            
            # create a new entry
            self._agentProbabilities[reactant].append([otherReact, rule.rate, targetReact, otherTargets])
        #print( self._agentProbabilities )

    def _check_state_variables(self, stateVariable1, stateVariable2, stateVariable3=None):
        if process_sympy(stateVariable1) in self._reactants and process_sympy(stateVariable2) in self._reactants and (stateVariable3 is None or process_sympy(stateVariable3) in self._reactants):
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
#         if self._systemSize is None:
#             assert false ## @todo is this necessary?
        if self._funcs is None:
            argList = []
            for reactant in self._reactants:
                argList.append(reactant)
            for reactant in self._constantReactants:
                argList.append(reactant)
            for rate in self._rates:
                argList.append(rate)
            if self._systemSize is not None:
                argList.append(self._systemSize)
            self._args = tuple(argList)
            self._funcs = {}
            for equation in self._equations:
                f = lambdify(self._args, self._equations[equation], "math")
                self._funcs[equation] = f
            
        return self._funcs
    
    ## get tuple to evalute functions returned by _getFuncs with, for 2d field-based plots
    def _getArgTuple2d(self, argDict, stateVariable1, stateVariable2, X, Y):
        argList = []
        for arg in self._args:
            if arg == stateVariable1:
                argList.append(X)
            elif arg == stateVariable2:
                argList.append(Y)
            elif arg == self._systemSize:
                argList.append(1)  # @todo: system size set to 1
            else:
                try:
                    argList.append(argDict[arg])
                except KeyError:
                    raise MuMoTValueError('Unexpected reactant \'' + str(arg) + '\': system size > 2?')
        return tuple(argList)

    ## get tuple to evalute functions returned by _getFuncs with, for 2d field-based plots
    def _getArgTuple3d(self, argDict, stateVariable1, stateVariable2, stateVariable3, X, Y, Z):
        argList = []
        for arg in self._args:
            if arg == stateVariable1:
                argList.append(X)
            elif arg == stateVariable2:
                argList.append(Y)
            elif arg == stateVariable3:
                argList.append(Z)                
            elif arg == self._systemSize:
                argList.append(1)  # @todo: system size set to 1
            else:
                try:
                    argList.append(argDict[arg])
                except KeyError:
                    raise MuMoTValueError('Unexpected reactant: system size > 3?')
            
        return tuple(argList)

    ## get tuple to evalute functions returned by _getFuncs with
    def _getArgTuple(self, argDict, reactants, reactantValues):
        assert False  # need to work this out
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
        tmpfile = tempfile.NamedTemporaryFile(dir=self._tmpdir.name, suffix='.' + self._renderImageFormat, delete=False)
        self._tmpfiles.append(tmpfile)
        preview(source, euler=False, output=self._renderImageFormat, viewer='file', filename=tmpfile.name)

        return tmpfile.name[tmpfile.name.find(self._tmpdirpath):]  

    

    def __init__(self):
        self._rules = []
        self._reactants = set()
        self._systemSize = None
        self._constantSystemSize = True
        self._reactantsLaTeX = None
        self._rates = set()
        self._ratesLaTeX = None
        self._equations = {}
        self._stoichiometry = {}
        self._pyDSmodel = None
        self._dot = None
        if not os.path.isdir(self._tmpdirpath):
            os.mkdir(self._tmpdirpath)
            os.system('chmod' + self._tmpdirpath + 'u+rwx')
            os.system('chmod' + self._tmpdirpath + 'g-rwx')
            os.system('chmod' + self._tmpdirpath + 'o+rwx')
        self._tmpdir = tempfile.TemporaryDirectory(dir=self._tmpdirpath)
        self._tmpfiles = []
        
    def __del__(self):
        ## @todo: check when this is invoked
        for tmpfile in self._tmpfiles:
            del tmpfile
        del self._tmpdir


class _Rule:
    """Single reaction rule."""

    lhsReactants = []
    rhsReactants = []
    rate = ""
    def __init__(self):
        self.lhsReactants = []
        self.rhsReactants = []
        self.rate = ""


class MuMoTcontroller:
    """Controller for a model view."""

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
    ## redraw function widgets have been assigned (for use by MuMoTmultiController)
    _redrawFunction = None
    ## widget for simple error messages to be displayed to user during interaction
    _errorMessage = None
    ## plot limits slider widget
    _plotLimitsWidget = None  # @todo: is it correct that this is a variable of the general MuMoTcontroller?? it might be simply added in the _widgetsPlotOnly dictionary
    ## system size slider widget
    _systemSizeWidget = None    
    ## bookmark button widget
    _bookmarkWidget = None
    ## advanced tab widget
    _advancedTabWidget = None
    ## download button widget
    _downloadWidget = None
    ## download link widget
    _downloadWidgetLink = None

    def __init__(self, paramValuesDict, paramLabelDict=None, continuousReplot=False, showPlotLimits=False, showSystemSize=False, advancedOpts=None, **kwargs):
        self._silent = kwargs.get('silent', False)
        self._paramLabelDict = paramLabelDict if paramLabelDict is not None else {}
        self._widgetsFreeParams = {}
        self._widgetsExtraParams = {}
        self._widgetsPlotOnly = {}
        self._extraWidgetsOrder = []
        
        for paramName in sorted(paramValuesDict.keys()):
            if paramName == 'plotLimits' or paramName == 'systemSize': continue
            if not paramValuesDict[paramName][-1]:
                paramValue = paramValuesDict[paramName]
                widget = widgets.FloatSlider(value=paramValue[0], min=paramValue[1], 
                                             max=paramValue[2], step=paramValue[3],
                                             readout_format='.' + str(_count_sig_decimals(str(paramValue[3]))) + 'f',
                                             description=r'\(' + _doubleUnderscorify(_greekPrependify(self._paramLabelDict.get(paramName, paramName))) + r'\)', 
                                             style={'description_width': 'initial'},
                                             continuous_update=continuousReplot)
                self._widgetsFreeParams[paramName] = widget
                if not self._silent:
                    display(widget)
        if showPlotLimits:
            if not paramValuesDict['plotLimits'][-1]:
                paramValue = paramValuesDict['plotLimits']
                self._plotLimitsWidget = widgets.FloatSlider(value=paramValue[0], min=paramValue[1], 
                                             max=paramValue[2], step=paramValue[3],
                                             readout_format='.' + str(_count_sig_decimals(str(paramValue[3]))) + 'f',
                                             description='Plot limits', 
                                             style={'description_width': 'initial'},
                                             continuous_update=continuousReplot)
                #@todo: it would be better to remove self._plotLimitsWidget and use the self._widgetsExtraParams['plotLimits'] = widget
                if not self._silent:
                    display(self._plotLimitsWidget)
                
        if showSystemSize:
            if not paramValuesDict['systemSize'][-1]:
                paramValue = paramValuesDict['systemSize']
                self._systemSizeWidget = widgets.FloatSlider(value=paramValue[0], min=paramValue[1], 
                                             max=paramValue[2], step=paramValue[3],
                                             readout_format='.' + str(_count_sig_decimals(str(paramValue[3]))) + 'f',
                                             description='System size', 
                                             style={'description_width': 'initial'},
                                             continuous_update=continuousReplot)
                if not self._silent:
                    display(self._systemSizeWidget)
        
        # create advanced widgets (that will be added into the 'Advanced options' tab)
        initialState = self._createAdvancedWidgets(advancedOpts, continuousReplot)
        self._orderAdvancedWidgets(initialState)
        # add widgets to the Advanced options tab
        if not self._silent:
            self._displayAdvancedOptionsTab()
        
                        
        self._bookmarkWidget = widgets.Button(description='', disabled=False, button_style='', tooltip='Paste bookmark to log', icon='fa-bookmark')
        self._bookmarkWidget.on_click(self._print_standalone_view_cmd)
        bookmark = kwargs.get('bookmark', True)
        
        self._downloadWidget = widgets.Button(description='', disabled=False, button_style='', tooltip='Download results', icon='fa-save')
        self._downloadWidgetLink =  HTML(self._create_download_link("","",""), visible=False)
        self._downloadWidgetLink.layout.visibility = 'hidden'
        self._downloadWidget.on_click(self._download_link_unsupported)

        if not self._silent and bookmark:
            #display(self._bookmarkWidget)
        
            box_layout = widgets.Layout(display='flex',
                                        flex_flow='row',
                                        align_items='stretch',
                                        width='70%')
            threeButtons = widgets.Box(children=[self._bookmarkWidget, self._downloadWidget, self._downloadWidgetLink], layout=box_layout)
            display(threeButtons)

        widget = widgets.HTML()
        widget.value = ''
        self._errorMessage = widget
        if not self._silent and bookmark:
            display(self._errorMessage)

    def _print_standalone_view_cmd(self, _includeParams):
        self._errorMessage.value = "Pasted bookmark to log - view with showLogs(tail = True)"
        self._view._print_standalone_view_cmd(True)

    ## set the functions that must be triggered when the widgets are changed.
    ## @param[in]    recomputeFunction    The function to be called when recomputing is necessary 
    ## @param[in]    redrawFunction    The function to be called when only redrawing (relying on previous computation) is sufficient 
    def _setReplotFunction(self, recomputeFunction, redrawFunction=None):
        """set the functions that must be triggered when the widgets are changed.
        :param recomputeFunction
            The function to be called when recomputing is necessary
        :param redrawFunction
            The function to be called when only redrawing (relying on previous computation) is sufficient""" 
        self._replotFunction = recomputeFunction
        self._redrawFunction = redrawFunction
        for widget in self._widgetsFreeParams.values():
            #widget.on_trait_change(recomputeFunction, 'value')
            widget.observe(recomputeFunction, 'value')
        for widget in self._widgetsExtraParams.values():
            widget.observe(recomputeFunction, 'value')
        if self._plotLimitsWidget is not None:
            self._plotLimitsWidget.observe(recomputeFunction, 'value')
        if self._systemSizeWidget is not None:
            self._systemSizeWidget.observe(recomputeFunction, 'value')
        if redrawFunction is not None:
            for widget in self._widgetsPlotOnly.values():
                widget.observe(redrawFunction, 'value')

    def _createAdvancedWidgets(self, _advancedOpts, _continuousReplot=False):
        """interface method to add advanced options (if needed)"""
        return None
    
    def _orderAdvancedWidgets(self, initialState):
        """interface method to sort the advanced options, in the self._extraWidgetsOrder list"""
        pass

    ## create and display the "Advanced options" tab (if not empty)
    def _displayAdvancedOptionsTab(self):
        """create and display the "Advanced options" tab (if not empty)"""
        advancedWidgets = []
        atLeastOneAdvancedWidget = False
        for widgetName in self._extraWidgetsOrder:
            if self._widgetsExtraParams.get(widgetName):
                advancedWidgets.append(self._widgetsExtraParams[widgetName])
                if not self._widgetsExtraParams[widgetName].layout.display == 'none':
                    atLeastOneAdvancedWidget = True
            elif self._widgetsPlotOnly.get(widgetName):
                advancedWidgets.append(self._widgetsPlotOnly[widgetName])
                if not self._widgetsPlotOnly[widgetName].layout.display == 'none':
                    atLeastOneAdvancedWidget = True
            #else:
                #print("WARNING! In the _extraWidgetsOrder is listed the widget " + widgetName + " which is although not found in _widgetsExtraParams or _widgetsPlotOnly")
        if advancedWidgets:  # if not empty
            advancedPage = widgets.Box(children=advancedWidgets)
            advancedPage.layout.flex_flow = 'column'
            self._advancedTabWidget = widgets.Accordion(children=[advancedPage])  # , selected_index=-1)
            self._advancedTabWidget.set_title(0, 'Advanced options')
            self._advancedTabWidget.selected_index = None
            if atLeastOneAdvancedWidget: display(self._advancedTabWidget)


    def _setView(self, view):
        self._view = view

    def showLogs(self, tail=False):
        """Show logs from associated view.

        Parameters
        ----------
        tail : bool, optional
           Flag to show only tail entries from logs. Defaults to False.
        """
        self._view.showLogs(tail)
        
    def _updateInitialStateWidgets(self, _=None):
        (allReactants, _) = self._view._mumotModel._getAllReactants()
        if len(allReactants) == 1: return
        sumNonConstReactants = 0
        for state in allReactants:
            sumNonConstReactants += self._widgetsExtraParams['init'+str(state)].value
        substitutedReactant = [react for react in allReactants if react not in self._view._mumotModel._reactants][0] if self._view._mumotModel._systemSize is not None else None
        disabledValue = 1
        for i, state in enumerate(sorted(allReactants, key=str)):
            if (substitutedReactant is None and i == 0) or (substitutedReactant is not None and state == substitutedReactant):
                disabledValue = 1-(sumNonConstReactants-self._widgetsExtraParams['init'+str(state)].value)
                break
            
        for i, state in enumerate(sorted(allReactants, key=str)):
            # oder of assignment is important (first, update the min and max, later, the value)
            toLinkPlotFunction = False
            # the self._view._controller pointer is necessary to work properly with multiControllers
            if self._view._controller._replotFunction is not None:
#             if self._replotFunction is not None:
                try:
                    self._widgetsExtraParams['init'+str(state)].unobserve(self._view._controller._replotFunction, 'value')
#                     self._widgetsExtraParams['init'+str(state)].unobserve(self._replotFunction, 'value')
                    toLinkPlotFunction = True
                except ValueError:
                    pass
             
            disabledState = (substitutedReactant is None and i == 0) or (substitutedReactant is not None and state == substitutedReactant)
            if disabledState:
                #print(str(state) + ": sum is " + str(sumNonConstReactants) + " - val " + str(disabledValue))
                self._widgetsExtraParams['init'+str(state)].value = disabledValue   
            else:
                #maxVal = 1-disabledValue if 1-disabledValue > self._widgetsExtraParams['init'+str(state)].min else self._widgetsExtraParams['init'+str(state)].min
                maxVal = disabledValue + self._widgetsExtraParams['init'+str(state)].value
                self._widgetsExtraParams['init'+str(state)].max = maxVal
            
            if toLinkPlotFunction:
                self._widgetsExtraParams['init'+str(state)].observe(self._view._controller._replotFunction, 'value')
    
    def _updateFinalViewWidgets(self, change=None):
        if change['new'] != 'final':
            if self._widgetsPlotOnly.get('final_x'): self._widgetsPlotOnly['final_x'].layout.display = 'none'
            if self._widgetsPlotOnly.get('final_y'): self._widgetsPlotOnly['final_y'].layout.display = 'none'
        else:
            if self._widgetsPlotOnly.get('final_x'): self._widgetsPlotOnly['final_x'].layout.display = 'flex'
            if self._widgetsPlotOnly.get('final_y'): self._widgetsPlotOnly['final_y'].layout.display = 'flex'

    def _setErrorWidget(self, errorWidget):
        self._errorMessage = errorWidget

            
    def _downloadFileWithJavascript(self, data_to_download):
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
    
    def _create_download_link(self, text, title="Download file", filename="file.txt"):
        """Create a download link"""  
        b64 = base64.b64encode(text.encode())
        payload = b64.decode()
        html = '<a download="{filename}" href="data:text/text;base64,{payload}" target="_blank">{title}</a>'
        html = html.format(payload=payload,title=title,filename=filename)
        return html

    def _reveal_download_link(self, _includeParams):
        """Make download link visible"""
        self._downloadWidgetLink.layout.visibility='visible'

    def _download_link_unsupported(self, _includeParams):
        """Report that results download is unsupported"""
        self._view._showErrorMessage("Results download for this view is currently unsupported")

class MuMoTbifurcationController(MuMoTcontroller):
    """Controller to enable Advanced options widgets for bifurcation view."""
    
    def _createAdvancedWidgets(self, BfcParams, continuousReplot=False):
        initialState = BfcParams['initialState'][0]
        if not BfcParams['initialState'][-1]:
            #for state,pop in initialState.items():
            for i, state in enumerate(sorted(initialState.keys(), key=str)):
                pop = initialState[state]
                widget = widgets.FloatSlider(value=pop[0],
                                             min=pop[1], 
                                             max=pop[2],
                                             step=pop[3],
                                             readout_format='.' + str(_count_sig_decimals(str(pop[3]))) + 'f',
                                             #r'$' + '\Phi_{' + _doubleUnderscorify(_greekPrependify(str(stateVarExpr1))) +'}$'
                                             description=r'$' + '\Phi_{' + _doubleUnderscorify(_greekPrependify(self._paramLabelDict.get(state, str(state)))) + '}$' + " at t=0: ",
                                             #description = r'\(' + _doubleUnderscorify(_greekPrependify('Phi_'+self._paramLabelDict.get(state,str(state)))) + r'\)' + " at t=0: ",
                                             style={'description_width': 'initial'},
                                             continuous_update=continuousReplot)
                
                if BfcParams['conserved'][0] == True:
                    # disable one population widget (if there are more than 1) to constrain the initial sum of population sizes to 1
                    if len(initialState) > 1:
                        # if there is not a 'substituted' reactant, the last population widget
                        if BfcParams['substitutedReactant'][0] is None and i == 0:
                            widget.disabled = True
                        elif BfcParams['substitutedReactant'][0] is not None and state == BfcParams['substitutedReactant'][0]: # if there is a 'substituted' reactant, this is the chosen one as the disabled pop 
                            widget.disabled = True
                        else:
                            widget.observe(self._updateInitialStateWidgets, 'value')
                        
                self._widgetsExtraParams['init'+str(state)] = widget
            
        # init bifurcation paramter value slider
        if not BfcParams['initBifParam'][-1]:
            initBifParam = BfcParams['initBifParam']
            widget = widgets.FloatSlider(value=initBifParam[0], min=initBifParam[1], 
                                             max=initBifParam[2], step=initBifParam[3],
                                             readout_format='.' + str(_count_sig_decimals(str(initBifParam[3]))) + 'f',
                                             description='Initial ' + r'\(' + _doubleUnderscorify(_greekPrependify(str(BfcParams['bifurcationParameter'][0]))) + r'\) : ',
                                             style={'description_width': 'initial:'},
                                             #layout=widgets.Layout(width='50%'),
                                             disabled=False,
                                             continuous_update=continuousReplot) 
            self._widgetsExtraParams['initBifParam'] = widget
        
        return initialState
    
    def _orderAdvancedWidgets(self, initialState):
        # define the widget order
        self._extraWidgetsOrder.append('initBifParam')
        for state in sorted(initialState.keys(), key=str):
            self._extraWidgetsOrder.append('init'+str(state))


class MuMoTtimeEvolutionController(MuMoTcontroller):
    """Controller class to enable Advanced options widgets for simulation of ODEs and time evolution of noise correlations."""
    
    def _createAdvancedWidgets(self, tEParams, continuousReplot=False):
        initialState = tEParams['initialState'][0]
        if not tEParams['initialState'][-1]:
            #for state,pop in initialState.items():
            for i, state in enumerate(sorted(initialState.keys(), key=str)):
                pop = initialState[state]
                widget = widgets.FloatSlider(value=pop[0],
                                             min=pop[1], 
                                             max=pop[2],
                                             step=pop[3],
                                             readout_format='.' + str(_count_sig_decimals(str(pop[3]))) + 'f',
                                             #description = "Reactant " + r'\(' + _doubleUnderscorify(_greekPrependify(self._paramLabelDict.get(state,str(state)))) + r'\)' + " at t=0: ",
                                             description=r'$' + '\Phi_{' + _doubleUnderscorify(_greekPrependify(self._paramLabelDict.get(state, str(state)))) + '}$' + " at t=0: ",
                                             #description = r'\(' + latex(Symbol('Phi_'+str(state))) + r'\)' + " at t=0: ",
                                             style={'description_width': 'initial'},
                                             continuous_update=continuousReplot)
                
                if tEParams['conserved'][0] == True:
                    # disable one population widget (if there are more than 1) to constrain the initial sum of population sizes to 1
                    if len(initialState) > 1:
                        # if there is not a 'substituted' reactant, the last population widget
                        if tEParams['substitutedReactant'][0] is None and i == 0:
                            widget.disabled = True
                        elif tEParams['substitutedReactant'][0] is not None and state == tEParams['substitutedReactant'][0]: # if there is a 'substituted' reactant, this is the chosen one as the disabled pop 
                            widget.disabled = True
                        else:
                            widget.observe(self._updateInitialStateWidgets, 'value')
                        
                self._widgetsExtraParams['init'+str(state)] = widget
            
        # Max time slider
        if not tEParams['maxTime'][-1]:
            maxTime = tEParams['maxTime']
            widget = widgets.FloatSlider(value=maxTime[0], min=maxTime[1], 
                                             max=maxTime[2], step=maxTime[3],
                                             readout_format='.' + str(_count_sig_decimals(str(maxTime[3]))) + 'f',
                                             description='Simulation time:',
                                             style={'description_width': 'initial'},
                                             #layout=widgets.Layout(width='50%'),
                                             disabled=False,
                                             continuous_update=continuousReplot) 
            self._widgetsExtraParams['maxTime'] = widget
        
        
        ## Checkbox for proportions or full populations plot
        if 'plotProportions' in tEParams:
            if not tEParams['plotProportions'][-1]:
                widget = widgets.Checkbox(
                    value=tEParams['plotProportions'][0],
                    description='Plot population proportions',
                    disabled=False
                )
                self._widgetsPlotOnly['plotProportions'] = widget
        
        return initialState
    
    def _orderAdvancedWidgets(self, initialState):
        # define the widget order
        for state in sorted(initialState.keys(), key=str):
            self._extraWidgetsOrder.append('init'+str(state))
        self._extraWidgetsOrder.append('maxTime')
        if 'plotProportions' in self._widgetsPlotOnly:
            self._extraWidgetsOrder.append('plotProportions')


class MuMoTfieldController(MuMoTcontroller):
    """Controller for field views"""
    
    def _createAdvancedWidgets(self, advancedOpts, continuousReplot=False):
        # Max time slider
        if not advancedOpts['maxTime'][-1]:
            maxTime = advancedOpts['maxTime']
            widget = widgets.FloatSlider(value=maxTime[0], min=maxTime[1], 
                                             max=maxTime[2], step=maxTime[3],
                                             readout_format='.' + str(_count_sig_decimals(str(maxTime[3]))) + 'f',
                                             description='Simulation time:',
                                             style={'description_width': 'initial'},
                                             #layout=widgets.Layout(width='50%'),
                                             disabled=False,
                                             continuous_update=continuousReplot) 
            self._widgetsExtraParams['maxTime'] = widget
        
        # Random seed input field
        if not advancedOpts['randomSeed'][-1]:
            widget = widgets.IntText(
                value=advancedOpts['randomSeed'][0],
                description='Random seed:',
                style={'description_width': 'initial'},
                disabled=False
            )
            self._widgetsExtraParams['randomSeed'] = widget
        
        ## @todo: this block of commented code can be used readily used to fix issue #95 
#         if not advancedOpts['final_x'][-1]:
#             opts = []
#             for reactant in sorted(initialState.keys(), key=str):
#                 #opts.append( ( "Reactant " + r'$'+ latex(Symbol(str(reactant)))+r'$', str(reactant) ) )
#                 opts.append(("Reactant " + r'\(' + _doubleUnderscorify(_greekPrependify(str(reactant))) + r'\)', str(reactant)))
#             dropdown = widgets.Dropdown( 
#                 options=opts,
#                 description='Final distribution (x axis):',
#                 value=advancedOpts['final_x'][0], 
#                 style={'description_width': 'initial'}
#             )
#             self._widgetsPlotOnly['final_x'] = dropdown
#         if not advancedOpts['final_y'][-1]:
#             opts = []
#             for reactant in sorted(initialState.keys(), key=str):
# #                 opts.append( ( r'$'+ _doubleUnderscorify(_greekPrependify(str(reactant))) +'$' , str(reactant) ) )
# #                 print("the reactant is " + str(reactant))
# #                 print("the _greekPrependify(str(reactant) is " + str(_greekPrependify(str(reactant))) )
# #                 print("the _doubleUnderscorify(_greekPrependify(str(reactant))) is " + str(_doubleUnderscorify(_greekPrependify(str(reactant)))) )
#                 opts.append(("Reactant " + r'\(' + _doubleUnderscorify(_greekPrependify(str(reactant))) + r'\)', str(reactant)))
#             dropdown = widgets.Dropdown( 
#                 options=opts,
#                 description='Final distribution (y axis):',
#                 value=advancedOpts['final_y'][0], 
#                 style={'description_width': 'initial'}
#             )
#             self._widgetsPlotOnly['final_y'] = dropdown
            
        
        ## @todo: this block of commented code can be used readily used to fix issue #283
#         ## Checkbox for proportions or full populations plot
#         if not advancedOpts['plotProportions'][-1]:
#             widget = widgets.Checkbox(
#                 value=advancedOpts['plotProportions'][0],
#                 description='Plot population proportions',
#                 disabled=False
#             )
#             self._widgetsPlotOnly['plotProportions'] = widget
        
        # Number of runs slider
        if not advancedOpts['runs'][-1]:
            runs = advancedOpts['runs']
            widget = widgets.IntSlider(value=runs[0], min=runs[1], 
                                             max=runs[2], step=runs[3],
                                             readout_format='.' + str(_count_sig_decimals(str(runs[3]))) + 'f',
                                             description='Number of runs:',
                                             style={'description_width': 'initial'},
                                             #layout=widgets.Layout(width='50%'),
                                             disabled=False,
                                             continuous_update=continuousReplot) 
            self._widgetsExtraParams['runs'] = widget
            
        ## Checkbox for realtime plot update
        if not advancedOpts['aggregateResults'][-1]:
            widget = widgets.Checkbox(
                value=advancedOpts['aggregateResults'][0],
                description='Aggregate results',
                disabled=False
            )
            self._widgetsExtraParams['aggregateResults'] = widget
    
        return None
            
    def _orderAdvancedWidgets(self, _noInitialState):
        # define the widget order
#         self._extraWidgetsOrder.append('final_x')
#         self._extraWidgetsOrder.append('final_y')
#         self._extraWidgetsOrder.append('plotProportions')
        self._extraWidgetsOrder.append('runs')
        self._extraWidgetsOrder.append('maxTime')
        self._extraWidgetsOrder.append('randomSeed')
        self._extraWidgetsOrder.append('aggregateResults')
        
class MuMoTstochasticSimulationController(MuMoTcontroller):
    """Controller for stochastic simulations (base class of MuMoTmultiagentController)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._downloadWidget.on_click(self._download_link_unsupported, remove = True)
        self._downloadWidget.on_click(self._reveal_download_link)


    def _createAdvancedWidgets(self, SSParams, continuousReplot=False):
        initialState = SSParams['initialState'][0]
        if not SSParams['initialState'][-1]:
            #for state,pop in initialState.items():
            for i, state in enumerate(sorted(initialState.keys(), key=str)):
                pop = initialState[state]
                widget = widgets.FloatSlider(value=pop[0],
                                             min=pop[1], 
                                             max=pop[2],
                                             step=pop[3],
                                             readout_format='.' + str(_count_sig_decimals(str(pop[3]))) + 'f',
                                             description=r'$' + _doubleUnderscorify(_greekPrependify(str(Symbol('Phi_{'+str(state)+'}')))) + '$' + " at t=0: ",
                                             #description = r'\(' + latex(Symbol('Phi_'+str(state))) + r'\)' + " at t=0: ",
                                             style={'description_width': 'initial'},
                                             continuous_update=continuousReplot)
                # disable one population widget (if there are more than 1) to constrain the initial sum of population sizes to 1
                if len(initialState) > 1:
                    # if there is not a 'substituted' reactant, the last population widget
                    if SSParams['substitutedReactant'][0] is None and i == 0:
                        widget.disabled = True
                    elif SSParams['substitutedReactant'][0] is not None and state == SSParams['substitutedReactant'][0]: # if there is a 'substituted' reactant, this is the chosen one as the disabled pop 
                        widget.disabled = True
                    else:
                        widget.observe(self._updateInitialStateWidgets, 'value')
                        
                self._widgetsExtraParams['init'+str(state)] = widget
            
        # Max time slider
        if not SSParams['maxTime'][-1]:
            maxTime = SSParams['maxTime']
            widget = widgets.FloatSlider(value=maxTime[0], min=maxTime[1], 
                                             max=maxTime[2], step=maxTime[3],
                                             readout_format='.' + str(_count_sig_decimals(str(maxTime[3]))) + 'f',
                                             description='Simulation time:',
                                             style={'description_width': 'initial'},
                                             #layout=widgets.Layout(width='50%'),
                                             disabled=False,
                                             continuous_update=continuousReplot) 
            self._widgetsExtraParams['maxTime'] = widget
        
        # Random seed input field
        if not SSParams['randomSeed'][-1]:
            widget = widgets.IntText(
                value=SSParams['randomSeed'][0],
                description='Random seed:',
                style={'description_width': 'initial'},
                disabled=False
            )
            self._widgetsExtraParams['randomSeed'] = widget
        
        try:
            ## Toggle buttons for plotting style 
            if not SSParams['visualisationType'][-1]:
                plotToggle = widgets.ToggleButtons(
                    options=[('Temporal evolution', 'evo'), ('Final distribution', 'final'), ('Barplot', 'barplot')],
                    value=SSParams['visualisationType'][0],
                    description='Plot:',
                    disabled=False,
                    button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                    tooltips=['Population change over time', 'Population distribution in each state at final timestep', 'Barplot of states at final timestep'],
                #     icons=['check'] * 3
                )
                plotToggle.observe(self._updateFinalViewWidgets, 'value')
                self._widgetsPlotOnly['visualisationType'] = plotToggle

        except widgets.trait_types.traitlets.TraitError:  # this widget could be redefined in a subclass and the init-value in SSParams['visualisationType'][0] might raise an exception
            pass
        
        if not SSParams['final_x'][-1] and (SSParams['visualisationType'][-1] == False or SSParams['visualisationType'][0] == 'final'):
            opts = []
            for reactant in sorted(initialState.keys(), key=str):
                #opts.append( ( "Reactant " + r'$'+ latex(Symbol(str(reactant)))+r'$', str(reactant) ) )
                opts.append(("Reactant " + r'\(' + _doubleUnderscorify(_greekPrependify(str(reactant))) + r'\)', str(reactant)))
            dropdown = widgets.Dropdown( 
                options=opts,
                description='Final distribution (x axis):',
                value=SSParams['final_x'][0], 
                style={'description_width': 'initial'}
            )
            if SSParams['visualisationType'][0] != 'final':
                dropdown.layout.display = 'none'
            self._widgetsPlotOnly['final_x'] = dropdown
        if not SSParams['final_y'][-1] and (SSParams['visualisationType'][-1] == False or SSParams['visualisationType'][0] == 'final'):
            opts = []
            for reactant in sorted(initialState.keys(), key=str):
#                 opts.append( ( r'$'+ _doubleUnderscorify(_greekPrependify(str(reactant))) +'$' , str(reactant) ) )
#                 print("the reactant is " + str(reactant))
#                 print("the _greekPrependify(str(reactant) is " + str(_greekPrependify(str(reactant))) )
#                 print("the _doubleUnderscorify(_greekPrependify(str(reactant))) is " + str(_doubleUnderscorify(_greekPrependify(str(reactant)))) )
                opts.append(("Reactant " + r'\(' + _doubleUnderscorify(_greekPrependify(str(reactant))) + r'\)', str(reactant)))
            dropdown = widgets.Dropdown( 
                options=opts,
                description='Final distribution (y axis):',
                value=SSParams['final_y'][0], 
                style={'description_width': 'initial'}
            )
            if SSParams['visualisationType'][0] != 'final':
                dropdown.layout.display = 'none'
            self._widgetsPlotOnly['final_y'] = dropdown
            
        
        ## Checkbox for proportions or full populations plot
        if not SSParams['plotProportions'][-1]:
            widget = widgets.Checkbox(
                value=SSParams['plotProportions'][0],
                description='Plot population proportions',
                disabled=False
            )
            self._widgetsPlotOnly['plotProportions'] = widget
        
        ## Checkbox for realtime plot update
        if not SSParams['realtimePlot'][-1]:
            widget = widgets.Checkbox(
                value=SSParams['realtimePlot'][0],
                description='Runtime plot update',
                disabled=False
            )
            self._widgetsExtraParams['realtimePlot'] = widget
        
        # Number of runs slider
        if not SSParams['runs'][-1]:
            runs = SSParams['runs']
            widget = widgets.IntSlider(value=runs[0], min=runs[1], 
                                             max=runs[2], step=runs[3],
                                             readout_format='.' + str(_count_sig_decimals(str(runs[3]))) + 'f',
                                             description='Number of runs:',
                                             style={'description_width': 'initial'},
                                             #layout=widgets.Layout(width='50%'),
                                             disabled=False,
                                             continuous_update=continuousReplot) 
            self._widgetsExtraParams['runs'] = widget
            
        ## Checkbox for realtime plot update
        if not SSParams['aggregateResults'][-1]:
            widget = widgets.Checkbox(
                value=SSParams['aggregateResults'][0],
                description='Aggregate results',
                disabled=False
            )
            self._widgetsPlotOnly['aggregateResults'] = widget
        
        self._addSpecificWidgets(SSParams, continuousReplot)
    
        return initialState
            
    def _addSpecificWidgets(self, SSParams, continuousReplot):
        pass
    
    def _orderAdvancedWidgets(self, initialState):
        # define the widget order
        for state in sorted(initialState.keys(), key=str):
            self._extraWidgetsOrder.append('init'+str(state))
        self._extraWidgetsOrder.append('maxTime')
        self._extraWidgetsOrder.append('randomSeed')
        self._extraWidgetsOrder.append('visualisationType')
        self._extraWidgetsOrder.append('final_x')
        self._extraWidgetsOrder.append('final_y')
        self._extraWidgetsOrder.append('plotProportions')
        self._extraWidgetsOrder.append('realtimePlot')
        self._extraWidgetsOrder.append('runs')
        self._extraWidgetsOrder.append('aggregateResults')
        

class MuMoTmultiagentController(MuMoTstochasticSimulationController):
    """Controller for multiagent views."""

    def _addSpecificWidgets(self, MAParams, continuousReplot=False):
        
        ## Network type dropdown selector
        if not MAParams['netType'][-1]:
            netDropdown = widgets.Dropdown( 
                options=[('Full graph', NetworkType.FULLY_CONNECTED), 
                         ('Erdos-Renyi', NetworkType.ERSOS_RENYI),
                         ('Barabasi-Albert', NetworkType.BARABASI_ALBERT),
                         ## @todo: add network topology generated by random points in space
                         ('Moving particles', NetworkType.DYNAMIC)
                         ],
                description='Network topology:',
                value=_decodeNetworkTypeFromString(MAParams['netType'][0]), 
                style={'description_width': 'initial'},
                disabled=False
            )
            netDropdown.observe(self._update_net_params, 'value')
            self._widgetsExtraParams['netType'] = netDropdown
        
        # Network connectivity slider
        if not MAParams['netParam'][-1]:
            netParam = MAParams['netParam']
            widget = widgets.FloatSlider(value=netParam[0],
                                        min=netParam[1], 
                                        max=netParam[2],
                                        step=netParam[3],
                                readout_format='.' + str(_count_sig_decimals(str(netParam[3]))) + 'f',
                                description='Network connectivity parameter', 
                                style={'description_width': 'initial'},
                                layout=widgets.Layout(width='50%'),
                                continuous_update=continuousReplot,
                                disabled=False
            )
            self._widgetsExtraParams['netParam'] = widget
        
        # Agent speed
        if not MAParams['particleSpeed'][-1]:
            particleSpeed = MAParams['particleSpeed']
            widget = widgets.FloatSlider(value=particleSpeed[0],
                                         min=particleSpeed[1], max=particleSpeed[2], step=particleSpeed[3],
                                readout_format='.' + str(_count_sig_decimals(str(particleSpeed[3]))) + 'f',
                                description='Particle speed', 
                                style={'description_width': 'initial'},
                                layout=widgets.Layout(width='50%'),
                                continuous_update=continuousReplot,
                                disabled=False
            )
            self._widgetsExtraParams['particleSpeed'] = widget
            
        # Random walk correlatedness
        if not MAParams['motionCorrelatedness'][-1]:
            motionCorrelatedness = MAParams['motionCorrelatedness']
            widget = widgets.FloatSlider(value=motionCorrelatedness[0],
                                         min=motionCorrelatedness[1],
                                         max=motionCorrelatedness[2],
                                         step=motionCorrelatedness[3],
                                readout_format='.' + str(_count_sig_decimals(str(motionCorrelatedness[3]))) + 'f',
                                description='Correlatedness of the random walk',
                                layout=widgets.Layout(width='50%'),
                                style={'description_width': 'initial'},
                                continuous_update=continuousReplot,
                                disabled=False
            )
            self._widgetsExtraParams['motionCorrelatedness'] = widget
        
        # Time scaling slider
        if not MAParams['timestepSize'][-1]:
            timestepSize = MAParams['timestepSize']
            widget = widgets.FloatSlider(value=timestepSize[0],
                                        min=timestepSize[1], 
                                        max=timestepSize[2],
                                        step=timestepSize[3],
                                readout_format='.' + str(_count_sig_decimals(str(timestepSize[3]))) + 'f',
                                description='Timestep size', 
                                style={'description_width': 'initial'},
                                layout=widgets.Layout(width='50%'),
                                continuous_update=continuousReplot
            )
            self._widgetsExtraParams['timestepSize'] = widget
        
        ## Toggle buttons for plotting style
        if not MAParams['visualisationType'][-1]:
            plotToggle = widgets.ToggleButtons(
                options=[('Temporal evolution', 'evo'), ('Network', 'graph'), ('Final distribution', 'final'), ('Barplot', 'barplot')],
                value=MAParams['visualisationType'][0],
                description='Plot:',
                disabled=False,
                button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                tooltips=['Population change over time', 'Population distribution in each state at final timestep', 'Barplot of states at final timestep'],
            )
            plotToggle.observe(self._updateFinalViewWidgets, 'value')
            self._widgetsPlotOnly['visualisationType'] = plotToggle
            
        # Particle display checkboxes
        if not MAParams['showTrace'][-1]:
            widget = widgets.Checkbox(
                value=MAParams['showTrace'][0],
                description='Show particle trace',
                disabled=False  # not (self._widgetsExtraParams['netType'].value == NetworkType.DYNAMIC)
            )
            self._widgetsPlotOnly['showTrace'] = widget
        if not MAParams['showInteractions'][-1]:
            widget = widgets.Checkbox(
                value=MAParams['showInteractions'][0],
                description='Show communication links',
                disabled=False  # not (self._widgetsExtraParams['netType'].value == NetworkType.DYNAMIC)
            )
            self._widgetsPlotOnly['showInteractions'] = widget
    
    def _orderAdvancedWidgets(self, initialState): 
        # define the widget order
        for state in sorted(initialState.keys(), key=str):
            self._extraWidgetsOrder.append('init'+str(state))
        self._extraWidgetsOrder.append('maxTime')
        self._extraWidgetsOrder.append('timestepSize')
        self._extraWidgetsOrder.append('netType')
        self._extraWidgetsOrder.append('netParam')
        self._extraWidgetsOrder.append('particleSpeed')
        self._extraWidgetsOrder.append('motionCorrelatedness')
        self._extraWidgetsOrder.append('randomSeed')
        self._extraWidgetsOrder.append('visualisationType')
        self._extraWidgetsOrder.append('final_x')
        self._extraWidgetsOrder.append('final_y')
        self._extraWidgetsOrder.append('showTrace')
        self._extraWidgetsOrder.append('showInteractions')
        self._extraWidgetsOrder.append('plotProportions')
        self._extraWidgetsOrder.append('realtimePlot')
        self._extraWidgetsOrder.append('runs')
        self._extraWidgetsOrder.append('aggregateResults')
        
    def _update_net_params(self, _=None):
        """Update the widgets related to the ``netType`` 
        
        It is linked - through ``observe()`` - before the ``_view`` is created.

        """
        if self._view: self._view._update_net_params(True)
    

class MuMoTview:
    """A view on a model."""

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
    ## parameter values when used without controller
    _fixedParams = None
    ## dictionary of rates and value
    _ratesDict = None
    ## total number of agents in the simulation
    _systemSize = None
    ## silent flag (TRUE = do not try to acquire figure handle from pyplot)
    _silent = None
    ## plot limits (for non-constant system size) @todo: not used?
    _plotLimits = None
    ## command name that generates this view
    _generatingCommand = None
    ## generating keyword arguments
    _generatingKwargs = None
    
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
            (paramNames, paramValues) = _process_params(params)
            self._fixedParams = dict(zip(paramNames, paramValues))
        
        # storing the rates for each rule
        if self._mumotModel:
            freeParamDict = self._get_argDict()
            self._ratesDict = {}
            for rule in self._mumotModel._rules:
                self._ratesDict[str(rule.rate)] = rule.rate.subs(freeParamDict) 
                if self._ratesDict[str(rule.rate)] == float('inf') or self._ratesDict[str(rule.rate)] is sympy.zoo:
                    self._ratesDict[str(rule.rate)] = sys.maxsize
            #print(self._ratesDict)
        self._systemSize = self._getSystemSize()
        
        if not(self._silent):
            _buildFig(self, figure)
    

    def _resetErrorMessage(self):
        if self._controller is not None:
            if not (self._silent):
                self._controller._errorMessage.value = ''

    def _showErrorMessage(self, message):
        if self._controller is not None:
            self._controller._errorMessage.value = self._controller._errorMessage.value + message
        else:
            print(message)


    def _show_computation_start(self):
        if self._controller is not None:
            self._controller._bookmarkWidget.style.button_color = 'pink'
        
               
    def _show_computation_stop(self):
        # ax = plt.gca()
        # ax.set_facecolor('xkcd:white')
        # print("pink off")
        if self._controller is not None:
            self._controller._bookmarkWidget.style.button_color = 'silver'

            
    def _setLog(self, log):
        self._logs = log


    def _log(self, analysis):
        print("Starting", analysis, "with parameters ", end='')
        paramNames = []
        paramValues = []
        if self._controller is not None:
            ## @todo: if the alphabetic order is not good, the view could store the desired order in (paramNames) when the controller is constructed
            for name in sorted(self._controller._widgetsFreeParams.keys()):
                paramNames.append(name)
                paramValues.append(self._controller._widgetsFreeParams[name].value)
            for name in sorted(self._controller._widgetsExtraParams.keys()):
                paramNames.append(name)
                paramValues.append(self._controller._widgetsExtraParams[name].value)
#         if self._paramNames is not None:
#             paramNames += map(str, self._paramNames)
#             paramValues += self._paramValues
            ## @todo: in soloView, this does not show the extra parameters (we should make clearer what the use of showLogs)
        for key, value in self._fixedParams.items():
            paramNames.append(str(key))
            paramValues.append(value)

        for i in zip(paramNames, paramValues):
            print('(' + i[0] + '=' + repr(i[1]) + '), ', end='')
        print("at", datetime.datetime.now())


    def _print_standalone_view_cmd(self, includeParams=True):
        logStr = self._build_bookmark(includeParams)
        if not self._silent and logStr is not None:
            with io.capture_output() as log:
                print(logStr)    
            self._logs.append(log)
        return logStr

    def _set_fixedParams(self, paramDict):
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
                paramInitCheck.append(latex(Symbol('Phi^0_'+str(reactant))))
                 
        if self._controller:
            for name, value in self._controller._widgetsFreeParams.items():
                if name in model._ratesLaTeX:
                    name = model._ratesLaTeX[name]
                name = name.replace('(', '')
                name = name.replace(')', '')
                if name not in paramInitCheck:
                    params.append((latex(name) , value.value))
        for name, value in self._fixedParams.items():
            #if name == 'systemSize' or name == 'plotLimits': continue
            if name not in self._mumotModel._rates and name not in self._mumotModel._constantReactants: continue
            name = repr(name)
            if name in model._ratesLaTeX:
                name = model._ratesLaTeX[name]
            name = name.replace('(', '')
            name = name.replace(')', '')
            params.append((latex(name) , value))
        params.append(('plotLimits' , self._getPlotLimits()))
        params.append(('systemSize' , self._getSystemSize()))
        
        return params

    def _get_bookmarks_params(self, refModel=None):
        params = self._get_params(refModel)
        logStr = "params = ["
        for name, value in params:
            logStr += "('" + str(name) + "', " + str(value) + "), "
        logStr = logStr[:-2]  # throw away last ", "
        logStr += "]"
        return logStr        


    def _build_bookmark(self, _=None):
        self._resetErrorMessage()
        self._showErrorMessage("Bookmark functionality not implemented for class " + str(self._generatingCommand))
        return


    def _getPlotLimits(self, defaultLimits=1):
#         if self._paramNames is not None and 'plotLimits' in self._paramNames:
        if self._fixedParams is not None and 'plotLimits' in self._fixedParams:
#             systemSize = self._paramValues[self._paramNames.index('plotLimits')]
            plotLimits = self._fixedParams['plotLimits']
        elif self._controller is not None and self._controller._plotLimitsWidget is not None:
            plotLimits = self._controller._plotLimitsWidget.value
        else:
            plotLimits = defaultLimits
            
        return plotLimits


    def _getSystemSize(self, defaultSize=1):
        # if self._paramNames is not None and 'systemSize' in self._paramNames:
        if self._fixedParams is not None and 'systemSize' in self._fixedParams:
#             systemSize = self._paramValues[self._paramNames.index('systemSize')]
            systemSize = self._fixedParams['systemSize']
        elif self._controller is not None and self._controller._systemSizeWidget is not None:
            systemSize = self._controller._systemSizeWidget.value
        else:
            systemSize = defaultSize
            
        return systemSize

    ## gets and returns names and values from widgets
    def _get_argDict(self):
        paramNames = []
        paramValues = []
        if self._controller is not None:
            for name, value in self._controller._widgetsFreeParams.items():
                #print("wdg-name: " + str(name) + " wdg-val: " + str(value.value))
                paramNames.append(name)
                paramValues.append(value.value)
        
#             if self._controller._widgetsExtraParams and 'initBifParam' in self._controller._widgetsExtraParams:     
#                 paramNames.append(self._bifurcationParameter_for_get_argDict)
#                 paramValues.append(self._controller._widgetsExtraParams['initBifParam'].value)
# 
#         if self._fixedParams and 'initBifParam' in self._fixedParams:
#             paramNames.append(self._bifurcationParameter_for_get_argDict)
#             paramValues.append(self._fixedParams['initBifParam'])

        if self._fixedParams is not None:
            for key, item in self._fixedParams.items():
                #print("fix-name: " + str(key) + " fix-val: " + str(item))
                paramNames.append(str(key))
                paramValues.append(item)
        
        argNamesSymb = list(map(Symbol, paramNames))
        argDict = dict(zip(argNamesSymb, paramValues))

        if self._mumotModel._systemSize:
            argDict[self._mumotModel._systemSize] = 1

        #@todo: is this necessary? for which view?
        systemSize = Symbol('systemSize')
        argDict[systemSize] = self._getSystemSize()
        
        return argDict
    
    
    ## calculates stationary states of 1d system
    def _get_fixedPoints1d(self):
        argDict = self._get_argDict()

        EQ1 = self._mumotModel._equations[self._stateVariable1].subs(argDict)
       
        eps = 1e-8
        EQsol = solve((EQ1), (self._stateVariable1), dict=True)
        
        realEQsol = [{self._stateVariable1: sympy.re(EQsol[kk][self._stateVariable1])} for kk in range(len(EQsol)) if sympy.Abs(sympy.im(EQsol[kk][self._stateVariable1])) <= eps]
        
        MAT = Matrix([EQ1])
        JAC = MAT.jacobian([self._stateVariable1])
        
        eigList = []
        
        for nn in range(len(realEQsol)): 
            evSet = {}
            JACsub = JAC.subs([(self._stateVariable1, realEQsol[nn][self._stateVariable1])])
            #evSet = JACsub.eigenvals()
            eigVects = JACsub.eigenvects()
            for kk in range(len(eigVects)):
                evSet[eigVects[kk][0]] = (eigVects[kk][1])
            eigList.append(evSet)
        return realEQsol, eigList  # returns two lists of dictionaries
    
    
    
    ## calculate stationary states of 2d system
    def _get_fixedPoints2d(self):
       
        argDict = self._get_argDict()
        
        EQ1 = self._mumotModel._equations[self._stateVariable1].subs(argDict)
        EQ2 = self._mumotModel._equations[self._stateVariable2].subs(argDict)
       
        eps = 1e-8
        EQsolA = solve((EQ1, EQ2), (self._stateVariable1, self._stateVariable2), dict=True)
        
        EQsol = []
        addIndexToSolList = []
        for nn in range(len(EQsolA)):
            if len(EQsolA[nn]) == 2:
                addIndexToSolList.append(nn)
            #else:
            #    self._showErrorMessage('Some solutions for Fixed Points may not be unique.')
        
        for el in addIndexToSolList:
            EQsol.append(EQsolA[el])
        
        if len(EQsol) == 0:
            self._showErrorMessage('Could not compute any unique solutions for Fixed Points. ')
            return None, None
        elif len(EQsol) < len(EQsolA):
            self._showErrorMessage('Some solutions for Fixed Points may not be unique. ')
        
        
        #for nn in range(len(EQsol)):
        #    if len(EQsol[nn]) != 2:
        #        self._showErrorMessage('Some or all solutions are NOT unique.')
        #        return None, None
        
        realEQsol = [{self._stateVariable1: sympy.re(EQsol[kk][self._stateVariable1]), self._stateVariable2: sympy.re(EQsol[kk][self._stateVariable2])} for kk in range(len(EQsol)) if (sympy.Abs(sympy.im(EQsol[kk][self._stateVariable1])) <= eps and sympy.Abs(sympy.im(EQsol[kk][self._stateVariable2])) <= eps)]
        
        MAT = Matrix([EQ1, EQ2])
        JAC = MAT.jacobian([self._stateVariable1, self._stateVariable2])
        
        eigList = []
        for nn in range(len(realEQsol)): 
            evSet = {}
            JACsub = JAC.subs([(self._stateVariable1, realEQsol[nn][self._stateVariable1]), (self._stateVariable2, realEQsol[nn][self._stateVariable2])])
            #evSet = JACsub.eigenvals()
            eigVects = JACsub.eigenvects()
            for kk in range(len(eigVects)):
                evSet[eigVects[kk][0]] = (eigVects[kk][1], eigVects[kk][2])
            eigList.append(evSet)
        return realEQsol, eigList  # returns two lists of dictionaries
    
    ## calculate stationary states of 3d system
    def _get_fixedPoints3d(self):
        
        argDict = self._get_argDict()
        
        EQ1 = self._mumotModel._equations[self._stateVariable1].subs(argDict)
        EQ2 = self._mumotModel._equations[self._stateVariable2].subs(argDict)
        EQ3 = self._mumotModel._equations[self._stateVariable3].subs(argDict)
        
        eps = 1e-8
        EQsolA = solve((EQ1, EQ2, EQ3), (self._stateVariable1, self._stateVariable2, self._stateVariable3), dict=True)
        
        EQsol = []
        addIndexToSolList = []
        for nn in range(len(EQsolA)):
            if len(EQsolA[nn]) == 3:
                addIndexToSolList.append(nn)
            #else:
            #    self._showErrorMessage('Some solutions for Fixed Points may not be unique.')
        
        for el in addIndexToSolList:
            EQsol.append(EQsolA[el])
        
        if len(EQsol) == 0:
            self._showErrorMessage('Could not compute any unique solutions for Fixed Points. ')
            return None, None
        elif len(EQsol) < len(EQsolA):
            self._showErrorMessage('Some solutions for Fixed Points may not be unique. ')
        
        #for nn in range(len(EQsol)):
        #    if len(EQsol[nn]) != 3:
        #        self._showErrorMessage('Some or all solutions are NOT unique.')
        #        return None, None
        
        realEQsol = [{self._stateVariable1: sympy.re(EQsol[kk][self._stateVariable1]), self._stateVariable2: sympy.re(EQsol[kk][self._stateVariable2]), self._stateVariable3: sympy.re(EQsol[kk][self._stateVariable3])} for kk in range(len(EQsol)) if (sympy.Abs(sympy.im(EQsol[kk][self._stateVariable1])) <= eps and sympy.Abs(sympy.im(EQsol[kk][self._stateVariable2])) <= eps and sympy.Abs(sympy.im(EQsol[kk][self._stateVariable3])) <= eps)]
        
        MAT = Matrix([EQ1, EQ2, EQ3])
        JAC = MAT.jacobian([self._stateVariable1, self._stateVariable2, self._stateVariable3])
        
        eigList = []
        for nn in range(len(realEQsol)): 
            evSet = {}
            JACsub = JAC.subs([(self._stateVariable1, realEQsol[nn][self._stateVariable1]), (self._stateVariable2, realEQsol[nn][self._stateVariable2]), (self._stateVariable3, realEQsol[nn][self._stateVariable3])])
            #evSet = JACsub.eigenvals()
            eigVects = JACsub.eigenvects()
            for kk in range(len(eigVects)):
                evSet[eigVects[kk][0]] = (eigVects[kk][1], eigVects[kk][2])
            eigList.append(evSet)
            
        return realEQsol, eigList  # returns two lists of dictionaries
    
    def _update_params(self):
        """method to update parameters from widgets, if the view requires view-specific params they can be updated implementing _update_view_specific_params()"""
        freeParamDict = self._get_argDict()
        if self._controller is not None:
            # getting the rates' value
            self._ratesDict = {}
            for rule in self._mumotModel._rules:
                self._ratesDict[str(rule.rate)] = rule.rate.subs(freeParamDict)
                if self._ratesDict[str(rule.rate)] == float('inf') or self._ratesDict[str(rule.rate)] is sympy.zoo:
                    self._ratesDict[str(rule.rate)] = sys.maxsize
                    errorMsg = "WARNING! Rate with division by zero. \nThe rule has a rate with division by zero: \n" + str(rule.lhsReactants) + " --> " + str(rule.rhsReactants) + " with rate " + str(rule.rate) + ".\n The system has run the simulation with the maximum system value: " + str(self._ratesDict[str(rule.rate)]) 
                    self._showErrorMessage(errorMsg)
            #print("_ratesDict=" + str(self._ratesDict))
            self._systemSize = self._getSystemSize()
        self._update_view_specific_params(freeParamDict)

    def _getWidgetParamValue(self, key, dict = None):
        """check fixedParams then generatingKwargs for a key value otherwise return from dict"""
        if self._fixedParams.get(key) is not None:
            return self._fixedParams[key]
        elif self._generatingKwargs.get(key) is not None:
            return self._generatingKwargs[key]
        elif dict is not None:
            if dict.get(key) is not None:
                return dict[key].value
            else:
                raise MuMoTValueError('Could not find value for key \'' + key + '\'; if using a multicontroller try moving keyword definition down to creation of constitutent controllers')
        else:
            return None

    def _getInitialState(self, state, freeParamDict):
        """ get initial state from widgets, otherwise original initial state"""
        if state in self._mumotModel._constantReactants:
            return freeParamDict[state]
        elif self._controller._widgetsExtraParams.get('init'+str(state)) is not None:
            return self._controller._widgetsExtraParams['init'+str(state)].value
        else:
            return self._initialState[state]

    def _update_view_specific_params(self, freeParamDict=None): # @todo JARM: I don't see what purpose this serves - it is mostly ignored and I don't think will function as intended
        """interface method to update view-specific params from widgets"""
        if freeParamDict is None:
            freeParamDict = {}
        pass

    def _safeSymbol(self, item):
        """used in _update_view_specific_params"""
        if type(item) is Symbol:
            return item
        else:
            return Symbol(item)
                        
    def showLogs(self, tail=False):
        """Show logs from view.

        Parameters
        ----------
        tail : bool, optional
           Flag to show only tail entries from logs. Defaults to False.
        """
        if tail:
            tailLength = 5
            print("Showing last " + str(min(tailLength, len(self._logs))) + " of " + str(len(self._logs)) + " log entries:")
            for log in self._logs[-tailLength:]:
                log.show()
        else:
            for log in self._logs:
                log.show()
    

class MuMoTmultiView(MuMoTview):
    """Multi-view view.
    
    Tied closely to :class:`MuMoTmultiController`.
    
    """
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
    ## controllers (for building bookmarks)
    _controllers = None

    def __init__(self, controller, model, views, controllers, subPlotNum, **kwargs):
        super().__init__(model, controller, **kwargs)
        self._generatingCommand = "mumot.MuMoTmultiController"
        self._views = views
        self._controllers = controllers
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
            plt.gcf().set_size_inches(9, 4.5)
        
    def _plot(self, _=None):
        fig = plt.figure(self._figureNum)
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
                    plt.subplot(self._numRows, self._numColumns, subPlotNum, projection='3d')
                else:
                    plt.subplot(self._numRows, self._numColumns, subPlotNum)
                func()
            plt.subplots_adjust(left=0.12, bottom=0.25, right=0.98, top=0.9, wspace=0.45, hspace=None)
            #plt.tight_layout()
#                subplotNum += 1


    def _setLog(self, log):
        for view in self._views:
            view._setLog(log)


    def _print_standalone_view_cmd(self, includeParams = False):
        model = self._views[0]._mumotModel  # @todo this suppose that all models are the same for all views
        with io.capture_output() as log:
            if self._controller._silent == False:
                logStr = "bookmark = "
            else:
                logStr = ""
            logStr += self._generatingCommand + "(["
            for controller in self._controllers:
                logStr += controller._view._print_standalone_view_cmd(False) + ", "
            logStr = logStr[:-2]  # throw away last ", "
            logStr += "]"
            if includeParams:
                logStr += ", " + self._get_bookmarks_params(model)
            if len(self._generatingKwargs) > 0:
                logStr += ", "
                for key in self._generatingKwargs:
                    if type(self._generatingKwargs[key]) == str:
                        logStr += key + " = " + "\'" + str(self._generatingKwargs[key]) + "\'" + ", "
                    else:
                        logStr += key + " = " + str(self._generatingKwargs[key]) + ", "
                    
                logStr = logStr[:-2]  # throw away last ", "
            if 'silent' not in self._generatingKwargs: logStr += ", silent = " + str(self._silent)
            if 'bookmark' not in self._generatingKwargs: logStr += ", bookmark = False"
            logStr += ")"
            #logStr = logStr.replace('\\', '\\\\') ## @todo is this necessary?

            if not self._silent and logStr is not None:
                print(logStr)    
                self._logs.append(log)
            return logStr


    def _set_fixedParams(self, paramDict):
        self._fixedParams = paramDict
        for view in self._views:
#            view._set_fixedParams(paramDict)
            view._set_fixedParams({**paramDict, **view._fixedParams})  # this operation merge the two dictionaries with the second overriding the values of the first


class MuMoTmultiController(MuMoTcontroller):
    """Multi-view controller."""

    ## replot function list to invoke on views
    _replotFunctions = None

    def __init__(self, controllers, params=None, initWidgets=None,  **kwargs):
        global figureCounter

        if initWidgets is None:
            initWidgets = {}

        self._silent = kwargs.get('silent', False)
        self._replotFunctions = []
        fixedParamNames = None
        paramValuesDict = {}
        paramLabelDict = {}
        showPlotLimits = False
        showSystemSize = False
        views = []
        subPlotNum = 1
        model = None
        ## @todo assuming same model for all views. This operation is NOT correct when multicotroller views have different models
        #paramValuesDict = controllers[0]._view._mumotModel._create_free_param_dictionary_for_controller(inputParams=params if params is not None else [], initWidgets=initWidgets, showSystemSize=True, showPlotLimits=True )
        
        if params is not None:
            (fixedParamNames, fixedParamValues) = _process_params(params)
        for controller in controllers:
            # pass through the fixed params to each constituent view
            view = controller._view
            if params is not None:
#                 view._fixedParams = dict(zip(fixedParamNames, fixedParamValues))
#                view._fixedParams = {**dict(zip(fixedParamNames, fixedParamValues)), **view._fixedParams} # this operation merge the two dictionaries with the second overriding the values of the first
                view._set_fixedParams({**dict(zip(fixedParamNames, fixedParamValues)), **view._fixedParams})  # this operation merge the two dictionaries with the second overriding the values of the first
            for name, value in controller._widgetsFreeParams.items():
                #if params is None or name not in fixedParamNames:
                #    paramValueDict[name] = (value.value, value.min, value.max, value.step)
                if name in initWidgets:
                    paramValuesDict[name] = _parse_input_keyword_for_numeric_widgets(inputValue=_get_item_from_params_list(params if params is not None else [], name),
                                    defaultValueRangeStep=[MuMoTdefault._initialRateValue, MuMoTdefault._rateLimits[0], MuMoTdefault._rateLimits[1], MuMoTdefault._rateStep], 
                                    initValueRangeStep=initWidgets.get(name), 
                                    validRange=(-float("inf"), float("inf")))
                else:
                    paramValuesDict[name] = (value.value, value.min, value.max, value.step, not(params is None or name not in map(str, view._fixedParams.keys())))
            if controller._plotLimitsWidget is not None:
                showPlotLimits = True
                if 'plotLimits' in initWidgets:
                    paramValuesDict['plotLimits'] = _parse_input_keyword_for_numeric_widgets(inputValue=_get_item_from_params_list(params if params is not None else [], 'plotLimits'),
                                    defaultValueRangeStep=[MuMoTdefault._plotLimits, MuMoTdefault._plotLimitsLimits[0], MuMoTdefault._plotLimitsLimits[1], MuMoTdefault._plotLimitsStep], 
                                    initValueRangeStep=initWidgets.get('plotLimits'), 
                                    validRange=(-float("inf"), float("inf"))) 
                else:
                    paramValuesDict['plotLimits'] = (controller._plotLimitsWidget.value, controller._plotLimitsWidget.min, controller._plotLimitsWidget.max, controller._plotLimitsWidget.step, not(params is None or 'plotLimits' not in map(str, view._fixedParams.keys())))
            if controller._systemSizeWidget is not None:
                showSystemSize = True
                if 'systemSize' in initWidgets:
                    paramValuesDict['systemSize'] = _parse_input_keyword_for_numeric_widgets(inputValue=_get_item_from_params_list(params if params is not None else [], 'systemSize'),
                                    defaultValueRangeStep=[MuMoTdefault._systemSize, MuMoTdefault._systemSizeLimits[0], MuMoTdefault._systemSizeLimits[1], MuMoTdefault._systemSizeStep], 
                                    initValueRangeStep=initWidgets.get('systemSize'), 
                                    validRange=(1, float("inf")))
                else:
                    paramValuesDict['systemSize'] = (controller._systemSizeWidget.value, controller._systemSizeWidget.min, controller._systemSizeWidget.max, controller._systemSizeWidget.step, not(params is None or 'systemSize' not in map(str, view._fixedParams.keys())))
            paramLabelDict.update(controller._paramLabelDict)
#             for name, value in controller._widgetsExtraParams.items():
#                 widgetsExtraParamsTmp[name] = value
            if type(controller) is MuMoTmultiController:
#            if controller._replotFunction is None: ## presume this controller is a multi controller (@todo check?)
                for view in controller._view._views:
                    views.append(view)         
                               
#                if view._controller._replotFunction is None: ## presume this controller is a multi controller (@todo check?)
                for func, _, axes3d in controller._replotFunctions:
                    self._replotFunctions.append((func, subPlotNum, axes3d))                    
#                else:
#                    self._replotFunctions.append((view._controller._replotFunction, subPlotNum, view._axes3d))                    
            else:
                views.append(controller._view)
#                 if controller._replotFunction is None: ## presume this controller is a multi controller (@todo check?)
#                     for func, foo in controller._replotFunctions:
#                         self._replotFunctions.append((func, subPlotNum))                    
#                 else:
                self._replotFunctions.append((controller._replotFunction, subPlotNum, controller._view._axes3d))                    
            subPlotNum += 1
            # check if all views refer to same model
            if model is None:
                model = view._mumotModel
            elif model != view._mumotModel:
                raise MuMoTValueError('Multicontroller views do not all refer to same model')
            

#         for view in self._views:
#             if view._controller._replotFunction is None: ## presume this controller is a multi controller (@todo check?)
#                 for func in view._controller._replotFunctions:
#                     self._replotFunctions.append(func)                    
#             else:
#                 self._replotFunctions.append(view._controller._replotFunction)
#             view._controller = self
        super().__init__(paramValuesDict, paramLabelDict, False, showPlotLimits, showSystemSize, params=params, **kwargs)
        
        # handle Extra and PlotOnly params
        addProgressBar = False
        for controller in controllers:
            # retrieve the _widgetsExtraParams from each controller
            for name, widget in controller._widgetsExtraParams.items():
                widget.unobserve(controller._replotFunction, 'value')
                self._widgetsExtraParams[name] = widget
            # retrieve the _widgetsPlotOnly from each controller
            for name, widget in controller._widgetsPlotOnly.items():
                widget.unobserve(controller._redrawFunction, 'value')
                # in multiController, plotProportions=True is forced 
                if name == "plotProportions":
                    widget.value = True
                    widget.disabled = True
                # this is necessary due to limit visualisation-type to only 'evo' and 'final'
                if name == "visualisationType": 
                    ## Toggle buttons for plotting style 
                    widget = widgets.ToggleButtons(
                        options=[('Temporal evolution', 'evo'), ('Final distribution', 'final')],
                        value=widget.value,
                        description='Plot:',
                        disabled=False,
                        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                        tooltips=['Population change over time', 'Population distribution in each state at final timestep'],
                    )
                    widget.observe(self._updateFinalViewWidgets, 'value')
                self._widgetsPlotOnly[name] = widget
            # retrieve the _extraWidgetsOrder from each controller
            self._extraWidgetsOrder.extend(x for x in controller._extraWidgetsOrder if x not in self._extraWidgetsOrder)
#             if controller._progressBar:
#                 addProgressBar = True
        if self._widgetsExtraParams or self._widgetsPlotOnly:
            # set widgets to possible initial/fixed values if specified in the multi-controller
            #for key, value in kwargs.items():
            for key in kwargs.keys() | initWidgets.keys():
                inputValue = kwargs.get(key) 
                ep1 = None
                ep2 = None
                if key == 'choose_yrange':
                    for controller in controllers:
                        controller._view._chooseYrange = kwargs.get('choose_yrange')
                if key == 'choose_xrange':
                    for controller in controllers:
                        controller._view._chooseXrange = kwargs.get('choose_xrange')
                if key == 'initialState': ep1 = views[0]._mumotModel._getAllReactants()  # @todo assuming same model for all views. This operation is NOT correct when multicotroller views have different models
                if key == 'visualisationType': ep1 = "multicontroller"
                if key == 'final_x' or key == 'final_y': ep1 = views[0]._mumotModel._getAllReactants()[0]  # @todo assuming same model for all views. This operation is NOT correct when multicotroller views have different models
                if key == 'netParam': 
                    ep1 = [kwargs.get('netType', self._widgetsExtraParams.get('netType')), kwargs.get('netType') is not None] 
                    maxSysSize = 1
                    for view in views:
                        maxSysSize = max(maxSysSize, view._getSystemSize())
                    ep2 = maxSysSize
                optionValues = _format_advanced_option(optionName=key, inputValue=inputValue, initValues=initWidgets.get(key), extraParam=ep1, extraParam2=ep2)
                # if option is fixed
                if optionValues[-1] == True:
                    if key == 'initialState':  # initialState is special
                        for state, pop in optionValues[0].items():
                            optionValues[0][state] = pop[0]
                            stateKey = "init" + str(state)
                            # delete the widgets
                            if stateKey in self._widgetsExtraParams:
                                del self._widgetsExtraParams[stateKey]
                    if key == 'netType':  # netType is special
                        optionValues[0] = _decodeNetworkTypeFromString(optionValues[0])  # @todo: if only netType (and not netParam) is specified, then multicotroller won't work...
                    if key == 'visualisationType' and optionValues[0] == 'final':  # visualisationType == 'final' is special
                        if self._widgetsPlotOnly.get('final_x') is not None:
                            self._widgetsPlotOnly['final_x'].layout.display = 'flex'
                        if self._widgetsPlotOnly.get('final_y') is not None:
                            self._widgetsPlotOnly['final_y'].layout.display = 'flex'
                    # set the value in all the views
                    for view in views:                            
                        view._fixedParams[key] = optionValues[0]
                    # delete the widgets
                    if key in self._widgetsExtraParams:
                        del self._widgetsExtraParams[key]
                    if key in self._widgetsPlotOnly:
                        del self._widgetsPlotOnly[key]
                else:
                    #update the values with the init values
                    if key in self._widgetsExtraParams:
                        if len(optionValues) == 5:
                            self._widgetsExtraParams[key].max = 10**7  # temp to avoid exception min>max
                            self._widgetsExtraParams[key].min = optionValues[1]
                            self._widgetsExtraParams[key].max = optionValues[2]
                            self._widgetsExtraParams[key].step = optionValues[3]
                            self._widgetsExtraParams[key].readout_format = '.' + str(_count_sig_decimals(str(optionValues[3]))) + 'f'
                        self._widgetsExtraParams[key].value = optionValues[0]
                    if key in self._widgetsPlotOnly:
                        if len(optionValues) == 5:
                            self._widgetsPlotOnly[key].max = 10**7  # temp to avoid exception min>max
                            self._widgetsPlotOnly[key].min = optionValues[1]
                            self._widgetsPlotOnly[key].max = optionValues[2]
                            self._widgetsPlotOnly[key].step = optionValues[3]
                            self._widgetsPlotOnly[key].readout_format = '.' + str(_count_sig_decimals(str(optionValues[3]))) + 'f'
                        self._widgetsPlotOnly[key].value = optionValues[0]
                    if key == 'initialState':
                        for state, pop in optionValues[0].items():
#                             self._widgetsExtraParams['init'+str(state)].unobserve(self._updateInitialStateWidgets, 'value')
                            self._widgetsExtraParams['init'+str(state)].max = float('inf')  # temp to avoid exception min>max
                            self._widgetsExtraParams['init'+str(state)].min = pop[1]
                            self._widgetsExtraParams['init'+str(state)].max = pop[2]
                            self._widgetsExtraParams['init'+str(state)].step = pop[3]
                            self._widgetsExtraParams['init'+str(state)].readout_format = '.' + str(_count_sig_decimals(str(pop[3]))) + 'f'
                            self._widgetsExtraParams['init'+str(state)].value = pop[0] 
#                             self._widgetsExtraParams['init'+str(state)].observe(self._updateInitialStateWidgets, 'value')
            
            # create the "Advanced options" tab
            if not self._silent:
                self._displayAdvancedOptionsTab()
        
        # if necessary adding the progress bar
        if addProgressBar:
            # Loading bar (useful to give user progress status for long executions)
            self._progressBar = widgets.FloatProgress(
                value=0,
                min=0,
                max=self._widgetsExtraParams['maxTime'].value if self._widgetsExtraParams.get('maxTime') is not None else views[0]._fixedParams.get('maxTime'),
                #step=1,
                description='Loading:',
                bar_style='success',  # 'success', 'info', 'warning', 'danger' or ''
                style={'description_width': 'initial'},
                orientation='horizontal'
            )
            if not self._silent:
                display(self._progressBar)

        self._view = MuMoTmultiView(self, model, views, controllers, subPlotNum - 1, **kwargs)
        if fixedParamNames is not None:
#            self._view._fixedParams = dict(zip(fixedParamNames, fixedParamValues))
            self._view._set_fixedParams(dict(zip(fixedParamNames, fixedParamValues))) 
                
        for controller in controllers:
            controller._setErrorWidget(self._errorMessage)
        
        ## @todo handle correctly the re-draw only widgets and function
        self._setReplotFunction(self._view._plot, self._view._plot)
        
        #silent = kwargs.get('silent', False)
        if not self._silent:
            self._view._plot()


class MuMoTtimeEvolutionView(MuMoTview):
    """Time evolution view on model including state variables and noise.
    
    Specialised by :class:`MuMoTintegrateView` and :class:`MuMoTnoiseCorrelationsView`.

    """
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
#   ## end time of numerical simulation of ODE system of the state variables
#    _tend = None
    ## simulation length in time units
    _maxTime = None
    ## time step of numerical simulation
    _tstep = None
    ## defines fontsize on the axes
    _chooseFontSize = None
    ## string that defines the x-label
    _xlab = None
    ## legend location: combinations like 'upper left', lower right, or 'center center' are allowed (9 options in total)
    _legend_loc = None
    ## legend fontsize, accepts integers
    _legend_fontsize = None
    ## total number of agents in the simulation
    _systemSize = None
    ## the system state at the start of the simulation (timestep zero)
    _initialState = None
    ## flag to plot proportions or full populations
    _plotProportions = None
    ## Parameters for controller specific to this time evolution view
    _tEParams = None
    
    ## displayed range for vertical axis
    _chooseXrange = None
    ## displayed range for horizontal axis
    _chooseYrange = None
    
    #def __init__(self, model, controller, stateVariable1, stateVariable2, stateVariable3 = None, stateVariable4 = None, figure = None, params = None, **kwargs):
    def __init__(self, model, controller, tEParams, showStateVars=None, figure=None, params=None, **kwargs):
        #if model._systemSize is None and model._constantSystemSize == True:
        #    print("Cannot construct time evolution -based plot until system size is set, using substitute()")
        #    return
        self._silent = kwargs.get('silent', False)
        super().__init__(model=model, controller=controller, figure=figure, params=params, **kwargs)
        #super().__init__(model, controller, figure, params, **kwargs)
        
        self._tEParams = tEParams
        self._chooseXrange = kwargs.get('choose_xrange', None)
        self._chooseYrange = kwargs.get('choose_yrange', None)
        
        with io.capture_output() as log:
#         if True:
#             log=''

            self._systemSize = self._getSystemSize()
             
            if self._controller is None:
                # storing the initial state
                self._initialState = {}
                for state, pop in tEParams["initialState"].items():
                    if isinstance(state, str):
                        self._initialState[process_sympy(state)] = pop  # convert string into SymPy symbol
                    else:
                        self._initialState[state] = pop
#####
#                # add to the _initialState the constant reactants
#                for constantReactant in self._mumotModel._getAllReactants()[1]:
#                    self._initialState[constantReactant] = freeParamDict[constantReactant]
#####
                # storing all values of MA-specific parameters
                self._maxTime = tEParams["maxTime"]
                self._plotProportions = tEParams["plotProportions"]
            
            else:
                # storing the initial state
                self._initialState = {}
                for state, pop in tEParams["initialState"][0].items():
                    if isinstance(state, str):
                        self._initialState[process_sympy(state)] = pop[0]  # convert string into SymPy symbol
                    else:
                        self._initialState[state] = pop[0]
######                        
#                # add to the _initialState the constant reactants
#                for constantReactant in self._mumotModel._getAllReactants()[1]:
#                    self._initialState[constantReactant] = freeParamDict[constantReactant]
######                    
                # storing fixed params
                for key, value in tEParams.items():
                    if value[-1]:
                        if key == 'initialState':
                            self._fixedParams[key] = self._initialState
                        else:
                            self._fixedParams[key] = value[0]
            
            if 'fontsize' in kwargs:
                self._chooseFontSize = kwargs['fontsize']
            else:
                self._chooseFontSize = None
            self._xlab = kwargs.get('xlab', 'time t')
            #self._ylab = kwargs.get('ylab', r'evolution of states')
            
            self._legend_loc = kwargs.get('legend_loc', 'upper right')
            self._legend_fontsize = kwargs.get('legend_fontsize', None)
            
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

    
    def _plot_NumSolODE(self):
        if not(self._silent):  # @todo is this necessary?
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
#                self._initialState = self._getWidgetParamValue('initialState', None)
#                self._initialState = { Symbol(key): self._getWidgetParamValue('initialState', None)[key] for key in self._getWidgetParamValue('initialState', None)}
                self._initialState = { self._safeSymbol(key): self._getWidgetParamValue('initialState', None)[key] for key in self._getWidgetParamValue('initialState', None)}
            else:
                for state in self._initialState.keys():
                    # add normal and constant reactants to the _initialState 
                    self._initialState[state] = self._getInitialState(state, freeParamDict) # freeParamDict[state] if state in self._mumotModel._constantReactants else self._controller._widgetsExtraParams['init'+str(state)].value
            if 'plotProportions' in self._tEParams: # @todo JARM: I don't really understand logic of checking _tEParams but then retrieving the value from elsewhere
                self._plotProportions = self._getWidgetParamValue('plotProportions', self._controller._widgetsPlotOnly) # self._fixedParams['plotProportions'] if self._fixedParams.get('plotProportions') is not None else self._controller._widgetsPlotOnly['plotProportions'].value
            self._maxTime = self._getWidgetParamValue('maxTime', self._controller._widgetsExtraParams) # self._fixedParams['maxTime'] if self._fixedParams.get('maxTime') is not None else self._controller._widgetsExtraParams['maxTime'].value
               
 
class MuMoTintegrateView(MuMoTtimeEvolutionView):
    """Numerical solution of ODEs plot view on model."""
    
    ## y-label with default specific to this MuMoTintegrateView class (can be set via keyword)
    _ylab = None
    ## initial conditions used for proportion plot
    _y0 = None
    ## save solution for redraw to switch between plotProportions = True and False
    _sol_ODE_dict = None
    ## ordered list of colors to be used
    _colors = None
    
    def _constructorSpecificParams(self, _):
        if self._controller is not None:
            self._generatingCommand = "integrate"
        
        self._colors = []
        for idx, state in enumerate(sorted(self._initialState.keys(), key=str)):
            if state in self._stateVarListDisplay:
                self._colors.append(LINE_COLOR_LIST[idx])
        #print(self._colors) 
    
    def __init__(self, *args, **kwargs):
        self._ylab = kwargs.get('ylab', 'reactants')
        super().__init__(*args, **kwargs)
        #self._generatingCommand = "numSimStateVar"

    def _plot_NumSolODE(self, _=None):
        self._show_computation_start()
        
        super()._plot_NumSolODE()
        
        with io.capture_output() as log:
            self._log("numerical integration of ODE system")
        self._logs.append(log)
        
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
        
        NrDP = int(self._maxTime/self._tstep) + 1
        time = np.linspace(0, self._maxTime, NrDP)
        
        initDict = self._initialState  #self._get_tEParams()   #self._initialState
         
   
        #if len(initDict) < 2 or len(initDict) > 4:
        #    self._showErrorMessage("Not implemented: This feature is available only for systems with 2, 3 or 4 time-dependent reactants!")

        y0 = []
        for nn in range(len(self._stateVarList)):
            #SVi0 = initDict[Symbol(latex(Symbol('Phi^0_'+str(self._stateVarList[nn]))))]
            SVi0 = initDict[Symbol(str(self._stateVarList[nn]))]
            y0.append(SVi0)
        
        self._y0 = y0        

        sol_ODE = odeint(self._get_eqsODE, y0, time)  
        
        sol_ODE_dict = {}
        for nn in range(len(self._stateVarList)):
            sol_ODE_dict[str(self._stateVarList[nn])] = sol_ODE[:, nn]
        
        self._sol_ODE_dict = sol_ODE_dict  
        #x_data = [time for kk in range(len(self._get_eqsODE(y0, time)))]
        x_data = [time for kk in range(len(self._stateVarListDisplay))]
        #y_data = [sol_ODE[:, kk] for kk in range(len(self._get_eqsODE(y0, time)))]
        y_data = [sol_ODE_dict[str(self._stateVarListDisplay[kk])] for kk in range(len(self._stateVarListDisplay))]
        
        if self._plotProportions == False:
            syst_Size = Symbol('systemSize')
            sysS = syst_Size.subs(self._get_argDict())
            #sysS = syst_Size.subs(self._getSystemSize())
            sysS = sympy.N(sysS)
            #y_scaling = np.sum(np.asarray(y0))
            #if y_scaling > 0:
            #    sysS = sysS/y_scaling
            for nn in range(len(y_data)):
                y_temp = np.copy(y_data[nn])
                for kk in range(len(y_temp)):
                    y_temp[kk] = y_temp[kk]*sysS
                y_data[nn] = y_temp
            c_labels = [r'$'+str(self._stateVarListDisplay[nn])+'$' for nn in range(len(self._stateVarListDisplay))]
        else:
            c_labels = [r'$'+latex(Symbol('Phi_'+str(self._stateVarListDisplay[nn]))) + '$' for nn in range(len(self._stateVarListDisplay))] 
        
        c_labels = [_doubleUnderscorify(_greekPrependify(c_labels[jj])) for jj in range(len(c_labels))]
#         
#         c_labels = [r'$'+latex(Symbol('Phi_'+str(self._stateVariable1)))+'$', r'$'+latex(Symbol('Phi_'+str(self._stateVariable2)))+'$'] 
#         if self._stateVariable3:
#             c_labels.append(r'$'+latex(Symbol('Phi_'+str(self._stateVariable3)))+'$')
#         if self._stateVariable4:
#             c_labels.append(r'$'+latex(Symbol('Phi_'+str(self._stateVariable4)))+'$')         
        
        if self._chooseXrange:
            choose_xrange = self._chooseXrange
        else:
            choose_xrange = [0, self._maxTime]
        _fig_formatting_2D(xdata=x_data, ydata=y_data , xlab=self._xlab, ylab=self._ylab, choose_xrange=choose_xrange,
                           choose_yrange=self._chooseYrange, fontsize=self._chooseFontSize, curvelab=c_labels, legend_loc=self._legend_loc, grid=True,
                           legend_fontsize=self._legend_fontsize, line_color_list=self._colors)
        
        
        with io.capture_output() as log:
            print('Last point on curve:')  
            if self._plotProportions == False:
                for nn in range(len(self._stateVarListDisplay)):
                    out = latex(str(self._stateVarListDisplay[nn])) + '(t =' + str(_roundNumLogsOut(x_data[nn][-1])) + ') = ' + str(_roundNumLogsOut(y_data[nn][-1]))
                    out = _doubleUnderscorify(_greekPrependify(out))
                    display(Math(out))
            else:
                for nn in range(len(self._stateVarListDisplay)):
                    out = latex(Symbol('Phi_'+str(self._stateVarListDisplay[nn]))) + '(t =' + str(_roundNumLogsOut(x_data[nn][-1])) + ') = ' + str(_roundNumLogsOut(y_data[nn][-1]))
                    out = _doubleUnderscorify(_greekPrependify(out))
                    display(Math(out))
        self._logs.append(log)
        
        self._show_computation_stop()
    
    
    def _redrawOnly(self, _=None):
        super()._plot_NumSolODE()
        self._update_params()
        NrDP = int(self._maxTime/self._tstep) + 1
        time = np.linspace(0, self._maxTime, NrDP)
        #x_data = [time for kk in range(len(self._get_eqsODE(y0, time)))]
        x_data = [time for kk in range(len(self._stateVarListDisplay))]
        #y_data = [sol_ODE[:, kk] for kk in range(len(self._get_eqsODE(y0, time)))]
        y_data = [self._sol_ODE_dict[str(self._stateVarListDisplay[kk])] for kk in range(len(self._stateVarListDisplay))]
        
        if self._plotProportions == False:
            syst_Size = Symbol('systemSize')
            sysS = syst_Size.subs(self._get_argDict())
            #sysS = syst_Size.subs(self._getSystemSize())
            sysS = sympy.N(sysS)
            #y_scaling = np.sum(np.asarray(self._y0))
            #if y_scaling > 0:
            #    sysS = sysS/y_scaling
            for nn in range(len(y_data)):
                y_temp = np.copy(y_data[nn])
                for kk in range(len(y_temp)):
                    y_temp[kk] = y_temp[kk]*sysS
                y_data[nn] = y_temp
            c_labels = [r'$'+str(self._stateVarListDisplay[nn])+'$' for nn in range(len(self._stateVarListDisplay))]
        else:
            c_labels = [r'$'+latex(Symbol('Phi_'+str(self._stateVarListDisplay[nn]))) + '$' for nn in range(len(self._stateVarListDisplay))] 
        
        c_labels = [_doubleUnderscorify(_greekPrependify(c_labels[jj])) for jj in range(len(c_labels))]
#         
#         c_labels = [r'$'+latex(Symbol('Phi_'+str(self._stateVariable1)))+'$', r'$'+latex(Symbol('Phi_'+str(self._stateVariable2)))+'$'] 
#         if self._stateVariable3:
#             c_labels.append(r'$'+latex(Symbol('Phi_'+str(self._stateVariable3)))+'$')
#         if self._stateVariable4:
#             c_labels.append(r'$'+latex(Symbol('Phi_'+str(self._stateVariable4)))+'$')         
#         
        if self._chooseXrange:
            choose_xrange = self._chooseXrange
        else:
            choose_xrange = [0, self._maxTime]
        _fig_formatting_2D(xdata=x_data, ydata=y_data , xlab=self._xlab, ylab=self._ylab, choose_xrange=choose_xrange,
                           choose_yrange=self._chooseYrange, fontsize=self._chooseFontSize, curvelab=c_labels, legend_loc=self._legend_loc, grid=True,
                           legend_fontsize=self._legend_fontsize)
        
        with io.capture_output() as log:
            print('Last point on curve:')  
            if self._plotProportions == False:
                for nn in range(len(self._stateVarListDisplay)):
                    out = latex(str(self._stateVarListDisplay[nn])) + '(t =' + str(x_data[nn][-1]) + ') = ' + str(str(y_data[nn][-1]))
                    out = _doubleUnderscorify(_greekPrependify(out))
                    display(Math(out))
            else:
                for nn in range(len(self._stateVarListDisplay)):
                    out = latex(Symbol('Phi_'+str(self._stateVarListDisplay[nn]))) + '(t =' + str(x_data[nn][-1]) + ') = ' + str(str(y_data[nn][-1]))
                    out = _doubleUnderscorify(_greekPrependify(out))
                    display(Math(out))
        self._logs.append(log)

    def _build_bookmark(self, includeParams=True):
        if not self._silent:
            logStr = "bookmark = "
        else:
            logStr = ""
        
        logStr += "<modelName>." + self._generatingCommand + "(showStateVars=["
        for nn in range(len(self._stateVarListDisplay)):
            if nn == len(self._stateVarListDisplay)-1:
                logStr += "'" + str(self._stateVarListDisplay[nn]) + "'], "
            else:
                logStr += "'" + str(self._stateVarListDisplay[nn]) + "', "
        
        if "initialState" not in self._generatingKwargs.keys():
            initState_str = {latex(state): pop for state, pop in self._initialState.items() if not state in self._mumotModel._constantReactants}
            logStr += "initialState = " + str(initState_str) + ", "
        if "maxTime" not in self._generatingKwargs.keys():
            logStr += "maxTime = " + str(self._maxTime) + ", "
        if "plotProportions" not in self._generatingKwargs.keys():
            logStr += "plotProportions = " + str(self._plotProportions) + ", "
        if includeParams:
            logStr += self._get_bookmarks_params() + ", "
        if len(self._generatingKwargs) > 0:
            for key in self._generatingKwargs:
                if type(self._generatingKwargs[key]) == str:
                    logStr += key + " = " + "\'" + str(self._generatingKwargs[key]) + "\'" + ", "
                else:
                    logStr += key + " = " + str(self._generatingKwargs[key]) + ", "    
        logStr += "bookmark = False"
        logStr += ")"
        logStr = _greekPrependify(logStr)
        logStr = logStr.replace('\\', '\\\\')
        logStr = logStr.replace('\\\\\\\\', '\\\\')
        
        return logStr


class MuMoTnoiseCorrelationsView(MuMoTtimeEvolutionView):
    """Noise correlations around fixed points plot view on model."""
    
    ## equations of motion for first order moments of noise variables
    _EOM_1stOrdMomDict = None
    ## equations of motion for second order moments of noise variables
    _EOM_2ndOrdMomDict = None
    ## upper bound of simulation time for dynamical system to reach equilibrium (can be set via keyword)
    _maxTimeDS = None
    ## time step of simulation for dynamical system to reach equilibrium (can be set via keyword)
    _tstepDS = None
    ## y-label with default specific to this MuMoTnoiseCorrelationsView class (can be set via keyword)
    _ylab = None
    
    def _constructorSpecificParams(self, _):
        if self._controller is not None:
            self._generatingCommand = "noiseCorrelations"
    
    def __init__(self, model, controller, NCParams, EOM_1stOrdMom, EOM_2ndOrdMom, figure=None, params=None, **kwargs):
        self._EOM_1stOrdMomDict = EOM_1stOrdMom
        self._EOM_2ndOrdMomDict = EOM_2ndOrdMom
        self._maxTimeDS = kwargs.get('maxTimeDS', 50)
        self._tstepDS = kwargs.get('tstepDS', 0.01)
        self._ylab = kwargs.get('ylab', 'noise correlations')
        self._silent = kwargs.get('silent', False)
        super().__init__(model=model, controller=controller, tEParams=NCParams, showStateVars=None, figure=figure, params=params, **kwargs)
        #super().__init__(model, controller, None, figure, params, **kwargs)
        
        if len(self._stateVarList) < 1 or len(self._stateVarList) > 3:
            self._showErrorMessage("Not implemented: This feature is available only for systems with 1, 2 or 3 time-dependent reactants!")
            return None
    
    def _plot_NumSolODE(self, _=None):
        self._show_computation_start()
        
        super()._plot_NumSolODE()
        
        with io.capture_output() as log:
            self._log("numerical integration of noise correlations")
        self._logs.append(log)
        
        # check input
        for nn in range(len(self._stateVarListDisplay)):
            if self._stateVarListDisplay[nn] not in self._stateVarList:
                self._showErrorMessage('Warning:  '+str(self._stateVarListDisplay[nn])+'  is no reactant in the current model.')
                return None

        eps = 5e-3
        systemSize = Symbol('systemSize')
        
        
        NrDP = int(self._maxTimeDS/self._tstepDS) + 1
        time = np.linspace(0, self._maxTimeDS, NrDP)
        #NrDP = int(self._tend/self._tstep) + 1
        #time = np.linspace(0, self._tend, NrDP)
        
        initDict = self._initialState
          
        SV1_0 = initDict[Symbol(str(self._stateVariable1))]
        y0 = [SV1_0]
        if self._stateVariable2:
            SV2_0 = initDict[Symbol(str(self._stateVariable2))]
            y0.append(SV2_0)
        if self._stateVariable3:
            SV3_0 = initDict[Symbol(str(self._stateVariable3))]
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
                        if all(sympy.sign(sympy.re(lam)) < 0 for lam in eigList[kk]) == True:
                            steadyStateReached = True
                            steadyStateDict = {self._stateVariable1: realEQsol[kk][self._stateVariable1], self._stateVariable2: realEQsol[kk][self._stateVariable2], self._stateVariable3: realEQsol[kk][self._stateVariable3]}
    
                elif self._stateVariable2:
                    if abs(realEQsol[kk][self._stateVariable1] - y_stationary[0]) <= eps and abs(realEQsol[kk][self._stateVariable2] - y_stationary[1]) <= eps:
                        if all(sympy.sign(sympy.re(lam)) < 0 for lam in eigList[kk]) == True:
                            steadyStateReached = True
                            steadyStateDict = {self._stateVariable1: realEQsol[kk][self._stateVariable1], self._stateVariable2: realEQsol[kk][self._stateVariable2]}
                
                else:
                    if abs(realEQsol[kk][self._stateVariable1] - y_stationary[0]) <= eps:
                        if all(sympy.sign(sympy.re(lam)) < 0 for lam in eigList[kk]) == True:
                            steadyStateReached = True
                            steadyStateDict = {self._stateVariable1: realEQsol[kk][self._stateVariable1]}
            
                
            if steadyStateReached == False:
                self._show_computation_stop()
                self._showErrorMessage('ODE system could not reach stable steady state: Try changing the initial conditions or model parameters using the sliders provided, increase simulation time, or decrease timestep tstep.') 
                return None
        else:
            steadyStateReached = 'uncertain'
            self._showErrorMessage('Warning: ODE system may not have reached a steady state. Values of state variables at t=maxTimeDS were substituted (maxTimeDS can be set via keyword \'maxTimeDS = <number>\').')
            if self._stateVariable3:
                steadyStateDict = {self._stateVariable1: y_stationary[0], self._stateVariable2: y_stationary[1], self._stateVariable3: y_stationary[2]}
            elif self._stateVariable2:
                steadyStateDict = {self._stateVariable1: y_stationary[0], self._stateVariable2: y_stationary[1]}
            else:
                steadyStateDict = {self._stateVariable1: y_stationary[0]}    
         
        with io.capture_output() as log:
            if steadyStateReached == 'uncertain':
                print('This plot depicts the noise-noise auto-correlation and cross-correlation functions around the following state (this might NOT be a steady state).')  
            else:  
                print('This plot depicts the noise-noise auto-correlation and cross-correlation functions around the following stable steady state:')
            for reactant in steadyStateDict:
                out = 'Phi^s_{' + latex(str(reactant)) + '} = ' + latex(_roundNumLogsOut(steadyStateDict[reactant]))
                out = _doubleUnderscorify(_greekPrependify(out))
                display(Math(out))
        self._logs.append(log)
         
        argDict = self._get_argDict()
        for key in self._mumotModel._constantReactants:
            if argDict[key] is not None:
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
        
        time_depend_noise = []
        for reactant in self._mumotModel._reactants:
            if reactant not in self._mumotModel._constantReactants:
                time_depend_noise.append(Symbol('eta_'+str(reactant)))
        
        noiseCorrEOM = []
        noiseCorrEOMdict = {}
        for sym in time_depend_noise:
            for key in EOM_1stOrdMomDict:
                noiseCorrEOMdict[sym*key] = expand(sym*EOM_1stOrdMomDict[key])
        
        M_1, M_2 = symbols('M_1 M_2')
        eta_SV1 = Symbol('eta_'+str(self._stateVariable1))
        cVar1 = symbols('cVar1')
        if self._stateVariable2:
            eta_SV2 = Symbol('eta_'+str(self._stateVariable2))
            cVar2, cVar3, cVar4 = symbols('cVar2 cVar3 cVar4')
        if self._stateVariable3:
            eta_SV3 = Symbol('eta_'+str(self._stateVariable3))
            cVar5, cVar6, cVar7, cVar8, cVar9 = symbols('cVar5 cVar6 cVar7 cVar8 cVar9')
        
        cVarSubdict = {}    
        if len(time_depend_noise) == 1:
            cVarSubdict[eta_SV1*M_1(eta_SV1)] = cVar1
            #auto-correlation
            noiseCorrEOM.append(noiseCorrEOMdict[eta_SV1*M_1(eta_SV1)])
        elif len(time_depend_noise) == 2:
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
                elif self._stateVariable2:
                    dydt[kk] = dydt[kk].subs({cVar1: yin[0], cVar2: yin[1], cVar3: yin[2], cVar4: yin[3]})
                else:
                    dydt[kk] = dydt[kk].subs({cVar1: yin[0]})
                dydt[kk] = dydt[kk].evalf()
            return dydt
        
        NrDP = int(self._maxTime/self._tstep) + 1
        time = np.linspace(0, self._maxTime, NrDP)
        
        if len(SOL_2ndOrdMomDict) > 0:
            if self._stateVariable3:
                y0 = [SOL_2ndOrdMomDict[M_2(eta_SV1**2)], SOL_2ndOrdMomDict[M_2(eta_SV2**2)], SOL_2ndOrdMomDict[M_2(eta_SV3**2)], 
                      SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV2)], SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV2)],
                      SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV3)], SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV3)],
                      SOL_2ndOrdMomDict[M_2(eta_SV2*eta_SV3)], SOL_2ndOrdMomDict[M_2(eta_SV2*eta_SV3)]]
            elif self._stateVariable2:
                y0 = [SOL_2ndOrdMomDict[M_2(eta_SV1**2)], SOL_2ndOrdMomDict[M_2(eta_SV2**2)], 
                      SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV2)], SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV2)]]
            else:
                y0 = [SOL_2ndOrdMomDict[M_2(eta_SV1**2)]]
        else:
            self._showErrorMessage('Could not compute Second Order Moments. Could not generate figure. Try different initial conditions in the Advanced options tab! ')
            return None
                
        sol_ODE = odeint(noiseODEsys, y0, time)  # sol_ODE overwritten
        
        x_data = [time for kk in range(len(y0))]  
        y_data = [sol_ODE[:, kk] for kk in range(len(y0))]
        noiseNorm = systemSize.subs(argDict)
        noiseNorm = sympy.N(noiseNorm)
        for nn in range(len(y_data)):
            y_temp = np.copy(y_data[nn])
            for kk in range(len(y_temp)):
                y_temp[kk] = y_temp[kk]/noiseNorm
            y_data[nn] = y_temp
            
        if self._stateVariable3:
            c_labels = [r'$<'+latex(eta_SV1)+'(t)'+latex(eta_SV1)+'(0)' + '>$', 
                        r'$<'+latex(eta_SV2)+'(t)'+latex(eta_SV2)+'(0)' + '>$',
                        r'$<'+latex(eta_SV3)+'(t)'+latex(eta_SV3)+'(0)' + '>$', 
                        r'$<'+latex(eta_SV2)+'(t)'+latex(eta_SV1)+'(0)' + '>$', 
                        r'$<'+latex(eta_SV1)+'(t)'+latex(eta_SV2)+'(0)' + '>$',
                        r'$<'+latex(eta_SV3)+'(t)'+latex(eta_SV1)+'(0)' + '>$', 
                        r'$<'+latex(eta_SV1)+'(t)'+latex(eta_SV3)+'(0)' + '>$',
                        r'$<'+latex(eta_SV3)+'(t)'+latex(eta_SV2)+'(0)' + '>$', 
                        r'$<'+latex(eta_SV2)+'(t)'+latex(eta_SV3)+'(0)' + '>$']
            
        elif self._stateVariable2:
            c_labels = [r'$<'+latex(eta_SV1)+'(t)'+latex(eta_SV1)+'(0)' + '>$', 
                        r'$<'+latex(eta_SV2)+'(t)'+latex(eta_SV2)+'(0)' + '>$', 
                        r'$<'+latex(eta_SV2)+'(t)'+latex(eta_SV1)+'(0)' + '>$', 
                        r'$<'+latex(eta_SV1)+'(t)'+latex(eta_SV2)+'(0)' + '>$']
        else:
            c_labels = [r'$<'+latex(eta_SV1)+'(t)'+latex(eta_SV1)+'(0)' + '>$']
        
        c_labels = [_doubleUnderscorify(_greekPrependify(c_labels[jj])) for jj in range(len(c_labels))]
        
        if self._chooseXrange:
            choose_xrange = self._chooseXrange
        else:
            choose_xrange = [0, self._maxTime]
        _fig_formatting_2D(xdata=x_data, ydata=y_data , xlab=self._xlab, ylab=self._ylab, choose_xrange=choose_xrange, 
                           choose_yrange=self._chooseYrange, fontsize=self._chooseFontSize, curvelab=c_labels, legend_loc=self._legend_loc, grid=True, 
                           legend_fontsize=self._legend_fontsize)
        
        self._show_computation_stop()
        
        
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
        M_1, M_2 = symbols('M_1 M_2')     
        eta_SV1 = Symbol('eta_'+str(self._stateVariable1))
        if self._stateVariable2:
            eta_SV2 = Symbol('eta_'+str(self._stateVariable2))
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
            solEQsys2ndOrdMom = linsolve(EQsys2ndOrdMom, [M_2(eta_SV1*eta_SV1), 
                                                             M_2(eta_SV2*eta_SV2), 
                                                             M_2(eta_SV3*eta_SV3), 
                                                             M_2(eta_SV1*eta_SV2), 
                                                             M_2(eta_SV1*eta_SV3), 
                                                             M_2(eta_SV2*eta_SV3)])
            if len(list(solEQsys2ndOrdMom)) == 0:
                return SOL_2ndOrdMomDict
            else:
                SOL_2ndOrderMom = list(solEQsys2ndOrdMom)[0]  # only one set of solutions (if any) in linear system of equations; hence index [0]
                SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV1)] = SOL_2ndOrderMom[0]
                SOL_2ndOrdMomDict[M_2(eta_SV2*eta_SV2)] = SOL_2ndOrderMom[1]
                SOL_2ndOrdMomDict[M_2(eta_SV3*eta_SV3)] = SOL_2ndOrderMom[2]
                SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV2)] = SOL_2ndOrderMom[3]
                SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV3)] = SOL_2ndOrderMom[4]
                SOL_2ndOrdMomDict[M_2(eta_SV2*eta_SV3)] = SOL_2ndOrderMom[5]
        
        elif self._stateVariable2:
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV1*eta_SV1)])
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV2*eta_SV2)])
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV1*eta_SV2)])
            solEQsys2ndOrdMom = linsolve(EQsys2ndOrdMom, [M_2(eta_SV1*eta_SV1), 
                                                             M_2(eta_SV2*eta_SV2), 
                                                             M_2(eta_SV1*eta_SV2)])
            
            if len(list(solEQsys2ndOrdMom)) == 0:
                return SOL_2ndOrdMomDict
            else:
                SOL_2ndOrderMom = list(solEQsys2ndOrdMom)[0]  # only one set of solutions (if any) in linear system of equations
                SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV1)] = SOL_2ndOrderMom[0]
                SOL_2ndOrdMomDict[M_2(eta_SV2*eta_SV2)] = SOL_2ndOrderMom[1]
                SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV2)] = SOL_2ndOrderMom[2]
            
        else:
            EQsys2ndOrdMom.append(EOM_2ndOrdMomDict[M_2(eta_SV1*eta_SV1)])
            solEQsys2ndOrdMom = linsolve(EQsys2ndOrdMom, [M_2(eta_SV1*eta_SV1)])
            if len(list(solEQsys2ndOrdMom)) == 0:
                return SOL_2ndOrdMomDict
            else:
                SOL_2ndOrderMom = list(solEQsys2ndOrdMom)[0]  # only one set of solutions (if any) in linear system of equations
                SOL_2ndOrdMomDict[M_2(eta_SV1*eta_SV1)] = SOL_2ndOrderMom[0]
            
        return SOL_2ndOrdMomDict

    
    def _build_bookmark(self, includeParams=True):
        if not self._silent:
            logStr = "bookmark = "
        else:
            logStr = ""
        
        logStr += "<modelName>." + self._generatingCommand + "("
        if "initialState" not in self._generatingKwargs.keys():
            initState_str = {latex(state): pop for state, pop in self._initialState.items() if not state in self._mumotModel._constantReactants}
            logStr += "initialState = " + str(initState_str) + ", "
        if "maxTime" not in self._generatingKwargs.keys():
            logStr += "maxTime = " + str(self._maxTime) + ", "
        if includeParams:
            logStr += self._get_bookmarks_params() + ", "
        if len(self._generatingKwargs) > 0:
            for key in self._generatingKwargs:
                if type(self._generatingKwargs[key]) == str:
                    logStr += key + " = " + "\'" + str(self._generatingKwargs[key]) + "\'" + ", "
                else:
                    logStr += key + " = " + str(self._generatingKwargs[key]) + ", "    
        logStr += "bookmark = False"
        logStr += ")"
        logStr = _greekPrependify(logStr)
        logStr = logStr.replace('\\', '\\\\')
        logStr = logStr.replace('\\\\\\\\', '\\\\')
        
        return logStr


class MuMoTfieldView(MuMoTview):
    """Field view on model.

    Specialised by :class:`MuMoTvectorView` and :class:`MuMoTstreamView`.

    """
    ## 1st state variable (x-dimension)
    _stateVariable1 = None
    ## 2nd state variable (y-dimension)
    _stateVariable2 = None
    ## 3rd state variable (z-dimension) 
    _stateVariable3 = None
    ## stores fixed points
    _FixedPoints = None
    ## stores 2nd Order moments of noise-noise correlations
    _SOL_2ndOrdMomDict = None
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
    ##x-label
    _xlab = None
    ## y-label
    _ylab = None
    ## z-label
    _zlab = None
    ## flag to run SSA simulations to compute noise ellipse
    _showSSANoise = None
    ## flag to show Noise
    _showNoise = None
    ## fontsize for axes labels
    _chooseFontSize = None
    ## displayed range for vertical axis
    _chooseXrange = None
    ## displayed range for horizontal axis
    _chooseYrange = None
    ## fixed points for logs 
    _realEQsol = None
    ## eigenvalues for logs 
    _EV = None
    ## eigenvectors for logs 
    _Evects = None
    ## random seed  (for computing SSA noise)
    _randomSeed = None
    ## simulation length (for computing SSA noise)
    _maxTime = None 
    ## reactants to display on the two axes
    #_finalViewAxes = None
    ## flag to plot proportions or full populations
    _plotProportions = None 
    ## number of runs to execute  (for computing SSA noise)
    _runs = None
    ## flag to set if the results from multimple runs must be aggregated or not (for computing SSA noise)  
    _aggregateResults = None
    
    
    def __init__(self, model, controller, fieldParams, SOL_2ndOrd, stateVariable1, stateVariable2, stateVariable3=None, figure=None, params=None, **kwargs):
        if model._systemSize is None and model._constantSystemSize == True:
            print("Cannot construct field-based plot until system size is set, using substitute()")
            return
        self._silent = kwargs.get('silent', False)
        super().__init__(model=model, controller=controller, figure=figure, params=params, **kwargs)
        
        with io.capture_output() as log:
        #if True:
            if 'fontsize' in kwargs:
                self._chooseFontSize = kwargs['fontsize']
            else:
                self._chooseFontSize = None
            self._showFixedPoints = kwargs.get('showFixedPoints', False)
            self._xlab = r'' + kwargs.get('xlab', r'$' + '\Phi_{' + _doubleUnderscorify(_greekPrependify(str(stateVariable1)))+'}$')
            self._ylab = r'' + kwargs.get('ylab', r'$' + '\Phi_{' + _doubleUnderscorify(_greekPrependify(str(stateVariable2)))+'}$')
            if stateVariable3:
                self._zlab = r'' + kwargs.get('zlab', r'$' + '\Phi_{' + _doubleUnderscorify(_greekPrependify(str(stateVariable3))) + '}$') 
            
            self._stateVariable1 = process_sympy(stateVariable1)
            self._stateVariable2 = process_sympy(stateVariable2)
            if stateVariable3 is not None:
                self._axes3d = True
                self._stateVariable3 = process_sympy(stateVariable3)
            _mask = {}
            
            self._SOL_2ndOrdMomDict = SOL_2ndOrd
            
            self._showNoise = kwargs.get('showNoise', False)
            
            if self._showNoise == True and self._SOL_2ndOrdMomDict is None and self._stateVariable3 is None:
                self._showSSANoise = True
            else:
                self._showSSANoise = False
            
            if stateVariable3 is None:    
                self._chooseXrange = kwargs.get('choose_xrange', None)
                self._chooseYrange = kwargs.get('choose_yrange', None)
            
            if self._controller is None:
                # storing all values of MA-specific parameters
                self._maxTime = fieldParams["maxTime"]
                self._randomSeed = fieldParams["randomSeed"]
#                 final_x = str(process_sympy(fieldParams.get("final_x", latex(sorted(self._mumotModel._getAllReactants()[0], key=str)[0]))))
#                 final_y = str(process_sympy(fieldParams.get("final_y", latex(sorted(self._mumotModel._getAllReactants()[0], key=str)[0]))))
#                 self._finalViewAxes = (final_x, final_y)
#                self._plotProportions = fieldParams["plotProportions"]
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
            self._randomSeed =  self._getWidgetParamValue('randomSeed', self._controller._widgetsExtraParams) # self._fixedParams['randomSeed'] if self._fixedParams.get('randomSeed') is not None else self._controller._widgetsExtraParams['randomSeed'].value
            #self._finalViewAxes = (self._getWidgetParamValue('final_x', self._controller._widgetsPlotOnly), self._getWidgetParamValue('final_y', self._controller._widgetsPlotOnly))
            #self._finalViewAxes = (self._fixedParams['final_x'] if self._fixedParams.get('final_x') is not None else self._controller._widgetsPlotOnly['final_x'].value, self._fixedParams['final_y'] if self._fixedParams.get('final_y') is not None else self._controller._widgetsPlotOnly['final_y'].value)
            #self._plotProportions = self._getWidgetParamValue('plotProportions', self._controller._widgetsPlotOnly) # self._fixedParams['plotProportions'] if self._fixedParams.get('plotProportions') is not None else self._controller._widgetsPlotOnly['plotProportions'].value
            self._maxTime = self._getWidgetParamValue('maxTime', self._controller._widgetsExtraParams) # self._fixedParams['maxTime'] if self._fixedParams.get('maxTime') is not None else self._controller._widgetsExtraParams['maxTime'].value
            self._runs = self._getWidgetParamValue('runs', self._controller._widgetsExtraParams) # self._fixedParams['runs'] if self._fixedParams.get('runs') is not None else self._controller._widgetsExtraParams['runs'].value
            self._aggregateResults = self._getWidgetParamValue('aggregateResults', self._controller._widgetsExtraParams) # self._fixedParams['aggregateResults'] if self._fixedParams.get('aggregateResults') is not None else self._controller._widgetsPlotOnly['aggregateResults'].value

    def _build_bookmark(self, includeParams=True): 
        logStr = "bookmark = " if not self._silent else ""
        logStr += "<modelName>." + self._generatingCommand + "('" + str(self._stateVariable1) + "', '" + str(self._stateVariable2) + "', "
        if self._stateVariable3 is not None:
            logStr += "'" + str(self._stateVariable3) + "', "
        # todo: plotting parameters are not kept, this could be solved with some work on the _generatingKwargs and should be made general for all views (similar to _get_bookmarks_params() )
#         if includeParams:
#             logStr += self._get_bookmarks_params() + ", "
#         if len(self._generatingKwargs) > 0:
#             for key in self._generatingKwargs:
#                 if key == 'xlab' or key == 'ylab' or key == 'zlab':
#                     logStr += key + " = '" + str(self._generatingKwargs[key]) + "', "
#                 else:
#                     logStr += key + " = " + str(self._generatingKwargs[key]) + ", "
        if includeParams:
            logStr += self._get_bookmarks_params()
            logStr += ", "
        logStr = logStr.replace('\\', '\\\\')
        logStr += "showNoise = " + str(self._showNoise)
        logStr += ", showFixedPoints = " + str(self._showFixedPoints)
        logStr += ", runs = " + str(self._runs)
        logStr += ", maxTime = " + str(self._maxTime)
        logStr += ", randomSeed = " + str(self._randomSeed)
#       todo: Following commented lines are ready to implement issue #95
#         if self._visualisationType == 'final':
#             # these loops are necessary to return the latex() format of the reactant 
#             for reactant in self._mumotModel._getAllReactants()[0]:
#                 if str(reactant) == self._finalViewAxes[0]:
#                     logStr += ", final_x = '" + latex(reactant).replace('\\', '\\\\') + "'"
#                     break
#             for reactant in self._mumotModel._getAllReactants()[0]:
#                 if str(reactant) == self._finalViewAxes[1]:
#                     logStr += ", final_y = '" + latex(reactant).replace('\\', '\\\\') + "'"
#                     break
        #logStr += ", plotProportions = " + str(self._plotProportions) todo: ready to implement issue #283
        logStr += ", aggregateResults = " + str(self._aggregateResults)
        logStr += ", silent = " + str(self._silent)
        logStr += ", bookmark = False"
        logStr += ")"
        logStr = _greekPrependify(logStr)
        logStr = logStr.replace('\\', '\\\\')
        logStr = logStr.replace('\\\\\\\\', '\\\\')
        
        return logStr
    
    
    def _plot_field(self):
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
        
        if self._stateVariable3 is None:
            if self._showFixedPoints == True or self._SOL_2ndOrdMomDict is not None or self._showSSANoise:
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
                    PhiSubDict = {}
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
                        if eigList[kk][key][0] > 1:
                            for jj in range(eigList[kk][key][0]):
                                EVsub.append(key.evalf())
                        else:
                            EVsub.append(key.evalf())
                            
                    EV.append(EVsub)
                    Evects.append(EvectsSub)
                    if self._mumotModel._constantSystemSize == True:
                        if (0 <= sympy.re(realEQsol[kk][self._stateVariable1]) <= 1 and 0 <= sympy.re(realEQsol[kk][self._stateVariable2]) <= 1):
                            EVplot.append(EVsub)
                            EvectsPlot.append(EvectsSub)
                    else:
                        EVplot.append(EVsub)
                        EvectsPlot.append(EvectsSub)
                #self._EV = EV
                #self._realEQsol = realEQsol
                #self._Evects = Evects
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
                vec1 = Matrix([[0], [1]])
                for nn in range(len(EvectsPlot)):
                    vec2 = EvectsPlot[nn][0]
                    vec2norm = vec2.norm()
                    vec2 = vec2/vec2norm
                    vec3 = EvectsPlot[nn][0]
                    vec3norm = vec3.norm()
                    vec3 = vec3/vec3norm
                    if vec2norm >= vec3norm:
                        angle_ell = sympy.acos(vec1.dot(vec2)/(vec1.norm()*vec2.norm())).evalf()
                    else:
                        angle_ell = sympy.acos(vec1.dot(vec3)/(vec1.norm()*vec3.norm())).evalf()
                    angle_ell = angle_ell.evalf()
                    projection_angle_list.append(angle_ell)
                    angle_ell_deg = 180*angle_ell/(sympy.pi).evalf()
                    angle_ell_list.append(round(angle_ell_deg, 5))
                projection_angle_list = [abs(projection_angle_list[kk]) if abs(projection_angle_list[kk]) <= sympy.N(sympy.pi/2) else sympy.N(sympy.pi)-abs(projection_angle_list[kk]) for kk in range(len(projection_angle_list))]
            
            if self._showFixedPoints == True or self._SOL_2ndOrdMomDict is not None or self._showSSANoise:
                if self._mumotModel._constantSystemSize == True:
                    FixedPoints = [[PhiSubList[kk][Phi_stateVar1] for kk in range(len(PhiSubList)) if (0 <= sympy.re(PhiSubList[kk][Phi_stateVar1]) <= 1 and 0 <= sympy.re(PhiSubList[kk][Phi_stateVar2]) <= 1)], 
                                 [PhiSubList[kk][Phi_stateVar2] for kk in range(len(PhiSubList)) if (0 <= sympy.re(PhiSubList[kk][Phi_stateVar1]) <= 1 and 0 <= sympy.re(PhiSubList[kk][Phi_stateVar2]) <= 1)]]
                    if self._SOL_2ndOrdMomDict:
                        Ell_width = [2.0*sympy.re(sympy.cos(sympy.N(sympy.pi/2)-projection_angle_list[kk])*sympy.sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar2**2)]/systemSize.subs(argDict)) + sympy.sin(sympy.N(sympy.pi/2)-projection_angle_list[kk])*sympy.sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar1**2)]/systemSize.subs(argDict))) for kk in range(len(SOL_2ndOrdMomDictList)) if (0 <= sympy.re(PhiSubList[kk][Phi_stateVar1]) <= 1 and 0 <= sympy.re(PhiSubList[kk][Phi_stateVar2]) <= 1)]
                        Ell_height = [2.0*sympy.re(sympy.cos(projection_angle_list[kk])*sympy.sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar2**2)]/systemSize.subs(argDict)) + sympy.sin(projection_angle_list[kk])*sympy.sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar1**2)]/systemSize.subs(argDict))) for kk in range(len(SOL_2ndOrdMomDictList)) if (0 <= sympy.re(PhiSubList[kk][Phi_stateVar1]) <= 1 and 0 <= sympy.re(PhiSubList[kk][Phi_stateVar2]) <= 1)]
                    #FixedPoints=[[realEQsol[kk][self._stateVariable1] for kk in range(len(realEQsol)) if (0 <= sympy.re(realEQsol[kk][self._stateVariable1]) <= 1 and 0 <= sympy.re(realEQsol[kk][self._stateVariable2]) <= 1)], 
                    #             [realEQsol[kk][self._stateVariable2] for kk in range(len(realEQsol)) if (0 <= sympy.re(realEQsol[kk][self._stateVariable1]) <= 1 and 0 <= sympy.re(realEQsol[kk][self._stateVariable2]) <= 1)]]
                else:
                    FixedPoints = [[PhiSubList[kk][Phi_stateVar1] for kk in range(len(PhiSubList))], 
                                 [PhiSubList[kk][Phi_stateVar2] for kk in range(len(PhiSubList))]]
                    if self._SOL_2ndOrdMomDict:
                        Ell_width = [2.0*sympy.re(sympy.cos(sympy.N(sympy.pi/2)-projection_angle_list[kk])*sympy.sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar2**2)]/systemSize.subs(argDict)) + sympy.sin(sympy.N(sympy.pi/2)-projection_angle_list[kk])*sympy.sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar1**2)]/systemSize.subs(argDict))) for kk in range(len(SOL_2ndOrdMomDictList))]
                        Ell_height = [2.0*sympy.re(sympy.cos(projection_angle_list[kk])*sympy.sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar2**2)]/systemSize.subs(argDict)) + sympy.sin(projection_angle_list[kk])*sympy.sqrt(SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar1**2)]/systemSize.subs(argDict))) for kk in range(len(SOL_2ndOrdMomDictList))]
                    #FixedPoints=[[realEQsol[kk][self._stateVariable1] for kk in range(len(realEQsol))], 
                    #             [realEQsol[kk][self._stateVariable2] for kk in range(len(realEQsol))]]
                    #Ell_width = [SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar1**2)] for kk in range(len(SOL_2ndOrdMomDictList))]
                    #Ell_height = [SOL_2ndOrdMomDictList[kk][M_2(eta_stateVar2**2)] for kk in range(len(SOL_2ndOrdMomDictList))] 
               
                FixedPoints.append(EVplot)
            else:
                FixedPoints = None
            
            self._FixedPoints = FixedPoints
            
            skipEllipse=False    
            if self._SOL_2ndOrdMomDict:   
                for kk in range(len(Ell_width)):
                    # remark: nan is imported from sympy
                    if Ell_width[kk] == nan or Ell_width[kk] == 0 or Ell_height[kk] == nan or Ell_height[kk] == 0:
                        skipEllipse=True
                        self._showErrorMessage('Noise could not be calculated analytically. ')
                        break
            
            if self._SOL_2ndOrdMomDict and skipEllipse==False:
                # swap width and height of ellipse if width > height
                for kk in range(len(Ell_width)):
                    #print(Ell_width[kk])
                    #print(type(Ell_width[kk]))
                    #print(Ell_height[kk])
                    #print(type(Ell_height[kk]))
                    ell_width_temp = Ell_width[kk]
                    ell_height_temp = Ell_height[kk]
                    if ell_width_temp > ell_height_temp:
                        Ell_height[kk] = ell_width_temp
                        Ell_width[kk] = ell_height_temp
                        
                ells = [mpatch.Ellipse(xy=[self._FixedPoints[0][nn], self._FixedPoints[1][nn]], width=Ell_width[nn]/systemSize.subs(argDict), height=Ell_height[nn]/systemSize.subs(argDict), angle=round(angle_ell_list[nn], 5)) for nn in range(len(self._FixedPoints[0]))]
                ax = plt.gca()
                for kk in range(len(ells)):
                    ax.add_artist(ells[kk])
                    ells[kk].set_alpha(0.5)
                    if sympy.re(EVplot[kk][0]) < 0 and sympy.re(EVplot[kk][1]) < 0:
                        Fcolor = LINE_COLOR_LIST[1]
                    elif sympy.re(EVplot[kk][0]) > 0 and sympy.re(EVplot[kk][1]) > 0:
                        Fcolor = LINE_COLOR_LIST[2]
                    else:
                        Fcolor = LINE_COLOR_LIST[0]
                    ells[kk].set_facecolor(Fcolor)
                #self._ells = ells
            else:
                if self._showNoise==True:
                    self._showSSANoise=True   
                
            if self._showSSANoise:
    #             print(FixedPoints)
    #             print(self._stateVariable1)
    #             print(self._stateVariable2)
    #             print(realEQsol)
                skipList=[]
                for kk in range(len(realEQsol)):
                    #print("printing ellipse for point " + str(realEQsol[kk]) )
                    # skip values out of range [0,1] and unstable equilibria
                    skip = False
                    for p in realEQsol[kk].values():
                        #if p < 0 or p > 1:
                        if p < 0:
                            skip = True
                            #print("Skipping for out range")
                            break
                        for eigenV in EV[kk]:
                            #skip if no stable fixed points detected
                            if sympy.re(eigenV) >= 0:
                                skip = True
                                #print("Skipping for positive eigenvalue")
                                break
                        if skip: break
                    if skip: 
                        skipList.append('skip')
                        continue
                    # generate proper init reactant list
                    initState = copy.deepcopy(realEQsol[kk])
                    for reactant in self._mumotModel._getAllReactants()[0]:
                        if reactant not in initState.keys():
                            #initState[reactant] = 1 - sum(initState.values())
                            iniSum = 0
                            for val in initState.values(): iniSum += np.real(val)
                            initState[reactant] = 1 - iniSum
                    #print(initState)
                    #print("Using params: " + str(self._get_params()))
                    getParams = []
                    for set_item in self._get_params():
                        getParams.append((_greekPrependify(set_item[0].replace('{', '').replace('}', '')), set_item[1]))
                    SSAView = MuMoTSSAView(self._mumotModel, None,
                                     params=getParams,
                                     SSParams={'maxTime': self._maxTime, 'runs': self._runs, 'realtimePlot': False, 'plotProportions': True, 'aggregateResults': self._aggregateResults, 'visualisationType': 'final',
                                                 'final_x': _greekPrependify(latex(self._stateVariable1)), 'final_y': _greekPrependify(latex(self._stateVariable2)), 
                                                 'initialState': initState, 'randomSeed': self._randomSeed}, silent=True)
                    #print(SSAView._printStandaloneViewCmd())
                    SSAView._figure = self._figure
                    SSAView._computeAndPlotSimulation()
                
                if len(realEQsol) == len(skipList):
                    self._showErrorMessage('No stable fixed points detected. Noise could not be calculated numerically.')
                                
        else:
            if self._showNoise == True:
                print('Please note: Currently \'showNoise\' only available for 2D stream and vector plots.')
            if self._showFixedPoints == True:
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
                    if self._mumotModel._constantSystemSize == True:
                        if (0 <= sympy.re(realEQsol[kk][self._stateVariable1]) <= 1 and 0 <= sympy.re(realEQsol[kk][self._stateVariable2]) <= 1 and 0 <= sympy.re(realEQsol[kk][self._stateVariable3]) <= 1):
                            EVplot.append(EVsub)
                    else:
                        EVplot.append(EVsub)
                    
                if self._mumotModel._constantSystemSize == True:
                    FixedPoints = [[realEQsol[kk][self._stateVariable1] for kk in range(len(realEQsol)) if (0 <= sympy.re(realEQsol[kk][self._stateVariable1]) <= 1) and (0 <= sympy.re(realEQsol[kk][self._stateVariable2]) <= 1) and (0 <= sympy.re(realEQsol[kk][self._stateVariable3]) <= 1)], 
                                 [realEQsol[kk][self._stateVariable2] for kk in range(len(realEQsol)) if (0 <= sympy.re(realEQsol[kk][self._stateVariable1]) <= 1) and (0 <= sympy.re(realEQsol[kk][self._stateVariable2]) <= 1) and (0 <= sympy.re(realEQsol[kk][self._stateVariable3]) <= 1)],
                                 [realEQsol[kk][self._stateVariable3] for kk in range(len(realEQsol)) if (0 <= sympy.re(realEQsol[kk][self._stateVariable1]) <= 1) and (0 <= sympy.re(realEQsol[kk][self._stateVariable2]) <= 1) and (0 <= sympy.re(realEQsol[kk][self._stateVariable3]) <= 1)]]
                else:
                    FixedPoints = [[realEQsol[kk][self._stateVariable1] for kk in range(len(realEQsol))], 
                                 [realEQsol[kk][self._stateVariable2] for kk in range(len(realEQsol))],
                                 [realEQsol[kk][self._stateVariable3] for kk in range(len(realEQsol))]]
                FixedPoints.append(EVplot)
#                
#                 with io.capture_output() as log:
#                     for kk in range(len(realEQsol)):
#                         print('Fixed point'+str(kk+1)+':', realEQsol[kk], 'with eigenvalues:', str(EV[kk]))
#                 self._logs.append(log)
#                 
            else:
                FixedPoints = None
        
            self._FixedPoints = FixedPoints
            
        self._realEQsol = realEQsol
        self._EV = EV
        self._Evects = Evects

        self._show_computation_stop()

    ## helper for _get_field_2d() and _get_field_3d()
    def _get_field(self):
        plotLimits = self._getPlotLimits()
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
#         argNamesSymb = list(map(Symbol, paramNames))
#         argDict = dict(zip(argNamesSymb, paramValues))
        argDict = self._get_argDict()
        funcs = self._mumotModel._getFuncs()
        
        return (funcs, argDict, plotLimits)

    ## get 2-dimensional field for plotting
    def _get_field2d(self, kind, meshPoints, plotLimits=1):
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
#                self._speed = np.ones(self._X.shape, dtype=float)
                self._speed = None
            if self._mumotModel._constantSystemSize:
                self._Xdot = np.ma.array(self._Xdot, mask=mask)
                self._Ydot = np.ma.array(self._Ydot, mask=mask)        
        #if len(self._logs) > 0:
        #    self._logs.insert(0, log)
        #else:
        self._logs.append(log)

    ## get 3-dimensional field for plotting        
    def _get_field3d(self, kind, meshPoints, plotLimits=1):
        with io.capture_output() as log:
            self._log(kind)
            (funcs, argDict, plotLimits) = self._get_field()
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
            self._Xdot = funcs[self._stateVariable1](*self._mumotModel._getArgTuple3d(argDict, self._stateVariable1, self._stateVariable2, self._stateVariable3, self._X, self._Y, self._Z))
            self._Ydot = funcs[self._stateVariable2](*self._mumotModel._getArgTuple3d(argDict, self._stateVariable1, self._stateVariable2, self._stateVariable3, self._X, self._Y, self._Z))
            self._Zdot = funcs[self._stateVariable3](*self._mumotModel._getArgTuple3d(argDict, self._stateVariable1, self._stateVariable2, self._stateVariable3, self._X, self._Y, self._Z))
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
        #if len(self._logs) > 0:
        #    self._logs.insert(0, log)
        #else:
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
                    print('Fixed point'+str(kk+1)+': ', realEQsolStr, 'with eigenvalues: ', evalString)
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
                    print('Fixed point'+str(kk+1)+': ', realEQsolStr, 'with eigenvalues: ', evalString,
                          'and eigenvectors: ', evecString)
        

class MuMoTvectorView(MuMoTfieldView):
    """Vector plot view on model."""

    ## dictionary containing the solutions of the second order noise moments in the stationary state
    _SOL_2ndOrdMomDict = None
    ## set of all reactants
    _checkReactants = None
    ## set of all constant reactants to get intersection with _checkReactants
    _checkConstReactants = None
    
    def __init__(self, model, controller, fieldParams, SOL_2ndOrd, stateVariable1, stateVariable2, stateVariable3=None, figure=None, params=None, **kwargs):
        #if model._systemSize is None and model._constantSystemSize == True:
        #    self._showErrorMessage("Cannot construct field-based plot until system size is set, using substitute()")
        #    return
        #if self._SOL_2ndOrdMomDict is None:
        #    self._showErrorMessage('Noise in the system could not be calculated: \'showNoise\' automatically disabled.')
        super().__init__(model=model, controller=controller, fieldParams=fieldParams, SOL_2ndOrd=SOL_2ndOrd, stateVariable1=stateVariable1, stateVariable2=stateVariable2, stateVariable3=stateVariable3, figure=figure, params=params, **kwargs)
        self._generatingCommand = "vector"

    def _plot_field(self, _=None):
        
        super()._plot_field()                   
        
        if self._stateVariable3 is None:   
            self._get_field2d("2d vector plot", 10)  # @todo: allow user to set mesh points with keyword
            fig_vector = plt.quiver(self._X, self._Y, self._Xdot, self._Ydot, units='width', color='black')  # @todo: define colormap by user keyword
            
            if self._mumotModel._constantSystemSize == True:
                plt.fill_between([0, 1], [1, 0], [1, 1], color='grey', alpha='0.25')
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
                #plt.xlim(0,self._X.max())
                #plt.ylim(0,self._Y.max())
            
            _fig_formatting_2D(figure=fig_vector, xlab=self._xlab, specialPoints=self._FixedPoints, showFixedPoints=self._showFixedPoints, ax_reformat=False, curve_replot=False,
                   ylab=self._ylab, fontsize=self._chooseFontSize, aspectRatioEqual=True, choose_xrange=choose_xrange, choose_yrange=choose_yrange)
            
        else:
            self._get_field3d("3d vector plot", 10)
            ax = self._figure.gca(projection='3d')
            fig_vec3d = ax.quiver(self._X, self._Y, self._Z, self._Xdot, self._Ydot, self._Zdot, length=0.01, color='black')  # @todo: define colormap by user keyword; normalise off maximum value in self._speed, and meshpoints?
            
            _fig_formatting_3D(figure=fig_vec3d, xlab=self._xlab, ylab=self._ylab, zlab=self._zlab, specialPoints=self._FixedPoints,
                               showFixedPoints=self._showFixedPoints, ax_reformat=True, showPlane=self._mumotModel._constantSystemSize, fontsize=self._chooseFontSize)
        
        with io.capture_output() as log:
            self._appendFixedPointsToLogs(self._realEQsol, self._EV, self._Evects)
        self._logs.append(log)


class MuMoTstreamView(MuMoTfieldView):
    """Stream plot view on model."""

    ## dictionary containing the solutions of the second order noise moments in the stationary state
    _SOL_2ndOrdMomDict = None
    ## set of all reactants
    _checkReactants = None
    ## set of all constant reactants to get intersection with _checkReactants
    _checkConstReactants = None
    
    def __init__(self, model, controller, fieldParams, SOL_2ndOrd, stateVariable1, stateVariable2, stateVariable3=None, figure=None, params=None, **kwargs):
        #if model._systemSize is None and model._constantSystemSize == True:
        #    self._showErrorMessage("Cannot construct field-based plot until system size is set, using substitute()")
        #    return
        #if self._SOL_2ndOrdMomDict is None:
        #    self._showErrorMessage('Noise in the system could not be calculated: \'showNoise\' automatically disabled.')

        self._checkReactants = model._reactants
        if model._constantReactants:
            self._checkConstReactants = model._constantReactants
        else:
            self._checkConstReactants = None
        super().__init__(model=model, controller=controller, fieldParams=fieldParams, SOL_2ndOrd=SOL_2ndOrd, stateVariable1=stateVariable1, stateVariable2=stateVariable2, stateVariable3=stateVariable3, figure=figure, params=params, **kwargs)
        self._generatingCommand = "stream"

    def _plot_field(self, _=None):
        
        # check number of time-dependent reactants
        checkReactants = copy.deepcopy(self._checkReactants)
        if self._checkConstReactants:
            checkConstReactants = copy.deepcopy(self._checkConstReactants)
            for reactant in checkReactants:
                if reactant in checkConstReactants:
                    checkReactants.remove(reactant)
        if len(checkReactants) != 2:
            self._showErrorMessage("Not implemented: This feature is available only for systems with exactly 2 time-dependent reactants!")
        
        
        super()._plot_field()                   
        
        if self._stateVariable3 is None:
            self._get_field2d("2d stream plot", 100)  # @todo: allow user to set mesh points with keyword
            
            if self._speed is not None:
                with io.capture_output() as log: # catch warnings from streamplot
                    fig_stream = plt.streamplot(self._X, self._Y, self._Xdot, self._Ydot, color=self._speed, cmap='gray')  # @todo: define colormap by user keyword
                self._logs.append(log)                    
            else:
                fig_stream = plt.streamplot(self._X, self._Y, self._Xdot, self._Ydot, color='k')  # @todo: define colormap by user keyword
            
            if self._mumotModel._constantSystemSize == True:
                plt.fill_between([0, 1], [1, 0], [1, 1], color='grey', alpha='0.25')
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
                #plt.xlim(0,self._X.max())
                #plt.ylim(0,self._Y.max())
            
            _fig_formatting_2D(figure=fig_stream, xlab=self._xlab, specialPoints=self._FixedPoints, showFixedPoints=self._showFixedPoints, 
                               ax_reformat=False, curve_replot=False, ylab=self._ylab, fontsize=self._chooseFontSize, aspectRatioEqual=True,
                               choose_xrange=choose_xrange, choose_yrange=choose_yrange)
            
            with io.capture_output() as log:
                self._appendFixedPointsToLogs(self._realEQsol, self._EV, self._Evects)
            self._logs.append(log)
                
        else:
            print('3d stream plot not yet implemented.')
        

class MuMoTbifurcationView(MuMoTview):
    """Bifurcation view on model."""
    
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
    ## state variable 1 for logs output
    _stateVarBif1Print = None
    ## state variable 2 for logs output
    _stateVarBif2Print = None
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
    
    ## Parameters for controller specific to this MuMoTbifurcationView
    _BfcParams = None
    ## bifurcation parameter prepared for use in _get_argDict function in MuMoTView
    _bifurcationParameter_for_get_argDict = None
    ## bifurcation parameter to be used in bookmark function
    _bifurcationParameter_for_bookmark = None
    ## the system state at the start of the simulation (timestep zero)
    _initialState = None
    ## list of state variables
    _stateVariableList = None
    
    ## list of symbols protected in PyDSTool
    _pydsProtected = ['gamma', 'Gamma']
    ## bifurcation parameter symbol passsed to PyDSTool
    _bifurcationParameterPyDS = None
    
    def _constructorSpecificParams(self, _):
        if self._controller is not None:
            self._generatingCommand = "bifurcation"
    
    def __init__(self, model, controller, BfcParams, bifurcationParameter, stateVarExpr1, stateVarExpr2=None, 
                 figure=None, params=None, **kwargs):
        
        self._silent = kwargs.get('silent', False)
        
        self._bifurcationParameter_for_get_argDict = str(process_sympy(bifurcationParameter))
        #self._bifurcationParameter_for_bookmark = _greekPrependify(_doubleUnderscorify(self._bifurcationParameter_for_get_argDict))
        self._bifurcationParameter_for_bookmark = bifurcationParameter
        
        super().__init__(model=model, controller=controller, figure=figure, params=params, **kwargs)
        
        self._chooseFontSize = kwargs.get('fontsize', None)
        if '-' in str(stateVarExpr1):
            self._LabelY = kwargs.get('ylab', r'$' + '\Phi_{' + str(stateVarExpr1)[:str(stateVarExpr1).index('-')].replace('\\\\', '\\') + '}' + '-' + '\Phi_{' + str(stateVarExpr1)[str(stateVarExpr1).index('-')+1:].replace('\\\\', '\\') + '}' + '$') 
        elif '+' in str(stateVarExpr1):
            self._LabelY = kwargs.get('ylab', r'$' + '\Phi_{' + str(stateVarExpr1)[:str(stateVarExpr1).index('-')].replace('\\\\', '\\') + '}' + '+' + '\Phi_{' + str(stateVarExpr1)[str(stateVarExpr1).index('-')+1:].replace('\\\\', '\\') + '}' + '$') 
        else:
            self._LabelY = kwargs.get('ylab', r'$' + '\Phi_{' + str(stateVarExpr1).replace('\\\\', '\\') + '}$') 
        #self._LabelY =  kwargs.get('ylab', r'$' + stateVarExpr1 +'$') 
        #self._LabelX = kwargs.get('xlab', r'$' + _doubleUnderscorify(_greekPrependify(bifurcationParameter)) +'$')
        self._LabelX = kwargs.get('xlab', r'$' + bifurcationParameter + '$')
        self._chooseXrange = kwargs.get('choose_xrange', None)
        self._chooseYrange = kwargs.get('choose_yrange', None)
        
        self._MaxNumPoints = kwargs.get('contMaxNumPoints', 100)
        
        self._bifurcationParameter = self._pydstoolify(bifurcationParameter)
        replBifParam = {}
        if self._bifurcationParameter in self._pydsProtected:
            self._bifurcationParameterPyDS = 'A'+self._bifurcationParameter
            replBifParam[self._bifurcationParameter] = self._bifurcationParameterPyDS
        else:
            self._bifurcationParameterPyDS = self._bifurcationParameter
        
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
        
        #print(self._stateVarBif1) 
        #print(self._stateVarBif2)
        
        self._BfcParams = BfcParams            

        if self._controller is None:
            # storing the initial state
            self._initialState = {}
            for state, pop in BfcParams["initialState"].items():
                if isinstance(state, str):
                    self._initialState[process_sympy(state)] = pop  # convert string into SymPy symbol
                else:
                    self._initialState[state] = pop

            # storing all values of Bfc-specific parameters
            self._initBifParam = BfcParams["initBifParam"]
        
        else:
            # storing the initial state
            self._initialState = {}
            for state, pop in BfcParams["initialState"][0].items():
                if isinstance(state, str):
                    self._initialState[process_sympy(state)] = pop[0]  # convert string into SymPy symbol
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
       
        
        #self._logs.append(log)
         
        self._pyDSmodel = dst.args(name='MuMoT Model' + str(id(self)))
        varspecs = {}
        stateVariableList = []
        replaceSV = {}
        for reactant in self._mumotModel._reactants:
            if reactant not in self._mumotModel._constantReactants:
                stateVariableList.append(reactant)
                reactantString = self._pydstoolify(reactant)
                if reactantString[0].islower() or reactantString in self._pydsProtected:
                    replaceSV[reactantString] = 'A'+reactantString
                    varspecs['A'+reactantString] = self._pydstoolify(self._mumotModel._equations[reactant])
                else:
                    varspecs[reactantString] = self._pydstoolify(self._mumotModel._equations[reactant])
        
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
                if self._stateVarBif1 == self._pydstoolify(self._stateVariable1):
                    self._stateVarBif2 = self._pydstoolify(self._stateVariable2)
                elif self._stateVarBif1 == self._pydstoolify(self._stateVariable2):
                    self._stateVarBif2 = self._pydstoolify(self._stateVariable1)
                self._stateVarBif2Print = self._stateVarBif2
                if self._stateVarBif2[0].islower() or self._stateVarBif2 in self._pydsProtected:
                    self._stateVarBif2 = 'A'+self._stateVarBif2
        else:
            self._stateVarBif2Print = self._stateVarBif2
            if self._stateVarBif2[0].islower() or self._stateVarBif2 in self._pydsProtected:
                self._stateVarBif2 = 'A'+self._stateVarBif2
        self._stateVarBif1Print = self._stateVarBif1
        if self._stateVarBif1[0].islower() or self._stateVarBif1 in self._pydsProtected:
            self._stateVarBif1 = 'A'+self._stateVarBif1
        
        if not self._silent:
            self._plot_bifurcation()     

    def _plot_bifurcation(self, _=None):
        self._show_computation_start()
        
        self._initFigure()
        self._update_params()
        
        with io.capture_output() as log:
            self._log("bifurcation plot")
            if self._stateVariable2:
                print('State variables are: ', self._stateVarBif1Print, 'and ', self._stateVarBif2Print, '.')
            else:
                print('State variable is: ', self._stateVarBif1Print, '.')
            print('The bifurcation parameter chosen is: ', self._bifurcationParameter, '.')
        self._logs.append(log)
         
        argDict = self._get_argDict()
        paramDict = {}
        replaceRates = {}
        for arg in argDict:
            if arg in self._mumotModel._rates or arg in self._mumotModel._constantReactants or arg == self._mumotModel._systemSize:
                if self._pydstoolify(arg) in self._pydsProtected:
                    paramDict['A'+self._pydstoolify(arg)] = argDict[arg]
                    if self._pydstoolify(arg) != self._bifurcationParameter:
                        replaceRates[self._pydstoolify(arg)] = 'A'+self._pydstoolify(arg)
                else:
                    paramDict[self._pydstoolify(arg)] = argDict[arg]
        
        for key, equation in self._pyDSmodel.varspecs.items():
            for replKey, replVal in replaceRates.items():
                equationNew = equation.replace(replKey, replVal)
                self._pyDSmodel.varspecs[key] = equationNew
        
        with io.capture_output() as log:
        
            self._pyDSmodel.pars = paramDict 
            
            XDATA = []  # list of arrays containing the bifurcation-parameter data for bifurcation diagram data 
            YDATA = []  # list of arrays containing the state variable data (either one variable, or the sum or difference of the two SVs) for bifurcation diagram data
            
            initDictList = []
            self._pyDSmodel_ics = {}
            for inState in self._initialState:
                if inState in self._stateVariableList: 
                    self._pyDSmodel_ics[inState] = self._initialState[inState]
            
            #print(self._pyDSmodel_ics
            #for ic in self._pyDSmodel_ics:
            #    if 'Phi0' in self._pydstoolify(ic):
            #        self._pyDSmodel_ics[self._pydstoolify(ic)[self._pydstoolify(ic).index('0')+1:]] = self._pyDSmodel_ics.pop(ic)  #{'A': 0.1, 'B': 0.9 }  
            
            if len(self._stateVariableList) == 1:    
                realEQsol, eigList = self._get_fixedPoints1d()
            elif len(self._stateVariableList) == 2:    
                realEQsol, eigList = self._get_fixedPoints2d()
                
            if realEQsol != [] and realEQsol is not None:
                for kk in range(len(realEQsol)):
                    if all(sympy.sign(sympy.re(lam)) < 0 for lam in eigList[kk]) == True:
                        initDictList.append(realEQsol[kk])
                #self._showErrorMessage('Stationary state(s) detected and continuated. Initial conditions for state variables specified on sliders in Advanced options tab were not used. (Those are only used in case the calculation of fixed points fails.) ')
                print(len(initDictList), 'stable steady state(s) detected and continuated. Initial conditions for state variables specified on sliders in Advanced options tab were not used. Those are only used in case the calculation of fixed points fails.')
            else:
                initDictList.append(self._pyDSmodel_ics)
                #self._showErrorMessage('Stationary states could not be calculated; used initial conditions specified on sliders in Advanced options tab instead. This means only one branch was attempted to be continuated and the starting point might not have been a stationary state. ')
                print('Stationary states could not be calculated; used initial conditions specified on sliders in Advanced options tab instead: ', self._pyDSmodel_ics, '. This means only one branch was continuated and the starting point might not have been a stationary state.')   
            
            specialPoints = []  # list of special points: LP and BP
            sPoints_X = []  # bifurcation parameter
            sPoints_Y = []  # stateVarBif1
            sPoints_Labels = []
            EIGENVALUES = []
            sPoints_Z = []  # stateVarBif2
            k_iter_BPlabel = 0
            k_iter_LPlabel = 0
            #print(initDictList)
            
            for nn in range(len(initDictList)):
                for key in initDictList[nn]:
                    old_key = key
                    new_key = self._pydstoolify(key)
                    if new_key[0].islower() or new_key in self._pydsProtected:
                        new_key = 'A'+new_key
                    initDictList[nn][new_key] = initDictList[nn].pop(old_key)
                    
                #self._pyDSmodel.ics = initDictList[nn]
                pyDSode = dst.Generator.Vode_ODEsystem(self._pyDSmodel)
                pyDSode.set(ics=initDictList[nn])
                #pyDSode.set(pars = self._getBifParInitCondFromSlider()) 
                pyDSode.set(pars={self._bifurcationParameterPyDS: self._initBifParam})   
                 
                #print(self._getBifParInitCondFromSlider())  
                pyDScont = dst.ContClass(pyDSode)
                EQ_iter = 1+nn
                k_iter_BP = 1
                k_iter_LP = 1
                pyDScontArgs = dst.args(name='EQ'+str(EQ_iter), type='EP-C')     # 'EP-C' stands for Equilibrium Point Curve. The branch will be labelled with the string aftr name='name'.
                pyDScontArgs.freepars = [self._bifurcationParameterPyDS]   # control parameter(s) (should be among those specified in self._pyDSmodel.pars)
                pyDScontArgs.MaxNumPoints = self._MaxNumPoints    # The following 3 parameters should work for most cases, as there should be a step-size adaption within PyDSTool.
                pyDScontArgs.MaxStepSize = 1e-1
                pyDScontArgs.MinStepSize = 1e-5
                pyDScontArgs.StepSize = 2e-3
                pyDScontArgs.LocBifPoints = ['LP', 'BP']       # 'Limit Points' and 'Branch Points may be detected'
                pyDScontArgs.SaveEigen = True            # to tell unstable from stable branches
                
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
                    self._show_computation_stop()
                    self._showErrorMessage('Division by zero<br>')
                     
                #pyDScont['EQ'+str(EQ_iter)].info()
                if self._stateVarBif2 is not None:
                    try:
                        XDATA.append(pyDScont['EQ'+str(EQ_iter)].sol[self._bifurcationParameterPyDS])
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
                            if (round(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('LP'+str(k_iter_LP))[self._bifurcationParameterPyDS], 4) not in [round(kk , 4) for kk in sPoints_X] 
                                and round(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('LP'+str(k_iter_LP))[self._stateVarBif1], 4) not in [round(kk , 4) for kk in sPoints_Y]
                                and round(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('LP'+str(k_iter_LP))[self._stateVarBif2], 4) not in [round(kk , 4) for kk in sPoints_Z]):
                                sPoints_X.append(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('LP'+str(k_iter_LP))[self._bifurcationParameterPyDS])
                                sPoints_Y.append(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('LP'+str(k_iter_LP))[self._stateVarBif1])
                                sPoints_Z.append(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('LP'+str(k_iter_LP))[self._stateVarBif2])
                                k_iter_LPlabel += 1
                                sPoints_Labels.append('LP'+str(k_iter_LPlabel))
                            k_iter_LP += 1
                                               
                        k_iter_BPlabel_previous = k_iter_BPlabel
                        while pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('BP'+str(k_iter_BP)):
                            if (round(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('BP'+str(k_iter_BP))[self._bifurcationParameterPyDS], 4) not in [round(kk , 4) for kk in sPoints_X] 
                                and round(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('BP'+str(k_iter_BP))[self._stateVarBif1], 4) not in [round(kk , 4) for kk in sPoints_Y]
                                and round(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('BP'+str(k_iter_BP))[self._stateVarBif2], 4) not in [round(kk , 4) for kk in sPoints_Z]):
                                sPoints_X.append(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('BP'+str(k_iter_BP))[self._bifurcationParameterPyDS])
                                sPoints_Y.append(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('BP'+str(k_iter_BP))[self._stateVarBif1])
                                sPoints_Z.append(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('BP'+str(k_iter_BP))[self._stateVarBif2])
                                k_iter_BPlabel += 1
                                sPoints_Labels.append('BP'+str(k_iter_BPlabel))
                            k_iter_BP += 1
                        for jj in range(1, k_iter_BP):
                            if 'BP'+str(jj+k_iter_BPlabel_previous) in sPoints_Labels:
                                EQ_iter_BP = jj
                                #print(EQ_iter_BP)
                                k_iter_next = 1
                                pyDScontArgs = dst.args(name='EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP), type='EP-C')     # 'EP-C' stands for Equilibrium Point Curve. The branch will be labelled with the string aftr name='name'.
                                pyDScontArgs.freepars = [self._bifurcationParameterPyDS]   # control parameter(s) (should be among those specified in self._pyDSmodel.pars)
                                pyDScontArgs.MaxNumPoints = self._MaxNumPoints    # The following 3 parameters should work for most cases, as there should be a step-size adaption within PyDSTool.
                                pyDScontArgs.MaxStepSize = 1e-1
                                pyDScontArgs.MinStepSize = 1e-5
                                pyDScontArgs.StepSize = 5e-3
                                pyDScontArgs.LocBifPoints = ['LP', 'BP']        # 'Limit Points' and 'Branch Points may be detected'
                                pyDScontArgs.SaveEigen = True             # to tell unstable from stable branches
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
                                    self._show_computation_stop()
                                    self._showErrorMessage('Division by zero<br>')
                                
                                XDATA.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].sol[self._bifurcationParameterPyDS])
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
                                    if (round(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('BP'+str(k_iter_next))[self._bifurcationParameterPyDS], 4) not in [round(kk , 4) for kk in sPoints_X] 
                                        and round(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('BP'+str(k_iter_next))[self._stateVarBif1], 4) not in [round(kk , 4) for kk in sPoints_Y]
                                        and round(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('BP'+str(k_iter_next))[self._stateVarBif2], 4) not in [round(kk , 4) for kk in sPoints_Z]):    
                                        sPoints_X.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('BP'+str(k_iter_next))[self._bifurcationParameterPyDS])
                                        sPoints_Y.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('BP'+str(k_iter_next))[self._stateVarBif1])
                                        sPoints_Z.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('BP'+str(k_iter_next))[self._stateVarBif2])
                                        sPoints_Labels.append('EQ_BP_BP'+str(k_iter_next))
                                    k_iter_next += 1
                                k_iter_next = 1
                                while pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('LP'+str(k_iter_next)):
                                    if (round(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('LP'+str(k_iter_next))[self._bifurcationParameterPyDS], 4) not in [round(kk , 4) for kk in sPoints_X] 
                                        and round(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('LP'+str(k_iter_next))[self._stateVarBif1], 4) not in [round(kk , 4) for kk in sPoints_Y]
                                        and round(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('LP'+str(k_iter_next))[self._stateVarBif2], 4) not in [round(kk , 4) for kk in sPoints_Z]):
                                        sPoints_X.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('LP'+str(k_iter_next))[self._bifurcationParameterPyDS])
                                        sPoints_Y.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('LP'+str(k_iter_next))[self._stateVarBif1])
                                        sPoints_Z.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('LP'+str(k_iter_next))[self._stateVarBif2])
                                        sPoints_Labels.append('EQ_BP_LP'+str(k_iter_next))
                                    k_iter_next += 1
    
                    except TypeError:
                        self._show_computation_stop()
                        print('Continuation failed; try with different parameters - use sliders. If that does not work, try changing maximum number of continuation points using the keyword contMaxNumPoints. If not set, default value is contMaxNumPoints=100.')       
                
                # bifurcation routine fr 1D system
                else:
                    try:
                        XDATA.append(pyDScont['EQ'+str(EQ_iter)].sol[self._bifurcationParameterPyDS])
                        YDATA.append(pyDScont['EQ'+str(EQ_iter)].sol[self._stateVarBif1])
                        
                        EIGENVALUES.append(np.array([pyDScont['EQ'+str(EQ_iter)].sol[kk].labels['EP']['data'].evals 
                                                     for kk in range(len(pyDScont['EQ'+str(EQ_iter)].sol[self._stateVarBif1]))]))
                        
                        while pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('LP'+str(k_iter_LP)):
                            if (round(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('LP'+str(k_iter_LP))[self._bifurcationParameterPyDS], 4) not in [round(kk , 4) for kk in sPoints_X] 
                                and round(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('LP'+str(k_iter_LP))[self._stateVarBif1], 4) not in [round(kk , 4) for kk in sPoints_Y]):
                                sPoints_X.append(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('LP'+str(k_iter_LP))[self._bifurcationParameterPyDS])
                                sPoints_Y.append(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('LP'+str(k_iter_LP))[self._stateVarBif1])
                                k_iter_LPlabel += 1
                                sPoints_Labels.append('LP'+str(k_iter_LPlabel))
                            k_iter_LP += 1
                        
                        
                        k_iter_BPlabel_previous = k_iter_BPlabel
                        while pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('BP'+str(k_iter_BP)):
                            if (round(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('BP'+str(k_iter_BP))[self._bifurcationParameterPyDS], 4) not in [round(kk , 4) for kk in sPoints_X] 
                                and round(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('BP'+str(k_iter_BP))[self._stateVarBif1], 4) not in [round(kk , 4) for kk in sPoints_Y]):
                                sPoints_X.append(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('BP'+str(k_iter_BP))[self._bifurcationParameterPyDS])
                                sPoints_Y.append(pyDScont['EQ'+str(EQ_iter)].getSpecialPoint('BP'+str(k_iter_BP))[self._stateVarBif1])
                                k_iter_BPlabel += 1
                                sPoints_Labels.append('BP'+str(k_iter_BPlabel))
                            k_iter_BP += 1
                        for jj in range(1, k_iter_BP):
                            if 'BP'+str(jj+k_iter_BPlabel_previous) in sPoints_Labels:
                                EQ_iter_BP = jj
                                print(EQ_iter_BP)
                                k_iter_next = 1
                                pyDScontArgs = dst.args(name='EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP), type='EP-C')     # 'EP-C' stands for Equilibrium Point Curve. The branch will be labelled with the string aftr name='name'.
                                pyDScontArgs.freepars = [self._bifurcationParameterPyDS]   # control parameter(s) (should be among those specified in self._pyDSmodel.pars)
                                pyDScontArgs.MaxNumPoints = self._MaxNumPoints    # The following 3 parameters should work for most cases, as there should be a step-size adaption within PyDSTool.
                                pyDScontArgs.MaxStepSize = 1e-1
                                pyDScontArgs.MinStepSize = 1e-5
                                pyDScontArgs.StepSize = 5e-3
                                pyDScontArgs.LocBifPoints = ['LP', 'BP']        # 'Limit Points' and 'Branch Points may be detected'
                                pyDScontArgs.SaveEigen = True             # to tell unstable from stable branches
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
                                    self._show_computation_stop()
                                    self._showErrorMessage('Division by zero<br>')
                                
                                XDATA.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].sol[self._bifurcationParameterPyDS])
                                YDATA.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].sol[self._stateVarBif1])
                                    
                                EIGENVALUES.append(np.array([pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].sol[kk].labels['EP']['data'].evals 
                                                             for kk in range(len(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].sol[self._stateVarBif1]))]))
                                while pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('BP'+str(k_iter_next)):
                                    if (round(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('BP'+str(k_iter_next))[self._bifurcationParameterPyDS], 4) not in [round(kk , 4) for kk in sPoints_X] 
                                        and round(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('BP'+str(k_iter_next))[self._stateVarBif1], 4) not in [round(kk , 4) for kk in sPoints_Y]):    
                                        sPoints_X.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('BP'+str(k_iter_next))[self._bifurcationParameterPyDS])
                                        sPoints_Y.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('BP'+str(k_iter_next))[self._stateVarBif1])
                                        sPoints_Labels.append('EQ_BP_BP'+str(k_iter_next))
                                    k_iter_next += 1
                                k_iter_next = 1
                                while pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('LP'+str(k_iter_next)):
                                    if (round(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('LP'+str(k_iter_next))[self._bifurcationParameterPyDS], 4) not in [round(kk , 4) for kk in sPoints_X] 
                                        and round(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('LP'+str(k_iter_next))[self._stateVarBif1], 4) not in [round(kk , 4) for kk in sPoints_Y]):
                                        sPoints_X.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('LP'+str(k_iter_next))[self._bifurcationParameterPyDS])
                                        sPoints_Y.append(pyDScont['EQ'+str(EQ_iter)+'BP'+str(EQ_iter_BP)].getSpecialPoint('LP'+str(k_iter_next))[self._stateVarBif1])
                                        sPoints_Labels.append('EQ_BP_LP'+str(k_iter_next))
                                    k_iter_next += 1
    
                    except TypeError:
                        self._show_computation_stop()
                        print('Continuation failed; try with different parameters - use sliders. If that does not work, try changing maximum number of continuation points using the keyword contMaxNumPoints. If not set, default value is contMaxNumPoints=100.')       
                
                
                del(pyDScontArgs)
                del(pyDScont)
                del(pyDSode)
            if self._SVoperation:
                if self._SVoperation == '-':    
                    specialPoints = [sPoints_X, np.asarray(sPoints_Y)-np.asarray(sPoints_Z), sPoints_Labels]
                elif self._SVoperation == '+':    
                    specialPoints = [sPoints_X, np.asarray(sPoints_Y)+np.asarray(sPoints_Z), sPoints_Labels]
                else:
                    self._showErrorMessage('Only \' +\' and \'-\' are supported operations between state variables.')
            else:
                specialPoints = [sPoints_X, np.asarray(sPoints_Y), sPoints_Labels]
                
            #print('Special Points on curve: ', specialPoints)
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
                          
            if XDATA != [] and self._chooseXrange is None:
                xmaxbif = np.max([np.max(XDATA[kk]) for kk in range(len(XDATA))])
                self._chooseXrange = [0, xmaxbif]
                
            if XDATA != [] and YDATA != []:
                #plt.clf()
                _fig_formatting_2D(xdata=XDATA, 
                                ydata=YDATA,
                                xlab=self._LabelX, 
                                ylab=self._LabelY,
                                specialPoints=specialPoints, 
                                eigenvalues=EIGENVALUES, 
                                choose_xrange=self._chooseXrange, choose_yrange=self._chooseYrange,
                                ax_reformat=False, curve_replot=False, fontsize=self._chooseFontSize)
                
            else:
                self._showErrorMessage('Bifurcation diagram could not be computed. Try changing parameter values on the sliders')
                return None

        self._logs.append(log)
        
        self._show_computation_stop()
        
        
    def _initFigure(self):
        #self._show_computation_start()
        if not self._silent:
            plt.figure(self._figureNum)
            plt.clf()
            self._resetErrorMessage()
        self._showErrorMessage(str(self))
        
        

    ## utility function to mangle variable names in equations so they are accepted by PyDStool
    def _pydstoolify(self, equation):
        equation = str(equation)
        equation = equation.replace('{', '')
        equation = equation.replace('}', '')
        equation = equation.replace('_', '')
        equation = equation.replace('\\', '')
        equation = equation.replace('^', '')
        
        return equation
    
    
    def _build_bookmark(self, includeParams=True):
        if not self._silent:
            logStr = "bookmark = "
        else:
            logStr = ""
        logStr += "<modelName>." + self._generatingCommand + "('" + str(self._bifurcationParameter_for_bookmark) + "', '" + str(self._stateVarExpr1) + "', "
        #if self._stateVarBif2 is not None:
        #    logStr += "'" + str(self._stateVarBif2) + "', "
        if "initialState" not in self._generatingKwargs.keys():
            initState_str = {latex(state): pop for state, pop in self._initialState.items() if not state in self._mumotModel._constantReactants}
            logStr += "initialState = " + str(initState_str) + ", "
        if "initBifParam" not in self._generatingKwargs.keys():
            logStr += "initBifParam = " + str(self._initBifParam) + ", "
        if includeParams:
            logStr += self._get_bookmarks_params() + ", "        
        if len(self._generatingKwargs) > 0:
            for key in self._generatingKwargs:
                if type(self._generatingKwargs[key]) == str:
                    logStr += key + " = " + "\'" + str(self._generatingKwargs[key]) + "\'" + ", "
                else:
                    logStr += key + " = " + str(self._generatingKwargs[key]) + ", "    
        logStr += "bookmark = False"
        logStr += ")"
        logStr = _greekPrependify(logStr)
        logStr = logStr.replace('\\', '\\\\')
        logStr = logStr.replace('\\\\\\\\', '\\\\')
        
        
        return logStr    

    def _update_view_specific_params(self, freeParamDict=None):
        """get other parameters specific to bifurcation()"""
        if freeParamDict is None:
            freeParamDict = {}

        if self._controller is not None:
            if self._getWidgetParamValue('initialState', None) is not None:
#                self._initialState = self._getWidgetParamValue('initialState', None)
#                self._initialState = { Symbol(key): self._getWidgetParamValue('initialState', None)[key] for key in self._getWidgetParamValue('initialState', None)}
                self._initialState = { self._safeSymbol(key): self._getWidgetParamValue('initialState', None)[key] for key in self._getWidgetParamValue('initialState', None)}
            else:
                for state in self._initialState.keys():
                    # add normal and constant reactants to the _initialState 
                    self._initialState[state] = self._getInitialState(state, freeParamDict) # freeParamDict[state] if state in self._mumotModel._constantReactants else self._controller._widgetsExtraParams['init'+str(state)].value
            
            self._initBifParam = self._getWidgetParamValue('initBifParam', self._controller._widgetsExtraParams) # self._fixedParams['initBifParam'] if self._fixedParams.get('initBifParam') is not None else self._controller._widgetsExtraParams['initBifParam'].value


    ## gets and returns names and values from widgets, overrides method defined in parent class MuMoTview
    def _get_argDict(self):
        paramNames = []
        paramValues = []
        if self._controller is not None:
            for name, value in self._controller._widgetsFreeParams.items():
                #print("wdg-name: " + str(name) + " wdg-val: " + str(value.value))
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
                #print("fix-name: " + str(key) + " fix-val: " + str(item))
                paramNames.append(str(key))
                paramValues.append(item)
        
        argNamesSymb = list(map(Symbol, paramNames))
        argDict = dict(zip(argNamesSymb, paramValues))

        if self._mumotModel._systemSize:
            argDict[self._mumotModel._systemSize] = 1

        #@todo: is this necessary? for which view?
        systemSize = Symbol('systemSize')
        argDict[systemSize] = self._getSystemSize()
        
        return argDict


class MuMoTstochasticSimulationView(MuMoTview):
    """Stochastic-simulations-view view.
    
    For views that allow for multiple runs with different random-seeds.
    
    """
    ## the system state at the start of the simulation (timestep zero) described as proportion of _systemSize
    _initialState = None
    ## variable to link a color to each reactant
    _colors = None
    ## list of colors to pass to the _fig_formatting method (that does not include constant reactants)
    _colors_list = None
    ## random seed
    _randomSeed = None
    ## simulation length (in the same time unit of the rates)
    _maxTime = None
    ## visualisation type
    _visualisationType = None
    ## reactants to display on the two axes
    _finalViewAxes = None
    ## flag to plot proportions or full populations
    _plotProportions = None
    ## realtimePlot flag (TRUE = the plot is updated each timestep of the simulation; FALSE = it is updated once at the end of the simulation)
    _realtimePlot = None
    ## latest computed results
    _latestResults = None
    ## number of runs to execute
    _runs = None
    ## flag to set if the results from multimple runs must be aggregated or not  
    _aggregateResults = None
    ## variable to store simulation time during simulation
    _t = 0
    ## variable to store the current simulation state
    _currentState = 0
    ## variable to store the time evolution of the simulation
    _evo = None
    ## progress bar
    _progressBar = None
    
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
        if not self._silent:
            display(self._progressBar)
            
        super().__init__(model=model, controller=controller, figure=figure, params=params, **kwargs)

        with io.capture_output() as log:
#         if True:

            freeParamDict = self._get_argDict()
            if self._controller is None:
                # storing the initial state
                self._initialState = {}
                for state, pop in SSParams["initialState"].items():
                    if isinstance(state, str):
                        self._initialState[process_sympy(state)] = pop  # convert string into SymPy symbol
                    else:
                        self._initialState[state] = pop
                # add to the _initialState the constant reactants
                for constantReactant in self._mumotModel._getAllReactants()[1]:
                    self._initialState[constantReactant] = freeParamDict[constantReactant]
                # storing all values of MA-specific parameters
                self._maxTime = SSParams["maxTime"]
                self._randomSeed = SSParams["randomSeed"]
                self._visualisationType = SSParams["visualisationType"]
                final_x = str(process_sympy(SSParams.get("final_x", latex(sorted(self._mumotModel._getAllReactants()[0], key=str)[0]))))
                #if isinstance(final_x, str): final_x = process_sympy(final_x)
                final_y = str(process_sympy(SSParams.get("final_y", latex(sorted(self._mumotModel._getAllReactants()[0], key=str)[0]))))
                #if isinstance(final_y, str): final_y = process_sympy(final_y)
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
                        self._initialState[process_sympy(state)] = pop[0]  # convert string into SymPy symbol
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
            #colors = cm.rainbow(np.linspace(0, 1, len(self._mumotModel._reactants) ))  # @UndefinedVariable
            self._colors = {}
            self._colors_list = []
            for idx, state in enumerate(sorted(self._initialState.keys(), key=str)):
                self._colors[state] = LINE_COLOR_LIST[idx] 
                if state not in self._mumotModel._constantReactants:
                    self._colors_list.append(LINE_COLOR_LIST[idx])
            
        self._logs.append(log)
        if not self._silent:
            self._computeAndPlotSimulation()
    
    def _constructorSpecificParams(self, _):
        pass
    
    def _computeAndPlotSimulation(self, _=None):
        with io.capture_output() as log:
#         if True:
#             log=''
            self._show_computation_start()
            self._update_params()
            self._log("Stochastic Simulation")
            # if you need to access the standalone view, you can use the command self._printStandaloneViewCmd(), this is very useful for developer and advanced users as indicated in issue #92 

            # Clearing the plot and setting the axes
            self._initFigure()
            
            self._latestResults = []
            for r in range(self._runs):
                runID = "[" + str(r+1) + "/" + str(self._runs) + "] " if self._runs > 1 else ''
                self._latestResults.append(self._runSingleSimulation(self._randomSeed+r, runID=runID))
            
            ## Final Plot
            if not self._realtimePlot or self._aggregateResults:
#                 for results in self._latestResults:
#                     self._updateSimultationFigure(results, fullPlot=True)
                self._updateSimultationFigure(self._latestResults, fullPlot=True)
            
            self._show_computation_stop()
        self._logs.append(log)
        if self._controller is not None: self._updateDownloadLink()


    def _update_view_specific_params(self, freeParamDict=None):
        """Getting other parameters specific to SSA."""

        if freeParamDict is None:
            freeParamDict = {}
        if self._controller is not None:
            if self._getWidgetParamValue('initialState', None) is not None:
#                self._initialState = self._getWidgetParamValue('initialState', None)
#                self._initialState = { Symbol(key): self._getWidgetParamValue('initialState', None)[key] for key in self._getWidgetParamValue('initialState', None)}
                self._initialState = { self._safeSymbol(key): self._getWidgetParamValue('initialState', None)[key] for key in self._getWidgetParamValue('initialState', None)}
            else:
                for state in self._initialState.keys():
                    # add normal and constant reactants to the _initialState 
                    self._initialState[state] = self._getInitialState(state, freeParamDict) # freeParamDict[state] if state in self._mumotModel._constantReactants else self._controller._widgetsExtraParams['init'+str(state)].value
            self._randomSeed =  self._getWidgetParamValue('randomSeed', self._controller._widgetsExtraParams) # self._fixedParams['randomSeed'] if self._fixedParams.get('randomSeed') is not None else self._controller._widgetsExtraParams['randomSeed'].value
            self._visualisationType = self._getWidgetParamValue('visualisationType', self._controller._widgetsPlotOnly) # self._fixedParams['visualisationType'] if self._fixedParams.get('visualisationType') is not None else self._controller._widgetsPlotOnly['visualisationType'].value
            if self._visualisationType == 'final':
                self._finalViewAxes = (self._getWidgetParamValue('final_x', self._controller._widgetsPlotOnly), self._getWidgetParamValue('final_y', self._controller._widgetsPlotOnly))
                #self._finalViewAxes = (self._fixedParams['final_x'] if self._fixedParams.get('final_x') is not None else self._controller._widgetsPlotOnly['final_x'].value, self._fixedParams['final_y'] if self._fixedParams.get('final_y') is not None else self._controller._widgetsPlotOnly['final_y'].value)
            self._plotProportions = self._getWidgetParamValue('plotProportions', self._controller._widgetsPlotOnly) # self._fixedParams['plotProportions'] if self._fixedParams.get('plotProportions') is not None else self._controller._widgetsPlotOnly['plotProportions'].value
            self._maxTime = self._getWidgetParamValue('maxTime', self._controller._widgetsExtraParams) # self._fixedParams['maxTime'] if self._fixedParams.get('maxTime') is not None else self._controller._widgetsExtraParams['maxTime'].value
            self._realtimePlot = self._getWidgetParamValue('realtimePlot', self._controller._widgetsExtraParams) # self._fixedParams['realtimePlot'] if self._fixedParams.get('realtimePlot') is not None else self._controller._widgetsExtraParams['realtimePlot'].value
            self._runs = self._getWidgetParamValue('runs', self._controller._widgetsExtraParams) # self._fixedParams['runs'] if self._fixedParams.get('runs') is not None else self._controller._widgetsExtraParams['runs'].value
            self._aggregateResults = self._getWidgetParamValue('aggregateResults', self._controller._widgetsPlotOnly) # self._fixedParams['aggregateResults'] if self._fixedParams.get('aggregateResults') is not None else self._controller._widgetsPlotOnly['aggregateResults'].value
    
    def _initSingleSimulation(self):
        self._progressBar.max = self._maxTime 
         
        # initialise populations by multiplying proportion with _systemSize
        #currentState = copy.deepcopy(self._initialState)
        #currentState = {s:p*self._systemSize for s,p in self._initialState.items()}
        self._currentState = {}
        leftOvers = {}
        for state, prop in self._initialState.items():
            pop = prop*self._systemSize
            if (not _almostEqual(pop, math.floor(pop))) and (state not in self._mumotModel._constantReactants):
                leftOvers[state] = pop - math.floor(pop)
            self._currentState[state] = math.floor(pop)
        # if approximations resulted in one agent less, it is added randomly (with probability proportional to the rounding quantities)
        sumReactants = sum([self._currentState[state] for state in self._currentState.keys() if state not in self._mumotModel._constantReactants])
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
            # update progress bar
            self._progressBar.value = self._t
            self._progressBar.description = "Loading " + runID + str(round(self._t/self._maxTime*100)) + "%:"
             
            timeInterval, self._currentState = self._simulationStep()
            # increment time
            self._t += timeInterval
             
            # log step
            for state, pop in self._currentState.items():
                if state in self._mumotModel._constantReactants: continue
                self._evo[state].append(pop)
            self._evo['time'].append(self._t)
            
#             print (self._evo)
            # Plotting each timestep
            if self._realtimePlot:
                self._updateSimultationFigure(allResults=self._latestResults, fullPlot=False, currentEvo=self._evo)
         
        self._progressBar.value = self._progressBar.max
        self._progressBar.description = "Completed 100%:"
#         print("Temporal evolution per state: " + str(self._evo))
        return self._evo
    
    def _updateSimultationFigure(self, allResults, fullPlot=True, currentEvo=None):
        if (self._visualisationType == "evo"):
#             if not fullPlot and len(currentEvo['time']) <= 2: # if incremental plot is requested, but it's the first item, we operate as fullPlot (to allow legend)
#                 fullPlot = True
    
            # if fullPlot, plot all time-evolution
            if fullPlot or len(currentEvo['time']) <= 2:
                y_max = 1.0 if self._plotProportions else self._systemSize
                if self._aggregateResults and len(allResults) > 1:  # plot in aggregate mode only if there's enough data
                    self._initFigure()
                    steps = 10
                    timesteps = list(np.arange(0, self._maxTime, step=self._maxTime/steps))
    #                 if timesteps[-1] - self._maxTime > self._maxTime/(steps*2):
    #                     timesteps.append(self._maxTime)
    #                 else:
    #                     timesteps[-1] = self._maxTime
                    if not _almostEqual(timesteps[-1], self._maxTime):
                        timesteps.append(self._maxTime)
                    
                    for state in sorted(self._initialState.keys(), key=str):
                        if state == 'time': continue
                        if state in self._mumotModel._constantReactants: continue
                        boxesData = []
                        avgs = []
                        for timestep in timesteps:
                            boxData = []
                            for results in allResults:
                                idx = max(0, bisect_left(results['time'], timestep))
                                if self._plotProportions:
                                    boxData.append(results[state][idx]/self._systemSize)
                                else:
                                    boxData.append(results[state][idx])
                            y_max = max(y_max, max(boxData))
                            boxesData.append(boxData)
                            avgs.append(np.mean(boxData))
                            #bplot = plt.boxplot(boxData, patch_artist=True, positions=[timestep], manage_xticks=False, widths=self._maxTime/(steps*3) )
    #                         print("Plotting bxplt at positions " + str(timestep) + " generated from idx = " + str(idx))
                        plt.plot(timesteps, avgs, color=self._colors[state])
                        bplots = plt.boxplot(boxesData, patch_artist=True, positions=timesteps, manage_xticks=False, widths=self._maxTime/(steps*3))
    #                     for patch, color in zip(bplots['boxes'], [self._colors[state]]*len(timesteps)):
    #                         patch.set_facecolor(color)
    #                     bplot['boxes'].set_facecolor(self._colors[state])
                        #plt.setp(bplots['boxes'], color=self._colors[state])
                        wdt = 2
                        for box in bplots['boxes']:
                            # change outline color
                            box.set(color=self._colors[state], linewidth=wdt)
                            #box.set( color='black', linewidth=2)
                            # change fill color
                            box.set(facecolor='None')
                            #box.set( facecolor = self._colors[state] )
                        plt.setp(bplots['whiskers'], color=self._colors[state], linewidth=wdt)
                        plt.setp(bplots['caps'], color=self._colors[state], linewidth=wdt)
                        plt.setp(bplots['medians'], color=self._colors[state], linewidth=wdt)
                        #plt.setp(bplots['fliers'], color=self._colors[state], marker='o', alpha=0.5)
                        for flier in bplots['fliers']:
                            flier.set_markerfacecolor(self._colors[state])
                            flier.set_markeredgecolor("None")
                            flier.set(marker='o', alpha=0.5)
                
                    padding_x = self._maxTime/20.0
                    padding_y = y_max/20.0
                    
                else:
                    for state in sorted(self._initialState.keys(), key=str):
                        if (state == 'time'): continue
                        if state in self._mumotModel._constantReactants: continue
                        #xdata = []
                        #xdata.append( results['time'] )
                        for results in allResults:
                            #ydata = []
                            if self._plotProportions:
                                ydata = [y/self._systemSize for y in results[state]]
                                #ydata.append(ytmp)
                                y_max = max(y_max, max(ydata))
                            else:
                                ydata = results[state]
                                y_max = max(y_max, max(results[state]))
                            #xdata=[list(np.arange(len(list(evo.values())[0])))]*len(evo.values()), ydata=list(evo.values()), curvelab=list(evo.keys())
                            plt.plot(results['time'], ydata, color=self._colors[state], lw=2)
                    #_fig_formatting_2D(xdata=xdata, ydata=ydata, curvelab=labels, curve_replot=False, choose_xrange=(0, self._maxTime), choose_yrange=(0, y_max) )
                    padding_x = self._maxTime/100.0
                    padding_y = y_max/100.0

                # plot legend
                if self._plotProportions:
                    stateNamesLabel = [r'$' + _doubleUnderscorify(_greekPrependify(str(Symbol('Phi_{'+str(state)+'}')))) + '$' for state in sorted(self._initialState.keys(), key=str) if state not in self._mumotModel._constantReactants]
                    #stateNamesLabel = [r'$'+latex(Symbol('Phi_'+str(state))) +'$' for state in sorted(self._initialState.keys(), key=str) if state not in self._mumotModel._constantReactants]
                else:
                    stateNamesLabel = [r'$' + _doubleUnderscorify(_greekPrependify(str(Symbol(str(state))))) + '$' for state in sorted(self._initialState.keys(), key=str) if state not in self._mumotModel._constantReactants]
                    #stateNamesLabel = [r'$'+latex(Symbol(str(state)))+'$' for state in sorted(self._initialState.keys(), key=str) if state not in self._mumotModel._constantReactants]
                markers = [plt.Line2D([0, 0], [0, 0], color=self._colors[state], marker='s', linestyle='', markersize=10) for state in sorted(self._initialState.keys(), key=str) if state not in self._mumotModel._constantReactants]
                plt.legend(markers, stateNamesLabel, loc='upper right', borderaxespad=0., numpoints=1)  # bbox_to_anchor=(0.885, 1),
                _fig_formatting_2D(figure=self._figure, xlab='time t', ylab='reactants', choose_xrange=(0-padding_x, self._maxTime+padding_x), choose_yrange=(0-padding_y, y_max+padding_y), aspectRatioEqual=False, grid=True)
                 
            if not fullPlot:  # If realtime-plot mode, draw only the last timestep rather than overlay all
                xdata = []
                ydata = []
                y_max = 1.0 if self._plotProportions else self._systemSize
                for state in sorted(self._initialState.keys(), key=str):
                    if (state == 'time'): continue
                    if state in self._mumotModel._constantReactants: continue
                    xdata.append(currentEvo['time'][-2:])
                    # modify if plotProportions
                    ytmp = [y / self._systemSize for y in currentEvo[state][-2:]] if self._plotProportions else currentEvo[state][-2:]
                    y_max = max(y_max, max(ytmp))
                    ydata.append(ytmp)
                _fig_formatting_2D(xdata=xdata, ydata=ydata, curve_replot=False, choose_xrange=(0, self._maxTime), choose_yrange=(0, y_max), aspectRatioEqual=False, LineThickness=2, grid=True, line_color_list=self._colors_list)
                
#                 y_max = 1.0 if self._plotProportions else self._systemSize
#                 for state in sorted(self._initialState.keys(), key=str):
#                     if (state == 'time'): continue
#                     # modify if plotProportions
#                     ytmp = [y / self._systemSize for y in currentEvo[state][-2:] ] if self._plotProportions else currentEvo[state][-2:]
#                     
#                     y_max = max(y_max, max(ytmp))
#                     plt.plot(currentEvo['time'][-2:], ytmp, color=self._colors[state], lw=2)
#                 padding_x = 0
#                 padding_y = 0
#                 _fig_formatting_2D(figure=self._figure, xlab="Time", ylab="Reactants", choose_xrange=(0-padding_x, self._maxTime+padding_x), choose_yrange=(0-padding_y, y_max+padding_y), aspectRatioEqual=False )

        elif (self._visualisationType == "final"):
            points_x = []
            points_y = []
            
            if not fullPlot:  # if it's a runtime plot 
                self._initFigure()  # the figure must be cleared each timestep
                for state in self._mumotModel._getAllReactants()[0]:  # the current point added to the list of points
                    if str(state) == self._finalViewAxes[0]:
                        points_x.append(currentEvo[state][-1]/self._systemSize if self._plotProportions else currentEvo[state][-1])
                        trajectory_x = [x/self._systemSize for x in currentEvo[state]] if self._plotProportions else currentEvo[state]
                    if str(state) == self._finalViewAxes[1]:
                        points_y.append(currentEvo[state][-1]/self._systemSize if self._plotProportions else currentEvo[state][-1])
                        trajectory_y = [y/self._systemSize for y in currentEvo[state]] if self._plotProportions else currentEvo[state]
                 
            if self._aggregateResults and len(allResults) > 2:  # plot in aggregate mode only if there's enough data
                self._initFigure()
                samples_x = []
                samples_y = []
                for state in self._mumotModel._getAllReactants()[0]:
                    if str(state) == self._finalViewAxes[0]:
                        for results in allResults:
                            samples_x.append(results[state][-1]/self._systemSize if self._plotProportions else results[state][-1])
                    if str(state) == self._finalViewAxes[1]:
                        for results in allResults:
                            samples_y.append(results[state][-1]/self._systemSize if self._plotProportions else results[state][-1])
                samples = np.column_stack((samples_x, samples_y))
                _plot_point_cov(samples, nstd=1, alpha=0.5, color='green')
            else:
                for state in self._mumotModel._getAllReactants()[0]:
                    if str(state) == self._finalViewAxes[0]:
                        for results in allResults:
                            points_x.append(results[state][-1]/self._systemSize if self._plotProportions else results[state][-1])
                    if str(state) == self._finalViewAxes[1]:
                        for results in allResults:
                            points_y.append(results[state][-1]/self._systemSize if self._plotProportions else results[state][-1])

            #_fig_formatting_2D(xdata=[xdata], ydata=[ydata], curve_replot=False, xlab=self._finalViewAxes[0], ylab=self._finalViewAxes[1])
            if not fullPlot: plt.plot(trajectory_x, trajectory_y, '-', c='0.6')
            plt.plot(points_x, points_y, 'ro')
            if self._plotProportions:
                xlab = r'$' + '\Phi_{' + _doubleUnderscorify(_greekPrependify(str(self._finalViewAxes[0])))+'}$'
                ylab = r'$' + '\Phi_{' + _doubleUnderscorify(_greekPrependify(str(self._finalViewAxes[1])))+'}$'
            else:
                xlab = r'$' + _doubleUnderscorify(_greekPrependify(str(self._finalViewAxes[0])))+'$'
                ylab = r'$' + _doubleUnderscorify(_greekPrependify(str(self._finalViewAxes[1])))+'$'                
            _fig_formatting_2D(figure=self._figure, aspectRatioEqual=True, xlab=xlab, ylab=ylab)
        elif (self._visualisationType == "barplot"):
            self._initFigure()
            
            finaldata = []
            colors = []
            stdev = []
            y_max = 1.0 if self._plotProportions else self._systemSize

            if fullPlot:
                for state in sorted(self._initialState.keys(), key=str):
                    if state == 'time': continue
                    if state in self._mumotModel._constantReactants: continue
                    if self._aggregateResults and len(allResults) > 0:
                        points = []
                        for results in allResults:
                            points.append(results[state][-1]/self._systemSize if self._plotProportions else results[state][-1])
                        avg = np.mean(points)
                        stdev.append(np.std(points))
                    else:
                        if allResults:
                            avg = allResults[-1][state][-1]/self._systemSize if self._plotProportions else allResults[-1][state][-1]
                        else:
                            avg = 0
                        stdev.append(0)
                    finaldata.append(avg)
                    #labels.append(state)
                    colors.append(self._colors[state])
            else:
                for state in sorted(self._initialState.keys(), key=str):
                    if state == 'time': continue
                    if state in self._mumotModel._constantReactants: continue
                    finaldata.append(currentEvo[state][-1]/self._systemSize if self._plotProportions else currentEvo[state][-1])
                    stdev.append(0)
                    #labels.append(state)
                    colors.append(self._colors[state])
             
#             plt.pie(finaldata, labels=labels, autopct=_make_autopct(piedata), colors=colors) #shadow=True, startangle=90,
            xpos = np.arange(len(finaldata))  # the x locations for the bars
            width = 1       # the width of the bars
            plt.bar(xpos, finaldata, width, color=colors, yerr=stdev, ecolor='black')
            # set axes
            ax = plt.gca()
            ax.set_xticks(xpos)  # for matplotlib < 2 ---> ax.set_xticks(xpos - (width/2) )
            y_max = max(y_max, max(finaldata))
            padding_y = y_max/100.0 if self._runs <= 1 else y_max/20.0 
            # set lables
            if self._plotProportions:
                stateNamesLabel = [r'$' + _doubleUnderscorify(_greekPrependify(str(Symbol('Phi_{'+str(state)+'}')))) + '$' for state in sorted(self._initialState.keys(), key=str) if state not in self._mumotModel._constantReactants]
                #stateNamesLabel = [r'$'+latex(Symbol('Phi_'+str(state))) +'$' for state in sorted(self._initialState.keys(), key=str)]
            else:
                stateNamesLabel = [r'$' + _doubleUnderscorify(_greekPrependify(str(Symbol(str(state))))) + '$' for state in sorted(self._initialState.keys(), key=str) if state not in self._mumotModel._constantReactants]
                #stateNamesLabel = [r'$'+latex(Symbol(str(state)))+'$' for state in sorted(self._initialState.keys(), key=str)]
            ax.set_xticklabels(stateNamesLabel)
            _fig_formatting_2D(figure=self._figure, xlab="reactants", ylab="population proportion" if self._plotProportions else "population size", aspectRatioEqual=False)
            plt.ylim((0, y_max+padding_y))  # @todo: to fix the choose_yrange of _fig_formatting_2D (issue #104) 
        # update the figure
        if not self._silent:
            self._figure.canvas.draw()
            
    def _redrawOnly(self, _=None):
        self._update_params()
        self._initFigure()
        self._updateSimultationFigure(self._latestResults, fullPlot=True)
#         for results in self._latestResults:
#             self._updateSimultationFigure(results, fullPlot=True)
        
    def _initFigure(self):
        if not self._silent:
            plt.figure(self._figureNum)
            plt.clf()

        if (self._visualisationType == 'evo'):
            pass 
        elif (self._visualisationType == "final"):
            #plt.axes().set_aspect('equal')
            if self._plotProportions:           
                plt.xlim((0, 1.0))
                plt.ylim((0, 1.0))
            else:
                plt.xlim((0, self._systemSize))
                plt.ylim((0, self._systemSize))
            #plt.axes().set_xlabel(self._finalViewAxes[0])
            #plt.axes().set_ylabel(self._finalViewAxes[1])
        elif (self._visualisationType == "barplot"):
            #plt.axes().set_aspect('equal') #for piechart
            #plt.axes().set_aspect('auto') # for barchart
            pass
    
    def _convertLatestDataIntoCSV(self):
        """Formatting of the latest simulation data in the CSV format"""
        csv_results = ""
        line = 'runID' + ',' + 'time'
        for state in sorted(self._initialState.keys(), key=str):
            line += ',' + str(state)
        line += '\n'
        csv_results += line
        for runID,runData in enumerate(self._latestResults):
            for t, timestep in enumerate(runData['time']):            
                line = str(runID) + ',' + str(timestep) 
                for state in sorted(self._initialState.keys(), key=str):
                    if state in self._mumotModel._constantReactants: continue
                    line += ',' + str(runData[state][t])
                line +=  '\n'
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
    ## number of simulation timesteps
    _maxTimeSteps = None
    ## time scaling (i.e. lenght of each timestep)
    _timestepSize = None
    ## visualise the agent trace (on moving particles)
    _showTrace = None
    ## visualise the agent trace (on moving particles)
    _showInteractions = None

    def _constructorSpecificParams(self, MAParams):
        if self._controller is None:
            self._timestepSize = MAParams.get('timestepSize', 1)
            self._netType = _decodeNetworkTypeFromString(MAParams['netType'])
            if self._netType != NetworkType.FULLY_CONNECTED: 
                self._netParam = MAParams['netParam']
                if self._netType == NetworkType.DYNAMIC: 
                    self._motionCorrelatedness = MAParams['motionCorrelatedness']
                    self._particleSpeed = MAParams['particleSpeed']
                    self._showTrace = MAParams.get('showTrace', False)
                    self._showInteractions = MAParams.get('showInteractions', False)
        else:
            # needed for bookmarking
            self._generatingCommand = "multiagent"
            # storing fixed params
            if self._fixedParams.get('netType') is not None:
                self._fixedParams['netType'] = _decodeNetworkTypeFromString(self._fixedParams['netType'])
            self._netType = _decodeNetworkTypeFromString(MAParams['netType'][0])
            self._update_net_params(False)
        
        self._mumotModel._getSingleAgentRules()
        #print(self._mumotModel._agentProbabilities)
        
        # check if any network is available or only moving particles
        onlyDynamic = False
        (_, allConstantReactants) = self._mumotModel._getAllReactants()
        for rule in self._mumotModel._rules:
            if EMPTYSET_SYMBOL in rule.lhsReactants + rule.rhsReactants:
                onlyDynamic = True
                break
            for react in rule.lhsReactants + rule.rhsReactants:
                if react in allConstantReactants:
                    onlyDynamic = True
            if onlyDynamic: break
            
        if onlyDynamic:
            #if (not self._controller) and (not self._netType == NetworkType.DYNAMIC): # if the user has specified the network type, we notify him/her through error-message
            #    self._errorMessage.value = "Only Moving-Particle netType is available when rules contain the emptyset."
            if not self._netType == NetworkType.DYNAMIC: print("Only Moving-Particle netType is available when rules contain the emptyset or constant reactants.")
            self._netType = NetworkType.DYNAMIC
            if self._controller:  # updating value and disabling widget
                if self._controller._widgetsExtraParams.get('netType') is not None:
                    self._controller._widgetsExtraParams['netType'].value = NetworkType.DYNAMIC
                    self._controller._widgetsExtraParams['netType'].disabled = True
                else:
                    self._fixedParams['netType'] = NetworkType.DYNAMIC
                self._controller._update_net_params()
            else:  # this is a standalone view
                # if the assigned value of net-param is not consistent with the input, raise a WARNING and set the default value to 0.1
                if self._netParam < 0 or self._netParam > 1:
                    wrnMsg = "WARNING! net-param value " + str(self._netParam) + " is invalid for Moving-Particles. Valid range is [0,1] indicating the particles' communication range. \n"
                    self._netParam = 0.1
                    wrnMsg += "New default values is '_netParam'=" + str(self._netParam)
                    print(wrnMsg)
    
    def _build_bookmark(self, includeParams=True):
        logStr = "bookmark = " if not self._silent else ""
        logStr += "<modelName>." + self._generatingCommand + "("
#         logStr += _find_obj_names(self._mumotModel)[0] + "." + self._generatingCommand + "("
        if includeParams:
            logStr += self._get_bookmarks_params()
            logStr += ", "
        logStr = logStr.replace('\\', '\\\\')
        
        initState_str = {latex(state): pop for state, pop in self._initialState.items() if not state in self._mumotModel._constantReactants}
        logStr += "initialState = " + str(initState_str)
        logStr += ", maxTime = " + str(self._maxTime)
        logStr += ", timestepSize = " + str(self._timestepSize)
        logStr += ", randomSeed = " + str(self._randomSeed)
        logStr += ", netType = '" + _encodeNetworkTypeToString(self._netType) + "'"
        if not self._netType == NetworkType.FULLY_CONNECTED:
            logStr += ", netParam = " + str(self._netParam)
        if self._netType == NetworkType.DYNAMIC:
            logStr += ", motionCorrelatedness = " + str(self._motionCorrelatedness)
            logStr += ", particleSpeed = " + str(self._particleSpeed)
            logStr += ", showTrace = " + str(self._showTrace)
            logStr += ", showInteractions = " + str(self._showInteractions)
        logStr += ", visualisationType = '" + str(self._visualisationType) + "'"
        if self._visualisationType == 'final':
            # these loops are necessary to return the latex() format of the reactant 
            for reactant in self._mumotModel._getAllReactants()[0]:
                if str(reactant) == self._finalViewAxes[0]:
                    logStr += ", final_x = '" + latex(reactant).replace('\\', '\\\\') + "'"
                    break
            for reactant in self._mumotModel._getAllReactants()[0]:
                if str(reactant) == self._finalViewAxes[1]:
                    logStr += ", final_y = '" + latex(reactant).replace('\\', '\\\\') + "'"
                    break
        logStr += ", plotProportions = " + str(self._plotProportions)
        logStr += ", realtimePlot = " + str(self._realtimePlot)
        logStr += ", runs = " + str(self._runs)
        logStr += ", aggregateResults = " + str(self._aggregateResults)
        logStr += ", silent = " + str(self._silent)
        logStr += ", bookmark = False"
#         if len(self._generatingKwargs) > 0:
#             for key in self._generatingKwargs:
#                 if type(self._generatingKwargs[key]) == str:
#                     logStr += key + " = " + "\'"+ str(self._generatingKwargs[key]) + "\'" + ", "
#                 else:
#                     logStr += key + " = " + str(self._generatingKwargs[key]) + ", "
            
        logStr += ")"
        return logStr
    
    def _printStandaloneViewCmd(self):
        MAParams = {}
        MAParams["initialState"] = {latex(state): pop for state, pop in self._initialState.items() if not state in self._mumotModel._constantReactants}
        MAParams["maxTime"] = self._maxTime 
        MAParams['timestepSize'] = self._timestepSize
        MAParams["randomSeed"] = self._randomSeed
        MAParams['netType'] = _encodeNetworkTypeToString(self._netType)
        if not self._netType == NetworkType.FULLY_CONNECTED:
            MAParams['netParam'] = self._netParam 
        if self._netType == NetworkType.DYNAMIC:
            MAParams['motionCorrelatedness'] = self._motionCorrelatedness
            MAParams['particleSpeed'] = self._particleSpeed
            MAParams['showTrace'] = self._showTrace
            MAParams['showInteractions'] = self._showInteractions
        MAParams["visualisationType"] = self._visualisationType
        if self._visualisationType == 'final':
            # this loop is necessary to return the latex() format of the reactant 
            for reactant in self._mumotModel._getAllReactants()[0]:
                if str(reactant) == self._finalViewAxes[0]: MAParams['final_x'] = latex(reactant)
                if str(reactant) == self._finalViewAxes[1]: MAParams['final_y'] = latex(reactant)
        MAParams["plotProportions"] = self._plotProportions
        MAParams["realtimePlot"] = self._realtimePlot
        MAParams["runs"] = self._runs
        MAParams["aggregateResults"] = self._aggregateResults
#         sortedDict = "{"
#         for key,value in sorted(MAParams.items()):
#             sortedDict += "'" + key + "': " + str(value) + ", "
#         sortedDict += "}"
        print("mumot.MuMoTmultiagentView(<modelName>, None, " + self._get_bookmarks_params().replace('\\', '\\\\') + ", SSParams = " + str(MAParams) + " )")
    
    def _update_view_specific_params(self, freeParamDict=None):
        """read the new parameters (in case they changed in the controller) specific to multiagent(). This function should only update local parameters and not compute data"""
        if freeParamDict is None:
            freeParamDict = {}

        super()._update_view_specific_params(freeParamDict)
        self._adjust_barabasi_network_range()
        if self._controller is not None:
            self._netType = self._getWidgetParamValue('netType', self._controller._widgetsExtraParams)
            if self._netType != NetworkType.FULLY_CONNECTED: # this used to refer only to value in self._fixedParams; possible bug?
                self._netParam = self._getWidgetParamValue('netParam', self._controller._widgetsExtraParams) # self._fixedParams['netParam'] if self._fixedParams.get('netParam') is not None else self._controller._widgetsExtraParams['netParam'].value
                if self._netType is None or self._netType == NetworkType.DYNAMIC: # this used to refer only to value in self._fixedParams; possible bug?
                    self._motionCorrelatedness = self._getWidgetParamValue('motionCorrelatedness', self._controller._widgetsExtraParams) # self._fixedParams['motionCorrelatedness'] if self._fixedParams.get('motionCorrelatedness') is not None else self._controller._widgetsExtraParams['motionCorrelatedness'].value
                    self._particleSpeed = self._getWidgetParamValue('particleSpeed', self._controller._widgetsExtraParams) # self._fixedParams['particleSpeed'] if self._fixedParams.get('particleSpeed') is not None else self._controller._widgetsExtraParams['particleSpeed'].value
                    self._showTrace = self._getWidgetParamValue('showTrace', self._controller._widgetsPlotOnly) # self._fixedParams['showTrace'] if self._fixedParams.get('showTrace') is not None else self._controller._widgetsPlotOnly['showTrace'].value
                    self._showInteractions = self._getWidgetParamValue('showInteractions', self._controller._widgetsPlotOnly) # self._fixedParams['showInteractions'] if self._fixedParams.get('showInteractions') is not None else self._controller._widgetsPlotOnly['showInteractions'].value
            self._timestepSize = self._getWidgetParamValue('timestepSize', self._controller._widgetsExtraParams) # self._fixedParams['timestepSize'] if self._fixedParams.get('timestepSize') is not None else self._controller._widgetsExtraParams['timestepSize'].value
        
        self._computeScalingFactor()
    
    def _initSingleSimulation(self):
        super()._initSingleSimulation()    
        # init the network
        self._initGraph()
        # init the agents
        self._initMultiagent()
        
    def _initFigure(self):
        super()._initFigure()
        if self._visualisationType == "graph": 
            #plt.axes().set_aspect('equal')
            ax = plt.gca()
            ax.set_aspect('equal')
    
    #def _updateSimultationFigure(self, evo, fullPlot=True):
    def _updateSimultationFigure(self, allResults, fullPlot=True, currentEvo=None):
        if (self._visualisationType == "graph"):
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
                    xs[self._agents[a]].append(self._positions[a][0])
                    ys[self._agents[a]].append(self._positions[a][1])
                    
                    if self._showInteractions:
                        agent_p = [self._positions[a][0] , self._positions[a][1]]
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
                            #plt.plot((self._positions[a][0], self._positions[n][0]),(self._positions[a][1], self._positions[n][1]), '-', c='y')
                    
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
            stateNamesLabel = [r'$' + _doubleUnderscorify(_greekPrependify(str(Symbol(str(state))))) + '$' for state in sorted(self._initialState.keys(), key=str) if state not in self._mumotModel._constantReactants]
            #stateNamesLabel = [r'$'+latex(Symbol(str(state)))+'$' for state in sorted(self._initialState.keys(), key=str)]
            markers = [plt.Line2D([0, 0], [0, 0], color=self._colors[state], marker='o', linestyle='', markersize=10) for state in sorted(self._initialState.keys(), key=str)]
            plt.legend(markers, stateNamesLabel, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0., numpoints=1)

        super()._updateSimultationFigure(allResults, fullPlot, currentEvo) 
  
    def _computeScalingFactor(self):
        # Determining the minimum speed of the process (thus the max-scaling factor)
        maxRatesAll = 0
        for reactant, reactions in self._mumotModel._agentProbabilities.items():
            if reactant == EMPTYSET_SYMBOL: continue  # not considering the spontaneous births as limiting component for simulation step
            sumRates = 0
            for reaction in reactions:
                sumRates += self._ratesDict[str(reaction[1])]
            #print("self._ratesDict " + str(self._ratesDict) )
            #print("reactant " + str(reactant) + " has sum rates: " + str(sumRates))
            if sumRates > maxRatesAll:
                maxRatesAll = sumRates
        
        if maxRatesAll > 0: maxTimestepSize = 1/maxRatesAll 
        else: maxTimestepSize = 1        
        # if the timestep size is too small (and generated a too large number of timesteps, it returns an error!)
        if math.ceil(self._maxTime / maxTimestepSize) > 10000000:
            errorMsg = "ERROR! Invalid rate values. The current rates limit the agent timestep to be too small and would correspond to more than 10 milions simulation timesteps.\n"\
                        "Please modify the free parameters value to allow quicker simulations."
            self._showErrorMessage(errorMsg)
            raise MuMoTValueError(errorMsg)
        if self._timestepSize > maxTimestepSize:
            self._timestepSize = maxTimestepSize
        self._maxTimeSteps = math.ceil(self._maxTime / self._timestepSize)
        if self._controller is not None and self._controller._widgetsExtraParams.get('timestepSize'):
            self._update_timestepSize_widget(self._timestepSize, maxTimestepSize, self._maxTimeSteps)
        else:
            if self._fixedParams.get('timestepSize') != self._timestepSize:
                self._showErrorMessage("Time step size was fixed to " + str(self._fixedParams.get('timestepSize')) + " but needs to be updated to " + str(self._timestepSize))
                self._fixedParams['timestepSize'] = self._timestepSize 
#         print("timestepSize s=" + str(self._timestepSize))
    
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
            self._controller._widgetsExtraParams['timestepSize'].min = min(maxTimestepSize/100, timestepSize)
            self._controller._widgetsExtraParams['timestepSize'].step = self._controller._widgetsExtraParams['timestepSize'].min
            self._controller._widgetsExtraParams['timestepSize'].readout_format = '.' + str(_count_sig_decimals(str(self._controller._widgetsExtraParams['timestepSize'].step))) + 'f'
        if self._controller._widgetsExtraParams.get('maxTime'):
            self._controller._widgetsExtraParams['maxTime'].description = "Simulation time (equivalent to " + str(maxTimeSteps) + " simulation timesteps)"
            self._controller._widgetsExtraParams['maxTime'].layout = widgets.Layout(width='70%')
        else:
            self._controller._widgetsExtraParams['timestepSize'].description = "Timestep size (total time is " + str(self._fixedParams['maxTime']) + " = " + str(maxTimeSteps) + " timesteps)"
            self._controller._widgetsExtraParams['timestepSize'].layout = widgets.Layout(width='70%')
                    
    def _initGraph(self):
        numNodes = sum(self._currentState.values())
        if (self._netType == NetworkType.FULLY_CONNECTED):
            #print("Generating full graph")
            self._graph = nx.complete_graph(numNodes)  # np.repeat(0, self.numNodes)
        elif (self._netType == NetworkType.ERSOS_RENYI):
            #print("Generating Erdos-Renyi graph (connected)")
            if self._netParam is not None and self._netParam > 0 and self._netParam <= 1: 
                self._graph = nx.erdos_renyi_graph(numNodes, self._netParam, np.random.randint(MAX_RANDOM_SEED))
                i = 0
                while (not nx.is_connected(self._graph)):
                    if i > 100000:
                        errorMsg = "ERROR! Invalid network parameter (link probability="+str(self._netParam)+") for E-R networks. After "+str(i)+" attempts of network initialisation, the network is never connected.\n"\
                               "Please increase the network parameter value."
                        print(errorMsg)
                        raise MuMoTValueError(errorMsg)
                    #print("Graph was not connected; Resampling!")
                    i = i+1
                    self._graph = nx.erdos_renyi_graph(numNodes, self._netParam, np.random.randint(MAX_RANDOM_SEED))
            else:
                errorMsg = "ERROR! Invalid network parameter (link probability) for E-R networks. It must be between 0 and 1; input is " + str(self._netParam) 
                print(errorMsg)
                raise MuMoTValueError(errorMsg)
        elif (self._netType == NetworkType.BARABASI_ALBERT):
            #print("Generating Barabasi-Albert graph")
            netParam = int(self._netParam)
            if netParam is not None and netParam > 0 and netParam <= numNodes: 
                self._graph = nx.barabasi_albert_graph(numNodes, netParam, np.random.randint(MAX_RANDOM_SEED))
            else:
                errorMsg = "ERROR! Invalid network parameter (number of edges per new node) for B-A networks. It must be an integer between 1 and " + str(numNodes) + "; input is " + str(self._netParam)
                print(errorMsg)
                raise MuMoTValueError(errorMsg)
        elif (self._netType == NetworkType.SPACE):
            ## @todo: implement network generate by placing points (with local communication range) randomly in 2D space
            errorMsg = "ERROR: Graphs of type SPACE are not implemented yet."
            print(errorMsg)
            raise MuMoTValueError(errorMsg)
        elif (self._netType == NetworkType.DYNAMIC):
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
            self._agents.extend([state]*pop)
        self._agents = np.random.permutation(self._agents).tolist()  # random shuffling of elements (useful to avoid initial clusters in networks)
        
        # init the positionHistory lists
        dynamicNetwork = self._netType == NetworkType.DYNAMIC
        if dynamicNetwork:
            self._positionHistory = []
            for _ in np.arange(sum(self._currentState.values())):
                self._positionHistory.append([])
        else:  # store the graph layout (only for 'graph' visualisation)
            self._positionHistory = nx.circular_layout(self._graph)
            
    def _simulationStep(self):
        tmp_agents = copy.deepcopy(self._agents)
        dynamic = self._netType == NetworkType.DYNAMIC
        if dynamic:
            tmp_positions = copy.deepcopy(self._positions)
            communication_range = self._netParam
            # store the position history
            for idx, _ in enumerate(self._agents):  # second element _ is the agent (unused)
                self._positionHistory[idx].append(self._positions[idx])
        children = []
        activeAgents = [True]*len(self._agents)
        #for idx, a in enumerate(self._agents):
        # to execute in random order the agents I just create a shuffled list of idx and I follow that
        indexes = np.arange(0, len(self._agents))
        indexes = np.random.permutation(indexes).tolist()  # shuffle the indexes
        for idx in indexes:
            a = self._agents[idx]
            # if moving-particles the agent moves
            if dynamic:
#                 print("Agent " + str(idx) + " moved from " + str(self._positions[idx]) )
                self._positions[idx] = self._updatePosition(self._positions[idx][0], self._positions[idx][1], self._positions[idx][2], self._particleSpeed, self._motionCorrelatedness)
#                 print("to position " + str(self._positions[idx]) )
            
            # the step is executed only if the agent is active
            if not activeAgents[idx]: continue
            
            # computing the list of neighbours for the given agent
            if dynamic:
                neighNodes = self._getNeighbours(idx, tmp_positions, communication_range)
            else:
                neighNodes = list(nx.all_neighbors(self._graph, idx))            
            neighNodes = np.random.permutation(neighNodes).tolist()  # random shuffling of neighNodes (to randomise interactions)
            neighAgents = [tmp_agents[x] for x in neighNodes]  # creating the list of neighbours' states 
            neighActive = [activeAgents[x] for x in neighNodes]  # creating the list of neighbour' activity-status

#                 print("Neighs of agent " + str(idx) + " are " + str(neighNodes) + " with states " + str(neighAgents) )
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
            idx = len(self._positions)-1
            self._positionHistory[idx].append(self._positions[idx])
            self._positions[idx] = (self._positions[idx][0], self._positions[idx][1], np.random.rand() * np.pi * 2.0)  # set random orientation 
#             self._positions[idx][2] = np.random.rand() * np.pi * 2.0 # set random orientation 
            self._positions[idx] = self._updatePosition(self._positions[idx][0], self._positions[idx][1], self._positions[idx][2], self._particleSpeed, self._motionCorrelatedness)

        # compute self birth (possible only for moving-particles view)
        for birth in self._mumotModel._agentProbabilities[EMPTYSET_SYMBOL]:
            birthRate = self._ratesDict[str(birth[1])] * self._timestepSize  # scale the rate
            decimal = birthRate % 1
            birthsNum = int(birthRate - decimal)
            np.random.rand()
            if (np.random.rand() < decimal): birthsNum += 1
            #print ( "Birth rate " + str(birth[1]) + " triggers " + str(birthsNum) + " newborns")
            for _ in range(birthsNum):
                for newborn in birth[2]:
                    self._agents.append(newborn)
                    self._positions.append((np.random.rand() * self._arena_width, np.random.rand() * self._arena_height, np.random.rand() * np.pi * 2.0))
                    self._positionHistory.append([])
                    self._positionHistory[len(self._positions)-1].append(self._positions[len(self._positions)-1])
        
        # Remove from lists (_agents, _positions, and _positionHistory) the 'dead' agents (possible only for moving-particles view)
        deads = [idx for idx, a in enumerate(self._agents) if a == EMPTYSET_SYMBOL]
#         print("Dead list is " + str(deads))
        for dead in reversed(deads):
            del self._agents[dead]
            del self._positions[dead]
            del self._positionHistory[dead]
            
        currentState = {state : self._agents.count(state) for state in self._initialState.keys()}  # if state not in self._mumotModel._constantReactants} #self._mumotModel._reactants | self._mumotModel._constantReactants}
        return (self._timestepSize, currentState)

    def _stepOneAgent(self, agent, neighs, activeNeighs):
        """One timestep for one agent."""
        rnd = np.random.rand()
        lastVal = 0
        neighChanges = [None]*len(neighs)
        # counting how many neighbours for each state (to be uses for the interaction probabilities)
        neighCount = {x: neighs.count(x) for x in self._initialState.keys()}  # self._mumotModel._reactants | self._mumotModel._constantReactants}
        for idx, neigh in enumerate(neighs):
            if not activeNeighs[idx]:
                neighCount[neigh] -= 1
#         print("Agent " + str(agent) + " with probSet=" + str(probSets))
#         print("nc:"+str(neighCount))
        for reaction in self._mumotModel._agentProbabilities[agent]:
            popScaling = 1
            rate = self._ratesDict[str(reaction[1])] * self._timestepSize  # scaling the rate by the timeStep size
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
        return (x, y, o)

    def _getNeighbours(self, agent, positions, distance_range):
        """Return the (index) list of neighbours of ``agent``."""
        neighbour_list = []
        for neigh in np.arange(len(positions)):
            if (not neigh == agent) and (self._distance_on_torus(positions[agent][0], positions[agent][1], positions[neigh][0], positions[neigh][1]) < distance_range):
                neighbour_list.append(neigh)
        return neighbour_list
    
    def _distance_on_torus(self, x_1, y_1, x_2, y_2):
        """Returns the minimum distance calucalted on the torus given by periodic boundary conditions."""
        return np.sqrt(min(abs(x_1 - x_2), self._arena_width - abs(x_1 - x_2))**2 + 
                    min(abs(y_1 - y_2), self._arena_height - abs(y_1 - y_2))**2)
    
    def _update_net_params(self, resetValueAndRange):
        """Update the widgets related to the netType 
        
        (it cannot be a :class:`MuMoTcontroller` method because with multi-controller it needs to point to the right ``_controller``)
        """
        # if netType is fixed, no update is necessary
        if self._fixedParams.get('netParam') is not None: return
        self._netType = self._fixedParams['netType'] if self._fixedParams.get('netType') is not None else self._controller._widgetsExtraParams['netType'].value
        
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
        if (self._netType == NetworkType.FULLY_CONNECTED):
#             self._controller._widgetsExtraParams['netParam'].min = 0
#             self._controller._widgetsExtraParams['netParam'].max = 1
#             self._controller._widgetsExtraParams['netParam'].step = 1
#             self._controller._widgetsExtraParams['netParam'].value = 0
#             self._controller._widgetsExtraParams['netParam'].disabled = True
#             self._controller._widgetsExtraParams['netParam'].description = "None"
            self._controller._widgetsExtraParams['netParam'].layout.display = 'none'
        elif (self._netType == NetworkType.ERSOS_RENYI):
#             self._controller._widgetsExtraParams['netParam'].disabled = False
            self._controller._widgetsExtraParams['netParam'].layout.display = 'flex'
            if resetValueAndRange:
                self._controller._widgetsExtraParams['netParam'].min = 0.1
                self._controller._widgetsExtraParams['netParam'].max = 1
                self._controller._widgetsExtraParams['netParam'].step = 0.1
                self._controller._widgetsExtraParams['netParam'].value = 0.5
            self._controller._widgetsExtraParams['netParam'].description = "Network connectivity parameter (link probability)"
        elif (self._netType == NetworkType.BARABASI_ALBERT):
#             self._controller._widgetsExtraParams['netParam'].disabled = False
            self._controller._widgetsExtraParams['netParam'].layout.display = 'flex'
            maxVal = self._systemSize-1
            if resetValueAndRange:
                self._controller._widgetsExtraParams['netParam'].min = 1
                self._controller._widgetsExtraParams['netParam'].max = maxVal
                self._controller._widgetsExtraParams['netParam'].step = 1
                self._controller._widgetsExtraParams['netParam'].value = min(maxVal, 3)
            self._controller._widgetsExtraParams['netParam'].description = "Network connectivity parameter (new edges)"            
        elif (self._netType == NetworkType.SPACE):
            self._controller._widgetsExtraParams['netParam'].value = -1
         
        if (self._netType == NetworkType.DYNAMIC):
#             self._controller._widgetsExtraParams['netParam'].disabled = False
            self._controller._widgetsExtraParams['netParam'].layout.display = 'flex'
            if resetValueAndRange:
                self._controller._widgetsExtraParams['netParam'].min = 0.0
                self._controller._widgetsExtraParams['netParam'].max = 1
                self._controller._widgetsExtraParams['netParam'].step = 0.05
                self._controller._widgetsExtraParams['netParam'].value = 0.1
            self._controller._widgetsExtraParams['netParam'].description = "Interaction range"
#             self._controller._widgetsExtraParams['particleSpeed'].disabled = False
#             self._controller._widgetsExtraParams['motionCorrelatedness'].disabled = False
#             self._controller._widgetsPlotOnly['showTrace'].disabled = False
#             self._controller._widgetsPlotOnly['showInteractions'].disabled = False
            if self._controller._widgetsExtraParams.get('particleSpeed') is not None:
                self._controller._widgetsExtraParams['particleSpeed'].layout.display = 'flex'
            if self._controller._widgetsExtraParams.get('motionCorrelatedness') is not None:
                self._controller._widgetsExtraParams['motionCorrelatedness'].layout.display = 'flex'
            if self._controller._widgetsPlotOnly.get('showTrace') is not None:
                self._controller._widgetsPlotOnly['showTrace'].layout.display = 'flex'
            if self._controller._widgetsPlotOnly.get('showInteractions') is not None:
                self._controller._widgetsPlotOnly['showInteractions'].layout.display = 'flex'
        else:
            #self._controller._widgetsExtraParams['particleSpeed'].disabled = True
#             self._controller._widgetsExtraParams['motionCorrelatedness'].disabled = True
#             self._controller._widgetsPlotOnly['showTrace'].disabled = True
#             self._controller._widgetsPlotOnly['showInteractions'].disabled = True
            if self._controller._widgetsExtraParams.get('particleSpeed') is not None:
                self._controller._widgetsExtraParams['particleSpeed'].layout.display = 'none'
            if self._controller._widgetsExtraParams.get('motionCorrelatedness') is not None:
                self._controller._widgetsExtraParams['motionCorrelatedness'].layout.display = 'none'
            if self._controller._widgetsPlotOnly.get('showTrace') is not None:
                self._controller._widgetsPlotOnly['showTrace'].layout.display = 'none'
            if self._controller._widgetsPlotOnly.get('showInteractions') is not None:
                self._controller._widgetsPlotOnly['showInteractions'].layout.display = 'none'
            
        self._controller._widgetsExtraParams['netParam'].readout_format = '.' + str(_count_sig_decimals(str(self._controller._widgetsExtraParams['netParam'].step))) + 'f'
        if toLinkPlotFunction:
            self._controller._widgetsExtraParams['netParam'].observe(self._controller._replotFunction, 'value')

    def _adjust_barabasi_network_range(self):
        """function to adjust the widget of the number of edges of the Barabasi-Albert network when the system size slider is changed"""
        if self._controller is None or not self._netType == NetworkType.BARABASI_ALBERT or self._controller._widgetsExtraParams.get('netParam') is None: return
        maxVal = self._systemSize-1
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

    ## a matrix form of the left-handside of the rules
    _reactantsMatrix = None 
    ## the effect of each rule
    _ruleChanges = None

    def _constructorSpecificParams(self, _):
        if self._controller is not None:
            self._generatingCommand = "SSA"  
    
    def _build_bookmark(self, includeParams=True):
        logStr = "bookmark = " if not self._silent else ""
        logStr += "<modelName>." + self._generatingCommand + "("
        if includeParams:
            logStr += self._get_bookmarks_params()
            logStr += ", "
        logStr = logStr.replace('\\', '\\\\')
        #initState_str = { self._mumotModel._reactantsLaTeX.get(str(state), str(state)): pop for state,pop in self._initialState.items() if not state in self._mumotModel._constantReactants}
        initState_str = {latex(state): pop for state, pop in self._initialState.items() if not state in self._mumotModel._constantReactants}
        logStr += "initialState = " + str(initState_str)
        logStr += ", maxTime = " + str(self._maxTime)
        logStr += ", randomSeed = " + str(self._randomSeed)
        logStr += ", visualisationType = '" + str(self._visualisationType) + "'"
        if self._visualisationType == 'final':
            # these loops are necessary to return the latex() format of the reactant 
            for reactant in self._mumotModel._getAllReactants()[0]:
                if str(reactant) == self._finalViewAxes[0]:
                    logStr += ", final_x = '" + latex(reactant).replace('\\', '\\\\') + "'"
                    break
            for reactant in self._mumotModel._getAllReactants()[0]:
                if str(reactant) == self._finalViewAxes[1]:
                    logStr += ", final_y = '" + latex(reactant).replace('\\', '\\\\') + "'"
                    break
        logStr += ", plotProportions = " + str(self._plotProportions)
        logStr += ", realtimePlot = " + str(self._realtimePlot)
        logStr += ", runs = " + str(self._runs)
        logStr += ", aggregateResults = " + str(self._aggregateResults)
        logStr += ", silent = " + str(self._silent)
        logStr += ", bookmark = False"
#         if len(self._generatingKwargs) > 0:
#             for key in self._generatingKwargs:
#                 if type(self._generatingKwargs[key]) == str:
#                     logStr += key + " = " + "\'"+ str(self._generatingKwargs[key]) + "\'" + ", "
#                 else:
#                     logStr += key + " = " + str(self._generatingKwargs[key]) + ", "
        logStr += ")"
        return logStr
    
    def _printStandaloneViewCmd(self):
        ssaParams = {}
#         initState_str = {}
#         for state,pop in self._initialState.items():
#             initState_str[str(state)] = pop
        ssaParams["initialState"] = {latex(state): pop for state, pop in self._initialState.items() if not state in self._mumotModel._constantReactants}
        ssaParams["maxTime"] = self._maxTime 
        ssaParams["randomSeed"] = self._randomSeed
        ssaParams["visualisationType"] = self._visualisationType
        if self._visualisationType == 'final':
            # this loop is necessary to return the latex() format of the reactant 
            for reactant in self._mumotModel._getAllReactants()[0]:
                if str(reactant) == self._finalViewAxes[0]: ssaParams['final_x'] = latex(reactant)
                if str(reactant) == self._finalViewAxes[1]: ssaParams['final_y'] = latex(reactant)
        ssaParams["plotProportions"] = self._plotProportions
        ssaParams['realtimePlot'] = self._realtimePlot
        ssaParams['runs'] = self._runs
        ssaParams['aggregateResults'] = self._aggregateResults
        #str( list(self._ratesDict.items()) )
        print("mumot.MuMoTSSAView(<modelName>, None, " + str(self._get_bookmarks_params().replace('\\', '\\\\')) + ", SSParams = " + str(ssaParams) + " )")
            
    def _simulationStep(self): 
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
                    prob *= self._currentState[reactant] * reactantOccurencies
                numReagents += reactantOccurencies
            if prob > 0 and numReagents > 1:
                prob /= sum(self._currentState.values())**(numReagents - 1) 
            probabilitiesOfChange[reaction_id] = prob
#         for rule in self._reactantsMatrix:
#             prob = sum([a*b for a,b in zip(rule,currentState)])
#             numReagents = sum(x > 0 for x in rule)
#             if numReagents > 1:
#                 prob /= sum(currentState)**( numReagents -1 ) 
#             probabilitiesOfChange.append(prob)
        probSum = sum(probabilitiesOfChange.values())
        if probSum == 0:  # no reaction are possible (the execution terminates with this population)
            infiniteTime = self._maxTime-self._t
            return (infiniteTime, self._currentState)
        # computing when is happening next reaction
        timeInterval = np.random.exponential(1/probSum)
        
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
#         print("current state: " + str(self._currentState))
#         print("triggered change: " + str(self._mumotModel._stoichiometry[reaction_id]))
        # apply the change
#         currentState = list(np.array(self._currentState) + np.array(self._ruleChanges[index]) )
        for reactant, re_stoch in self._mumotModel._stoichiometry[reaction_id].items():
            if reactant == 'rate' or re_stoch == 'const': continue
            self._currentState[reactant] += re_stoch[1] - re_stoch[0]
            if (self._currentState[reactant] < 0):
                print("ERROR! Population size became negative: " + str(self._currentState) + "; Error in the algorithm execution.")
                sys.exit()
        #print(self._currentState)
        
        return (timeInterval, self._currentState)
    

def parseModel(modelDescription):
    """Create model from text description.

    Parameters
    ----------
    modelDescription : str
        A reference to an input cell in a Jupyter Notebook (e.g. ``In[4]``)
        that uses the ``%%model`` cell magic and contains several rules e.g. ::

            %%model
            U -> A : g_A
            U -> B : g_B
            A -> U : a_A
            B -> U : a_B
            A + U -> A + A : r_A
            B + U -> B + B : r_B
            A + B -> A + U : s
            A + B -> B + U : s

        or a model expressed as a multi-line string comprised of rules e.g. ::

            '''
            U -> A : g_A
            U -> B : g_B
            A -> U : a_A
            B -> U : a_B
            A + U -> A + A : r_A
            B + U -> B + B : r_B
            A + B -> A + U : s
            A + B -> B + U : s
            '''

    Returns
    -------
    :class:`MuMoTmodel` or None (with warning)
        The instantiated MuMoT model.

    """
    # @todo: add system size to model description
    if "get_ipython" in modelDescription:
        # hack to extract model description from input cell tagged with %%model magic
        modelDescr = modelDescription.split("\"")[0].split("'")[5]
    elif "->" in modelDescription:
        # model description provided as string
        modelDescr = modelDescription
    else:
        # assume input describes filename and attempt to load
        print("Input does not appear to be valid model - attempting to load from file `" + modelDescription + "`...")
        raise MuMoTWarning("Loading from file not currently supported")

        return None

    # Strip out any basic LaTeX equation formatting user has input
    modelDescr = modelDescr.replace('$', '')
    modelDescr = modelDescr.replace(r'\\\\', '')
    # Add white space to make parsing easy
    modelDescr = modelDescr.replace('+', ' + ')
    modelDescr = modelDescr.replace('->', ' -> ')
    modelDescr = modelDescr.replace(':', ' : ')
    # Split rules line-by-line, one rule per line.
    # Delimiters can be: windows newline, unix newline, or latex-y newline.
    modelRules = [s.strip()
                  for s in re.split("\r\n|\n|\\\\n", modelDescr)
                  if not s.isspace()]
    # parse and construct the model
    reactants = set()
    constantReactants = set()
    rates = set()
    rules = []
    model = MuMoTmodel()

    for rule in modelRules:
        if len(rule) > 0:
            tokens = rule.split()
            reactantCount = 0
            constantReactantCount = 0
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
                token = token.replace("\\\\", '\\')
                if token in GREEK_LETT_RESERVED_LIST:
                    raise MuMoTSyntaxError("Reserved letter " + token + " encountered: the list of reserved letters is " + str(GREEK_LETT_RESERVED_LIST_PRINT))                
                constantReactant = False

                if state == 'A':
                    if token not in ("+", "->", ":"):
                        state = 'B'
                        if '^' in token:
                            raise MuMoTSyntaxError("Reactants cannot contain '^' :" + token + " in " + rule)
                        reactantCount += 1
                        if (token[0] == '(' and token[-1] == ')'):
                            constantReactant = True
                            token = token.replace('(', '')
                            token = token.replace(')', '')
                        if token == '\emptyset':
                            constantReactant = True
                            token = '1'
                        expr = process_sympy(token)
                        reactantAtoms = expr.atoms()
                        if len(reactantAtoms) != 1:
                            raise MuMoTSyntaxError("Non-singleton symbol set in token " + token + " in rule " + rule)
                        for reactant in reactantAtoms:
                            pass  # this loop extracts element from singleton set
                        if constantReactant:
                            constantReactantCount += 1
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
                            raise MuMoTSyntaxError("Reactants cannot contain '^' :" + token + " in " + rule)
                        reactantCount -= 1
                        if (token[0] == '(' and token[-1] == ')'):
                            constantReactant = True
                            token = token.replace('(', '')
                            token = token.replace(')', '')
                        if token == '\emptyset':
                            constantReactant = True
                            token = '1'
                        expr = process_sympy(token)
                        reactantAtoms = expr.atoms()
                        if len(reactantAtoms) != 1:
                            raise MuMoTSyntaxError("Non-singleton symbol set in token " + token + " in rule " + rule)
                        for reactant in reactantAtoms:
                            pass  # this loop extracts element from singleton set
                        if constantReactant:
                            constantReactantCount -= 1
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
                raise MuMoTSyntaxError("Unequal number of reactants on lhs and rhs of rule " + rule)

            if constantReactantCount != 0:
                model._constantSystemSize = False
            
    model._rules = rules
    model._reactants = reactants
    model._constantReactants = constantReactants
    # check intersection of reactants and constantReactants is empty
    intersect = model._reactants.intersection(model._constantReactants)
    if len(intersect) != 0:
        raise MuMoTSyntaxError("Following reactants defined as both constant and variable: " + str(intersect))
    model._rates = rates
    model._equations = _deriveODEsFromRules(model._reactants, model._rules)
    model._ratesLaTeX = {}
    rates = map(latex, list(model._rates))
    for (rate, latex_str) in zip(model._rates, rates):
        model._ratesLaTeX[repr(rate)] = latex_str
    constantReactants = map(latex, list(model._constantReactants))
    for (reactant, latexStr) in zip(model._constantReactants, constantReactants):
        #model._ratesLaTeX[repr(reactant)] = '(' + latexStr + ')'
        model._ratesLaTeX[repr(reactant)] = '\Phi_{' + latexStr + '}'
    
    model._stoichiometry = _getStoichiometry(model._rules, model._constantReactants)
    
    return model

def about():
    """Display version, author and documentation information
    """
    print("Multiscale Modelling Tool (MuMoT): Version " + __version__)
    print("Authors: James A. R. Marshall, Andreagiovanni Reina, Thomas Bose")
    print("Documentation: https://mumot.readthedocs.io/")

def setVerboseExceptions(verbose=True):
    """Set the verbosity of exception handling.

    Parameters
    ----------
    verbose : bool, optional
        Whether to show a exception traceback.  Defaults to True.

    """
    if verbose:
        ipython.showtraceback = _show_traceback
    else:
        ipython.showtraceback = _hide_traceback


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


def _getStoichiometry(rules, const_reactants):
    """Produce dictionary with stoichiometry of all reactions with key ReactionNr.

    ReactionNr represents another dictionary with reaction rate, reactants and their stoichiometry.
    """
    # @todo: shall this become a model method?
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


def _deriveMasterEquation(stoichiometry):
    """Derive the Master equation
    
    Returns dictionary used in :method:`MuMoTmodel.showMasterEquation`.
    """
    substring = None
    P, E_op, x, y, v, w, t, m = symbols('P E_op x y v w t m')
    V = Symbol('\overline{V}', real=True, constant=True)
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

    if len(nvec) < 1 or len(nvec) > 4:
        print("Derivation of Master Equation works for 1, 2, 3 or 4 different reactants only")
        
        return None, None
    
#    assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'
    
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
        
        if len(nvec) == 1:
            sol_dict_rhs[key1] = (prod1, simplify(prod2*V), P(nvec[0], t), stoich[key1]['rate']*rate_fact)
        elif len(nvec) == 2:
            sol_dict_rhs[key1] = (prod1, simplify(prod2*V), P(nvec[0], nvec[1], t), stoich[key1]['rate']*rate_fact)
        elif len(nvec) == 3:
            sol_dict_rhs[key1] = (prod1, simplify(prod2*V), P(nvec[0], nvec[1], nvec[2], t), stoich[key1]['rate']*rate_fact)
        else:
            sol_dict_rhs[key1] = (prod1, simplify(prod2*V), P(nvec[0], nvec[1], nvec[2], nvec[3], t), stoich[key1]['rate']*rate_fact)

    return sol_dict_rhs, substring


def _doVanKampenExpansion(rhs, stoich):
    """Return the left-hand side and right-hand side of van Kampen expansion."""
    P, E_op, x, y, v, w, t, m = symbols('P E_op x y v w t m')
    V = Symbol('\overline{V}', real=True, constant=True)
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
    if len(nvec) < 1 or len(nvec) > 4:
        print("van Kampen expansion works for 1, 2, 3 or 4 different reactants only")
        
        return None, None, None    
#    assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'
    
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
    
    
    if len(nvec) == 1:
        lhs_vKE = (Derivative(P(nvec[0], t), t).subs({nvec[0]: NoiseDict[nvec[0]]})
                  - sympy.sqrt(V)*Derivative(PhiDict[nvec[0]], t)*Derivative(P(nvec[0], t), nvec[0]).subs({nvec[0]: NoiseDict[nvec[0]]}))
        for key in rhs_dict:
            op = rhs_dict[key][0].subs({nvec[0]: NoiseDict[nvec[0]]})
            func1 = rhs_dict[key][1].subs({nvec[0]: V*PhiDict[nvec[0]]+sympy.sqrt(V)*NoiseDict[nvec[0]]})
            func2 = rhs_dict[key][2].subs({nvec[0]: NoiseDict[nvec[0]]})
            func = func1*func2
            #if len(op.args[0].args) ==0:
            term = (op*func).subs({op*func: func + op.args[1]/sympy.sqrt(V)*Derivative(func, op.args[0]) + op.args[1]**2/(2*V)*Derivative(func, op.args[0], op.args[0])})
            #    
            #else:
            #    term = (op.args[1]*func).subs({op.args[1]*func: func + op.args[1].args[1]/sympy.sqrt(V)*Derivative(func, op.args[1].args[0]) 
            #                           + op.args[1].args[1]**2/(2*V)*Derivative(func, op.args[1].args[0], op.args[1].args[0])})
            #    term = (op.args[0]*term).subs({op.args[0]*term: term + op.args[0].args[1]/sympy.sqrt(V)*Derivative(term, op.args[0].args[0]) 
            #                           + op.args[0].args[1]**2/(2*V)*Derivative(term, op.args[0].args[0], op.args[0].args[0])})
            rhs_vKE += rhs_dict[key][3].subs(PhiConstDict)*(term.doit() - func)
    elif len(nvec) == 2:
        lhs_vKE = (Derivative(P(nvec[0], nvec[1], t), t).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]]})
                  - sympy.sqrt(V)*Derivative(PhiDict[nvec[0]], t)*Derivative(P(nvec[0], nvec[1], t), nvec[0]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]]})
                  - sympy.sqrt(V)*Derivative(PhiDict[nvec[1]], t)*Derivative(P(nvec[0], nvec[1], t), nvec[1]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]]}))

        for key in rhs_dict:
            op = rhs_dict[key][0].subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]]})
            func1 = rhs_dict[key][1].subs({nvec[0]: V*PhiDict[nvec[0]]+sympy.sqrt(V)*NoiseDict[nvec[0]], nvec[1]: V*PhiDict[nvec[1]]+sympy.sqrt(V)*NoiseDict[nvec[1]]})
            func2 = rhs_dict[key][2].subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]]})
            func = func1*func2
            if len(op.args[0].args) == 0:
                term = (op*func).subs({op*func: func + op.args[1]/sympy.sqrt(V)*Derivative(func, op.args[0]) + op.args[1]**2/(2*V)*Derivative(func, op.args[0], op.args[0])})
                
            else:
                term = (op.args[1]*func).subs({op.args[1]*func: func + op.args[1].args[1]/sympy.sqrt(V)*Derivative(func, op.args[1].args[0]) 
                                       + op.args[1].args[1]**2/(2*V)*Derivative(func, op.args[1].args[0], op.args[1].args[0])})
                term = (op.args[0]*term).subs({op.args[0]*term: term + op.args[0].args[1]/sympy.sqrt(V)*Derivative(term, op.args[0].args[0]) 
                                       + op.args[0].args[1]**2/(2*V)*Derivative(term, op.args[0].args[0], op.args[0].args[0])})
            #term_num, term_denom = term.as_numer_denom()
            rhs_vKE += rhs_dict[key][3].subs(PhiConstDict)*(term.doit() - func)
    elif len(nvec) == 3:
        lhs_vKE = (Derivative(P(nvec[0], nvec[1], nvec[2], t), t).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]]})
                  - sympy.sqrt(V)*Derivative(PhiDict[nvec[0]], t)*Derivative(P(nvec[0], nvec[1], nvec[2], t), nvec[0]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]]})
                  - sympy.sqrt(V)*Derivative(PhiDict[nvec[1]], t)*Derivative(P(nvec[0], nvec[1], nvec[2], t), nvec[1]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]]})
                  - sympy.sqrt(V)*Derivative(PhiDict[nvec[2]], t)*Derivative(P(nvec[0], nvec[1], nvec[2], t), nvec[2]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]]}))
        rhs_dict, substring = rhs(stoich)
        rhs_vKE = 0
        for key in rhs_dict:
            op = rhs_dict[key][0].subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]]})
            func1 = rhs_dict[key][1].subs({nvec[0]: V*PhiDict[nvec[0]]+sympy.sqrt(V)*NoiseDict[nvec[0]], nvec[1]: V*PhiDict[nvec[1]]+sympy.sqrt(V)*NoiseDict[nvec[1]], nvec[2]: V*PhiDict[nvec[2]]+sympy.sqrt(V)*NoiseDict[nvec[2]]})
            func2 = rhs_dict[key][2].subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]]})
            func = func1*func2
            if len(op.args[0].args) == 0:
                term = (op*func).subs({op*func: func + op.args[1]/sympy.sqrt(V)*Derivative(func, op.args[0]) + op.args[1]**2/(2*V)*Derivative(func, op.args[0], op.args[0])})
                
            elif len(op.args) == 2:
                term = (op.args[1]*func).subs({op.args[1]*func: func + op.args[1].args[1]/sympy.sqrt(V)*Derivative(func, op.args[1].args[0]) 
                                       + op.args[1].args[1]**2/(2*V)*Derivative(func, op.args[1].args[0], op.args[1].args[0])})
                term = (op.args[0]*term).subs({op.args[0]*term: term + op.args[0].args[1]/sympy.sqrt(V)*Derivative(term, op.args[0].args[0]) 
                                       + op.args[0].args[1]**2/(2*V)*Derivative(term, op.args[0].args[0], op.args[0].args[0])})
            elif len(op.args) == 3:
                term = (op.args[2]*func).subs({op.args[2]*func: func + op.args[2].args[1]/sympy.sqrt(V)*Derivative(func, op.args[2].args[0]) 
                                       + op.args[2].args[1]**2/(2*V)*Derivative(func, op.args[2].args[0], op.args[2].args[0])})
                term = (op.args[1]*term).subs({op.args[1]*term: term + op.args[1].args[1]/sympy.sqrt(V)*Derivative(term, op.args[1].args[0]) 
                                       + op.args[1].args[1]**2/(2*V)*Derivative(term, op.args[1].args[0], op.args[1].args[0])})
                term = (op.args[0]*term).subs({op.args[0]*term: term + op.args[0].args[1]/sympy.sqrt(V)*Derivative(term, op.args[0].args[0]) 
                                       + op.args[0].args[1]**2/(2*V)*Derivative(term, op.args[0].args[0], op.args[0].args[0])})
            else:
                print('Something went wrong!')
            rhs_vKE += rhs_dict[key][3].subs(PhiConstDict)*(term.doit() - func)    
    else:
        lhs_vKE = (Derivative(P(nvec[0], nvec[1], nvec[2], nvec[3], t), t).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]], nvec[3]: NoiseDict[nvec[3]]})
                  - sympy.sqrt(V)*Derivative(PhiDict[nvec[0]], t)*Derivative(P(nvec[0], nvec[1], nvec[2], nvec[3], t), nvec[0]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]], nvec[3]: NoiseDict[nvec[3]]})
                  - sympy.sqrt(V)*Derivative(PhiDict[nvec[1]], t)*Derivative(P(nvec[0], nvec[1], nvec[2], nvec[3], t), nvec[1]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]], nvec[3]: NoiseDict[nvec[3]]})
                  - sympy.sqrt(V)*Derivative(PhiDict[nvec[2]], t)*Derivative(P(nvec[0], nvec[1], nvec[2], nvec[3], t), nvec[2]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]], nvec[3]: NoiseDict[nvec[3]]})
                  - sympy.sqrt(V)*Derivative(PhiDict[nvec[3]], t)*Derivative(P(nvec[0], nvec[1], nvec[2], nvec[3], t), nvec[3]).subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]], nvec[3]: NoiseDict[nvec[3]]}))
        rhs_dict, substring = rhs(stoich)
        rhs_vKE = 0
        for key in rhs_dict:
            op = rhs_dict[key][0].subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]], nvec[3]: NoiseDict[nvec[3]]})
            func1 = rhs_dict[key][1].subs({nvec[0]: V*PhiDict[nvec[0]]+sympy.sqrt(V)*NoiseDict[nvec[0]], nvec[1]: V*PhiDict[nvec[1]]+sympy.sqrt(V)*NoiseDict[nvec[1]], nvec[2]: V*PhiDict[nvec[2]]+sympy.sqrt(V)*NoiseDict[nvec[2]], nvec[3]: V*PhiDict[nvec[3]]+sympy.sqrt(V)*NoiseDict[nvec[3]]})
            func2 = rhs_dict[key][2].subs({nvec[0]: NoiseDict[nvec[0]], nvec[1]: NoiseDict[nvec[1]], nvec[2]: NoiseDict[nvec[2]], nvec[3]: NoiseDict[nvec[3]]})
            func = func1*func2
            if len(op.args[0].args) == 0:
                term = (op*func).subs({op*func: func + op.args[1]/sympy.sqrt(V)*Derivative(func, op.args[0]) + op.args[1]**2/(2*V)*Derivative(func, op.args[0], op.args[0])})
                
            elif len(op.args) == 2:
                term = (op.args[1]*func).subs({op.args[1]*func: func + op.args[1].args[1]/sympy.sqrt(V)*Derivative(func, op.args[1].args[0]) 
                                       + op.args[1].args[1]**2/(2*V)*Derivative(func, op.args[1].args[0], op.args[1].args[0])})
                term = (op.args[0]*term).subs({op.args[0]*term: term + op.args[0].args[1]/sympy.sqrt(V)*Derivative(term, op.args[0].args[0]) 
                                       + op.args[0].args[1]**2/(2*V)*Derivative(term, op.args[0].args[0], op.args[0].args[0])})
            elif len(op.args) == 3:
                term = (op.args[2]*func).subs({op.args[2]*func: func + op.args[2].args[1]/sympy.sqrt(V)*Derivative(func, op.args[2].args[0]) 
                                       + op.args[2].args[1]**2/(2*V)*Derivative(func, op.args[2].args[0], op.args[2].args[0])})
                term = (op.args[1]*term).subs({op.args[1]*term: term + op.args[1].args[1]/sympy.sqrt(V)*Derivative(term, op.args[1].args[0]) 
                                       + op.args[1].args[1]**2/(2*V)*Derivative(term, op.args[1].args[0], op.args[1].args[0])})
                term = (op.args[0]*term).subs({op.args[0]*term: term + op.args[0].args[1]/sympy.sqrt(V)*Derivative(term, op.args[0].args[0]) 
                                       + op.args[0].args[1]**2/(2*V)*Derivative(term, op.args[0].args[0], op.args[0].args[0])})
            elif len(op.args) == 4:
                term = (op.args[3]*func).subs({op.args[3]*func: func + op.args[3].args[1]/sympy.sqrt(V)*Derivative(func, op.args[3].args[0]) 
                                       + op.args[3].args[1]**2/(2*V)*Derivative(func, op.args[3].args[0], op.args[3].args[0])})
                term = (op.args[2]*term).subs({op.args[2]*term: term + op.args[2].args[1]/sympy.sqrt(V)*Derivative(term, op.args[2].args[0]) 
                                       + op.args[2].args[1]**2/(2*V)*Derivative(term, op.args[2].args[0], op.args[2].args[0])})
                term = (op.args[1]*term).subs({op.args[1]*term: term + op.args[1].args[1]/sympy.sqrt(V)*Derivative(term, op.args[1].args[0]) 
                                       + op.args[1].args[1]**2/(2*V)*Derivative(term, op.args[1].args[0], op.args[1].args[0])})
                term = (op.args[0]*term).subs({op.args[0]*term: term + op.args[0].args[1]/sympy.sqrt(V)*Derivative(term, op.args[0].args[0]) 
                                       + op.args[0].args[1]**2/(2*V)*Derivative(term, op.args[0].args[0], op.args[0].args[0])})
            else:
                print('Something went wrong!')
            rhs_vKE += rhs_dict[key][3].subs(PhiConstDict)*(term.doit() - func)
    
    return rhs_vKE.expand(), lhs_vKE, substring


# def _get_orderedLists_vKE( _getStoichiometry,rules):
def _get_orderedLists_vKE(stoich):
    """Create list of dictionaries where the key is the system size order."""
    V = Symbol('\overline{V}', real=True, constant=True)
    stoichiometry = stoich
    rhs_vke, lhs_vke, substring = _doVanKampenExpansion(_deriveMasterEquation, stoichiometry)
    Vlist_lhs = []
    Vlist_rhs = []
    for jj in range(len(rhs_vke.args)):
        try:
            Vlist_rhs.append(simplify(rhs_vke.args[jj]).collect(V, evaluate=False))
        except NotImplementedError:
            prod = 1
            for nn in range(len(rhs_vke.args[jj].args)-1):
                prod *= rhs_vke.args[jj].args[nn]
            tempdict = prod.collect(V, evaluate=False)
            for key in tempdict:
                Vlist_rhs.append({key: prod/key*rhs_vke.args[jj].args[-1]})
    
    for jj in range(len(lhs_vke.args)):
        try:
            Vlist_lhs.append(simplify(lhs_vke.args[jj]).collect(V, evaluate=False))
        except NotImplementedError:
            prod = 1
            for nn in range(len(lhs_vke.args[jj].args)-1):
                prod *= lhs_vke.args[jj].args[nn]
            tempdict = prod.collect(V, evaluate=False)
            for key in tempdict:
                Vlist_lhs.append({key: prod/key*lhs_vke.args[jj].args[-1]})
    return Vlist_lhs, Vlist_rhs, substring


def _getFokkerPlanckEquation(_get_orderedLists_vKE, stoich):
    """Return the Fokker-Planck equation."""
    P, t = symbols('P t')
    V = Symbol('\overline{V}', real=True, constant=True)
    Vlist_lhs, Vlist_rhs, substring = _get_orderedLists_vKE(stoich)
    rhsFPE = 0
    lhsFPE = 0
    for kk in range(len(Vlist_rhs)):
        for key in Vlist_rhs[kk]:
            if key == 1:
                rhsFPE += Vlist_rhs[kk][key]  
    for kk in range(len(Vlist_lhs)):
        for key in Vlist_lhs[kk]:
            if key == 1:
                lhsFPE += Vlist_lhs[kk][key]            
        
    FPE = lhsFPE-rhsFPE
    
    nvec = []
    for key1 in stoich:
        for key2 in stoich[key1]:
            if key2 != 'rate' and stoich[key1][key2] != 'const':
                if not key2 in nvec:
                    nvec.append(key2)
    nvec = sorted(nvec, key=default_sort_key)
    if len(nvec) < 1 or len(nvec) > 4:
        print("Derivation of Fokker Planck equation works for 1, 2, 3 or 4 different reactants only")
        
        return None, None
#    assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'
    
    NoiseDict = {}
    for kk in range(len(nvec)):
        NoiseDict[nvec[kk]] = Symbol('eta_'+str(nvec[kk]))
    
    if len(Vlist_lhs)-1 == 1:
        SOL_FPE = solve(FPE, Derivative(P(NoiseDict[nvec[0]], t), t), dict=True)[0]
    elif len(Vlist_lhs)-1 == 2:
        SOL_FPE = solve(FPE, Derivative(P(NoiseDict[nvec[0]], NoiseDict[nvec[1]], t), t), dict=True)[0]
    elif len(Vlist_lhs)-1 == 3:
        SOL_FPE = solve(FPE, Derivative(P(NoiseDict[nvec[0]], NoiseDict[nvec[1]], NoiseDict[nvec[2]], t), t), dict=True)[0]
    elif len(Vlist_lhs)-1 == 4:
        SOL_FPE = solve(FPE, Derivative(P(NoiseDict[nvec[0]], NoiseDict[nvec[1]], NoiseDict[nvec[2]], NoiseDict[nvec[3]], t), t), dict=True)[0]
    else:
        print('Not implemented yet.')
           
    return SOL_FPE, substring


def _getNoiseEOM(_getFokkerPlanckEquation, _get_orderedLists_vKE, stoich):
    """Calculates noise in the system.

    Returns equations of motion for noise.

    """
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
    if len(nvec) < 1 or len(nvec) > 4:
        print("showNoiseEquations works for 1, 2, 3 or 4 different reactants only")
        
        return
#    assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'

    NoiseDict = {}
    for kk in range(len(nvec)):
        NoiseDict[nvec[kk]] = Symbol('eta_'+str(nvec[kk]))
    FPEdict, substring = _getFokkerPlanckEquation(_get_orderedLists_vKE, stoich)
    
    NoiseSub1stOrder = {}
    NoiseSub2ndOrder = {}
    
    if len(NoiseDict) == 1:
        Pdim = P(NoiseDict[nvec[0]], t)
    elif len(NoiseDict) == 2:
        Pdim = P(NoiseDict[nvec[0]], NoiseDict[nvec[1]], t)
    elif len(NoiseDict) == 3:
        Pdim = P(NoiseDict[nvec[0]], NoiseDict[nvec[1]], NoiseDict[nvec[2]], t)
    else:
        Pdim = P(NoiseDict[nvec[0]], NoiseDict[nvec[1]], NoiseDict[nvec[2]], NoiseDict[nvec[3]], t)
        
    for noise1 in NoiseDict:
        NoiseSub1stOrder[NoiseDict[noise1]*Pdim] = M_1(NoiseDict[noise1])
        for noise2 in NoiseDict:
            for noise3 in NoiseDict:
                key = NoiseDict[noise1]*NoiseDict[noise2]*Derivative(Pdim, NoiseDict[noise3])
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
                key2 = NoiseDict[noise1]*Derivative(Pdim, NoiseDict[noise2], NoiseDict[noise3])
                if key2 not in NoiseSub1stOrder:
                    NoiseSub1stOrder[key2] = 0   
    
    for noise1 in NoiseDict:
        for noise2 in NoiseDict:
            key = NoiseDict[noise1]*NoiseDict[noise2]*Pdim
            if key not in NoiseSub2ndOrder:
                NoiseSub2ndOrder[key] = M_2(NoiseDict[noise1]*NoiseDict[noise2])
            for noise3 in NoiseDict:
                for noise4 in NoiseDict:
                    key2 = NoiseDict[noise1]*NoiseDict[noise2]*NoiseDict[noise3]*Derivative(Pdim, NoiseDict[noise4])
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
                    key3 = NoiseDict[noise1]*NoiseDict[noise2]*Derivative(Pdim, NoiseDict[noise3], NoiseDict[noise4])
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
            if len(NoiseDict) == 1:
                eq1stOrderMoment = collect(eq1stOrderMoment, M_1(NoiseDict[nvec[0]]))
            elif len(NoiseDict) == 2:
                eq1stOrderMoment = collect(eq1stOrderMoment, M_1(NoiseDict[nvec[0]]))
                eq1stOrderMoment = collect(eq1stOrderMoment, M_1(NoiseDict[nvec[1]]))
            elif len(NoiseDict) == 3:
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
                NoiseSubs1stOrder[M_1(NoiseDict[noise])] = r'\left< \vphantom{Dg}\right.' + latex(NoiseDict[noise]) + r'\left. \vphantom{Dg}\right>'
    
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
                if len(NoiseDict) == 1:
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]]))
                elif len(NoiseDict) == 2:
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[1]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]]))
                elif len(NoiseDict) == 3:
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
                    NoiseSubs2ndOrder[M_2(NoiseDict[noise1]*NoiseDict[noise2])] = r'\left< \vphantom{Dg}\right.' + latex(NoiseDict[noise1]*NoiseDict[noise2]) + r'\left. \vphantom{Dg}\right>' 
      
    return EQsys1stOrdMom, EOM_1stOrderMom, NoiseSubs1stOrder, EQsys2ndOrdMom, EOM_2ndOrderMom, NoiseSubs2ndOrder 


def _getNoiseStationarySol(_getNoiseEOM, _getFokkerPlanckEquation, _get_orderedLists_vKE, stoich):
    """Calculate noise in the system.

    Returns analytical solution for stationary noise.

    """
    P, M_1, M_2, t = symbols('P M_1 M_2 t')
    
    EQsys1stOrdMom, EOM_1stOrderMom, NoiseSubs1stOrder, EQsys2ndOrdMom, EOM_2ndOrderMom, NoiseSubs2ndOrder = _getNoiseEOM(_getFokkerPlanckEquation, _get_orderedLists_vKE, stoich)
    
    nvec = []
    for key1 in stoich:
        for key2 in stoich[key1]:
            if key2 != 'rate' and stoich[key1][key2] != 'const':
                if not key2 in nvec:
                    nvec.append(key2)
    nvec = sorted(nvec, key=default_sort_key)
    if len(nvec) < 1 or len(nvec) > 4:
        print("showNoiseSolutions works for 1, 2, 3 or 4 different reactants only")
        
        return
#    assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'

    NoiseDict = {}
    for kk in range(len(nvec)):
        NoiseDict[nvec[kk]] = Symbol('eta_'+str(nvec[kk]))
    if len(NoiseDict) == 1:
        SOL_1stOrderMom = solve(EQsys1stOrdMom, [M_1(NoiseDict[nvec[0]])], dict=True)        
    elif len(NoiseDict) == 2:
        SOL_1stOrderMom = solve(EQsys1stOrdMom, [M_1(NoiseDict[nvec[0]]), M_1(NoiseDict[nvec[1]])], dict=True)
    elif len(NoiseDict) == 3:
        SOL_1stOrderMom = solve(EQsys1stOrdMom, [M_1(NoiseDict[nvec[0]]), M_1(NoiseDict[nvec[1]]), M_1(NoiseDict[nvec[2]])], dict=True)
    else:
        SOL_1stOrderMom = solve(EQsys1stOrdMom, [M_1(NoiseDict[nvec[0]]), M_1(NoiseDict[nvec[1]]), M_1(NoiseDict[nvec[2]]), M_1(NoiseDict[nvec[3]])], dict=True)
    
    if len(SOL_1stOrderMom[0]) != len(NoiseDict):
        print('Solution for 1st order noise moments NOT unique!')
        return None, None, None, None
                    
    SOL_2ndOrdMomDict = {} 
    
    if len(NoiseDict) == 1:
        SOL_2ndOrderMom = list(linsolve(EQsys2ndOrdMom, [M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]])]))[0]  # only one set of solutions (if any) in linear system of equations        
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]])] = SOL_2ndOrderMom[0]
    
    elif len(NoiseDict) == 2:
        SOL_2ndOrderMom = list(linsolve(EQsys2ndOrdMom, [M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]]), 
                                                         M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[1]]), 
                                                         M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]])]))[0]  # only one set of solutions (if any) in linear system of equations
        
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
    
    elif len(NoiseDict) == 3:
        SOL_2ndOrderMom = list(linsolve(EQsys2ndOrdMom, [M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[0]]), 
                                                         M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[1]]), 
                                                         M_2(NoiseDict[nvec[2]]*NoiseDict[nvec[2]]), 
                                                         M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[1]]), 
                                                         M_2(NoiseDict[nvec[0]]*NoiseDict[nvec[2]]), 
                                                         M_2(NoiseDict[nvec[1]]*NoiseDict[nvec[2]])]))[0]  # only one set of solutions (if any) in linear system of equations; hence index [0]
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
                                                         M_2(NoiseDict[nvec[2]]*NoiseDict[nvec[3]])]))[0]  # only one set of solutions (if any) in linear system of equations; hence index [0]
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
 

    

def _getODEs_vKE(_get_orderedLists_vKE, stoich):
    """Return the ODE system derived from Master equation."""
    P, t = symbols('P t')
    V = Symbol('\overline{V}', real=True, constant=True)
    Vlist_lhs, Vlist_rhs, substring = _get_orderedLists_vKE(stoich)
    rhsODE = 0
    lhsODE = 0
    for kk in range(len(Vlist_rhs)):
        for key in Vlist_rhs[kk]:
            if key == sympy.sqrt(V):
                rhsODE += Vlist_rhs[kk][key]            
    for kk in range(len(Vlist_lhs)):
        for key in Vlist_lhs[kk]:
            if key == sympy.sqrt(V):
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
    if len(nvec) < 1 or len(nvec) > 4:
        print("van Kampen expansions works for 1, 2, 3 or 4 different reactants only")
        
        return
#    assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'
    
    PhiDict = {}
    NoiseDict = {}
    for kk in range(len(nvec)):
        NoiseDict[nvec[kk]] = Symbol('eta_'+str(nvec[kk]))
        PhiDict[nvec[kk]] = Symbol('Phi_'+str(nvec[kk]))
        
    PhiSubDict = None    
    if not substring is None:
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
                    PhiSubDict[key] = PhiSubDict[key].subs({sym: 1})  # here we assume that if a reactant in the substitution string is not a time-dependent reactant it can only be the total number of reactants which is constant, i.e. 1=N/N
    
    
    if len(Vlist_lhs)-1 == 1:
        ode1 = 0
        for kk in range(len(ODE.args)):
            prod = 1
            for nn in range(len(ODE.args[kk].args)-1):
                prod *= ODE.args[kk].args[nn]
            if ODE.args[kk].args[-1] == Derivative(P(NoiseDict[nvec[0]], t), NoiseDict[nvec[0]]):
                ode1 += prod
            else:
                print('Check ODE.args!')
                
        if PhiSubDict:
            ode1 = ode1.subs(PhiSubDict)
            
            for key in PhiSubDict:
                ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]] , t), dict=True)
                ODEsys = {**ODE_1[0]}            
        else:        
            ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]] , t), dict=True)
            ODEsys = {**ODE_1[0]}
    
    elif len(Vlist_lhs)-1 == 2:
        ode1 = 0
        ode2 = 0
        for kk in range(len(ODE.args)):
            prod = 1
            for nn in range(len(ODE.args[kk].args)-1):
                prod *= ODE.args[kk].args[nn]
            if ODE.args[kk].args[-1] == Derivative(P(NoiseDict[nvec[0]], NoiseDict[nvec[1]], t), NoiseDict[nvec[0]]):
                ode1 += prod
            elif ODE.args[kk].args[-1] == Derivative(P(NoiseDict[nvec[0]], NoiseDict[nvec[1]], t), NoiseDict[nvec[1]]):
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
            prod = 1
            for nn in range(len(ODE.args[kk].args)-1):
                prod *= ODE.args[kk].args[nn]
            if ODE.args[kk].args[-1] == Derivative(P(NoiseDict[nvec[0]], NoiseDict[nvec[1]], NoiseDict[nvec[2]], t), NoiseDict[nvec[0]]):
                ode1 += prod
            elif ODE.args[kk].args[-1] == Derivative(P(NoiseDict[nvec[0]], NoiseDict[nvec[1]], NoiseDict[nvec[2]], t), NoiseDict[nvec[1]]):
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
            prod = 1
            for nn in range(len(ODE.args[kk].args)-1):
                prod *= ODE.args[kk].args[nn]
            if ODE.args[kk].args[-1] == Derivative(P(NoiseDict[nvec[0]], NoiseDict[nvec[1]], NoiseDict[nvec[2]], NoiseDict[nvec[3]], t), NoiseDict[nvec[0]]):
                ode1 += prod
            elif ODE.args[kk].args[-1] == Derivative(P(NoiseDict[nvec[0]], NoiseDict[nvec[1]], NoiseDict[nvec[2]], NoiseDict[nvec[3]], t), NoiseDict[nvec[1]]):
                ode2 += prod
            elif ODE.args[kk].args[-1] == Derivative(P(NoiseDict[nvec[0]], NoiseDict[nvec[1]], NoiseDict[nvec[2]], NoiseDict[nvec[3]], t), NoiseDict[nvec[2]]):
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
            expr = process_sympy(name.replace('\\\\', '\\'))
            atoms = expr.atoms()
            if len(atoms) > 1:
                raise MuMoTSyntaxError("Non-singleton parameter name in parameter " + name)
            for atom in atoms:
                # parameter name should contain a single atom
                pass
            paramsRet.append(atom)
            
    return (paramsRet, paramValues)


def _raiseModelError(expected, read, rule):
    raise MuMoTSyntaxError("Expected " + expected + " but read '" + read + "' in rule: " + rule)


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
    with warnings.catch_warnings():  # ignore warnings when plt.hold has been deprecated in installed libraries - still need to try plt.hold(True) in case older libraries in use
        warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        plt.hold(True)  
    if figure is None:
        if figureCounter > 2:
            plt.ion()
        object._figure = plt.figure(object._figureNum) 
    else:
        object._figure = figure


def _buildFigOLD(object, figure=None):
    """Generic function for constructing figures in :class:`MuMoTview` and :class:`MuMoTmultiController` classes."""
    global figureCounter
    object._figureNum = figureCounter
    figureCounter += 1
    plt.ion()
    with warnings.catch_warnings():  # ignore warnings when plt.hold has been deprecated in installed libraries - still need to try plt.hold(True) in case older libraries in use
        warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        plt.hold(True)  
    if figure is None:
        object._figure = plt.figure(object._figureNum) 
    else:
        object._figure = figure

def _round_to_1(x):
    '''used for determining significant digits for axes formatting in plots MuMoTstreamView and MuMoTbifurcationView.'''
    if x == 0: return 1
    return round(x, -int(floor(log10(abs(x)))))

def _fig_formatting_3D(figure, xlab=None, ylab=None, zlab=None, ax_reformat=False, 
                       specialPoints=None, showFixedPoints=False, **kwargs):
    """Function for editing properties of 3D plots. 

    Called by :class:`MuMoTvectorView`.

    """
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
    if ax_reformat == False:
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
            xMLocator_major = _round_to_1(max_xrange/4)
        else:
            xMLocator_major = _round_to_1(max_xrange/6)
        #xMLocator_minor = xMLocator_major/2
        if max_yrange < 1.0:
            yMLocator_major = _round_to_1(max_yrange/4)
        else:
            yMLocator_major = _round_to_1(max_yrange/6)
        #yMLocator_minor = yMLocator_major/2
        if max_zrange < 1.0:
            zMLocator_major = _round_to_1(max_zrange/4)
        else:
            zMLocator_major = _round_to_1(max_zrange/6)
        #zMLocator_minor = yMLocator_major/2
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xMLocator_major))
        #ax.xaxis.set_minor_locator(ticker.MultipleLocator(xMLocator_minor))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(yMLocator_major))
        #ax.yaxis.set_minor_locator(ticker.MultipleLocator(yMLocator_minor))
        ax.zaxis.set_major_locator(ticker.MultipleLocator(zMLocator_major))
        #ax.zaxis.set_minor_locator(ticker.MultipleLocator(zMLocator_minor))
        
    ax.set_xlim3d(x_lim_left, x_lim_right)
    ax.set_ylim3d(y_lim_bot, y_lim_top)
    if kwargs.get('showPlane', False) == True:
        ax.set_zlim3d(0, z_lim_top)
    else:
        ax.set_zlim3d(z_lim_bot, z_lim_top)
                    
    if showFixedPoints == True:
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
        chooseFontSize = kwargs['fontsize']
    elif len(xlabelstr) > 40 or len(ylabelstr) > 40 or len(zlabelstr) > 40:
        chooseFontSize = 12
    elif 31 <= len(xlabelstr) <= 40 or 31 <= len(ylabelstr) <= 40 or 31 <= len(zlabelstr) <= 40:
        chooseFontSize = 16
    elif 26 <= len(xlabelstr) <= 30 or 26 <= len(ylabelstr) <= 30 or 26 <= len(zlabelstr) <= 30:
        chooseFontSize = 22
    else:
        chooseFontSize = 26
    
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = 20
    ax.set_xlabel(r''+str(xlabelstr), fontsize=chooseFontSize)
    ax.set_ylabel(r''+str(ylabelstr), fontsize=chooseFontSize)
    if len(str(zlabelstr)) > 1:
        ax.set_zlabel(r''+str(zlabelstr), fontsize=chooseFontSize, rotation=90)
    else:
        ax.set_zlabel(r''+str(zlabelstr), fontsize=chooseFontSize)
        
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
                       xlab=None, ylab=None, curvelab=None, aspectRatioEqual=False, line_color_list=LINE_COLOR_LIST, 
                       **kwargs):
    """Format 2D plots.

    Called by :class:`MuMoTvectorView`, :class:`MuMoTstreamView` and :class:`MuMoTbifurcationView`

    """
    showLegend = kwargs.get('showLegend', False)
    
    linestyle_list = ['solid', 'dashed', 'dashdot', 'dotted', 'solid', 'dashed', 'dashdot', 'dotted', 'solid']
    
    if xdata and ydata:
        if len(xdata) == len(ydata):
            #plt.figure(figsize=(8,6), dpi=80)
            #ax = plt.axes()
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
    #print(data_x)
    
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
    
    if ax_reformat == False and figure is not None:
        xmajortickslocs = ax.xaxis.get_majorticklocs()
        xminortickslocs = ax.xaxis.get_minorticklocs()
        ymajortickslocs = ax.yaxis.get_majorticklocs()
        yminortickslocs = ax.yaxis.get_minorticklocs()
        x_lim_left = ax.get_xbound()[0]  # ax.xaxis.get_data_interval()[0]
        x_lim_right = ax.get_xbound()[1]  # ax.xaxis.get_data_interval()[1]
        y_lim_bot = ax.get_ybound()[0]  # ax.yaxis.get_data_interval()[0]
        y_lim_top = ax.get_ybound()[1]  # ax.yaxis.get_data_interval()[1]
        #print(ax.yaxis.get_data_interval())
        
    if curve_replot == True:
        plt.cla()
    
    if ax_reformat == False and figure is not None:
        ax.set_xticks(xmajortickslocs)
        ax.set_xticks(xminortickslocs, minor=True)
        ax.set_yticks(ymajortickslocs)
        ax.set_yticks(yminortickslocs, minor=True)
        ax.tick_params(axis='both', which='major', length=5, width=2)
        ax.tick_params(axis='both', which='minor', length=3, width=1)
        plt.xlim(x_lim_left, x_lim_right)
        plt.ylim(y_lim_bot, y_lim_top)
        
    if figure is None or curve_replot == True:

        if 'LineThickness' in kwargs:
            LineThickness = kwargs['LineThickness']
        else:
            LineThickness = 4
        
        if eigenvalues:
            showLegend = False
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
                #sign_change=0
                for kk in range(len(eigenvalues[nn])):
                    if kk > 0:
                        if len(eigenvalues[0][0]) == 1:
                            if (np.sign(np.round(np.real(eigenvalues[nn][kk][0]), round_digit))*np.sign(np.round(np.real(eigenvalues[nn][kk-1][0]), round_digit)) <= 0):
                                #print('sign change')
                                #sign_change+=1
                                #print(sign_change)
                                #if specialPoints is not None and specialPoints[0]!=[]:
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
                                
                                data_x_tmp_first = data_x_tmp[-1]
                                data_y_tmp_first = data_y_tmp[-1]
                                nr_sol_stab = 0
                                nr_sol_saddle = 0
                                nr_sol_unst = 0
                                data_x_tmp = []
                                data_y_tmp = []
                                data_x_tmp.append(data_x_tmp_first)
                                data_y_tmp.append(data_y_tmp_first)
                                #if specialPoints is not None and specialPoints[0]!=[]:
                                #    data_x_tmp.append(specialPoints[0][sign_change-1])
                                #    data_y_tmp.append(specialPoints[1][sign_change-1])
                        elif len(eigenvalues[0][0]) == 2:
                            if (np.sign(np.round(np.real(eigenvalues[nn][kk][0]), round_digit))*np.sign(np.round(np.real(eigenvalues[nn][kk-1][0]), round_digit)) <= 0
                                or np.sign(np.round(np.real(eigenvalues[nn][kk][1]), round_digit))*np.sign(np.round(np.real(eigenvalues[nn][kk-1][1]), round_digit)) <= 0):
                                #print('sign change')
                                #sign_change+=1
                                #print(sign_change)
                                #if specialPoints is not None and specialPoints[0]!=[]:
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
                                
                                data_x_tmp_first = data_x_tmp[-1]
                                data_y_tmp_first = data_y_tmp[-1]
                                nr_sol_stab = 0
                                nr_sol_saddle = 0
                                nr_sol_unst = 0
                                data_x_tmp = []
                                data_y_tmp = []
                                data_x_tmp.append(data_x_tmp_first)
                                data_y_tmp.append(data_y_tmp_first)
                                #if specialPoints is not None and specialPoints[0]!=[]:
                                #    data_x_tmp.append(specialPoints[0][sign_change-1])
                                #    data_y_tmp.append(specialPoints[1][sign_change-1])
                    
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
                    plt.plot(data_x[nn], data_y[nn], c=line_color_list[nn], 
                             ls=linestyle_list[nn], lw=LineThickness, label=r''+str(curvelab[nn]))
                except:
                    plt.plot(data_x[nn], data_y[nn], c=line_color_list[nn], 
                             ls=linestyle_list[nn], lw=LineThickness)
        
        
    if len(xlabelstr) > 40 or len(ylabelstr) > 40:
        chooseFontSize = 10  # 16
    elif 31 <= len(xlabelstr) <= 40 or 31 <= len(ylabelstr) <= 40:
        chooseFontSize = 14  # 20
    elif 26 <= len(xlabelstr) <= 30 or 26 <= len(ylabelstr) <= 30:
        chooseFontSize = 18  # 26
    else:
        chooseFontSize = 24  # 30
        
    if 'fontsize' in kwargs:
        if not kwargs['fontsize'] is None:
            chooseFontSize = kwargs['fontsize']

    plt.xlabel(r''+str(xlabelstr), fontsize=chooseFontSize)
    plt.ylabel(r''+str(ylabelstr), fontsize=chooseFontSize)
    #ax.set_xlabel(r''+str(xlabelstr), fontsize = chooseFontSize)
    #ax.set_ylabel(r''+str(ylabelstr), fontsize = chooseFontSize)
     
    if figure is None or ax_reformat == True or choose_xrange is not None or choose_yrange is not None:
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
            xMLocator_major = _round_to_1(max_xrange/5)
        else:
            xMLocator_major = _round_to_1(max_xrange/10)
        xMLocator_minor = xMLocator_major/2
        if max_yrange < 1.0:
            yMLocator_major = _round_to_1(max_yrange/5)
        else:
            yMLocator_major = _round_to_1(max_yrange/10)
        yMLocator_minor = yMLocator_major/2
        
        if choose_xrange:
            plt.xlim(choose_xrange[0]-xMLocator_minor/10.0, choose_xrange[1]+xMLocator_minor/10.0)
        else:
            plt.xlim(XaxisMin-xMLocator_minor/10.0, XaxisMax+xMLocator_minor/10.0)
        if choose_yrange:
            plt.ylim(choose_yrange[0]-yMLocator_minor/10.0, choose_yrange[1]+yMLocator_minor/10.0)
        else:
            plt.ylim(YaxisMin-yMLocator_minor/10.0, YaxisMax+yMLocator_minor/10.0)

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
                    if a > plt.xlim()[0]+(plt.xlim()[1]-plt.xlim()[0])/2:
                        x_offset = -(plt.xlim()[1]-plt.xlim()[0])*0.02
                    else:
                        x_offset = (plt.xlim()[1]-plt.xlim()[0])*0.02
                    if b > plt.ylim()[0]+(plt.ylim()[1]-plt.ylim()[0])/2:
                        y_offset = -(plt.ylim()[1]-plt.ylim()[0])*0.05
                    else:
                        y_offset = (plt.ylim()[1]-plt.ylim()[0])*0.05
                    plt.text(a+x_offset, b+y_offset, c, fontsize=18)
    
    if showFixedPoints == True:
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
        
    if curvelab is not None or showLegend == True:
        #if 'legend_loc' in kwargs:
        #    legend_loc = kwargs['legend_loc']
        #else:
        #    legend_loc = 'upper left'
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
    

def _decodeNetworkTypeFromString(netTypeStr):
    # init the network type
    admissibleNetTypes = {'full': NetworkType.FULLY_CONNECTED, 
                          'erdos-renyi': NetworkType.ERSOS_RENYI, 
                          'barabasi-albert': NetworkType.BARABASI_ALBERT, 
                          'dynamic': NetworkType.DYNAMIC}
    
    if netTypeStr not in admissibleNetTypes:
        print("ERROR! Invalid network type argument! Valid strings are: " + str(admissibleNetTypes))
    return admissibleNetTypes.get(netTypeStr, None)


def _encodeNetworkTypeToString(netType):
    # init the network type
    netTypeEncoding = {NetworkType.FULLY_CONNECTED: 'full', 
                          NetworkType.ERSOS_RENYI: 'erdos-renyi', 
                          NetworkType.BARABASI_ALBERT: 'barabasi-albert', 
                          NetworkType.DYNAMIC: 'dynamic'}
    
    if netType not in netTypeEncoding:
        print("ERROR! Invalid netTypeEncoding table! Tryed to encode network type: " + str(netType))
    return netTypeEncoding.get(netType, 'none')


def _make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)
    return my_autopct


def _almostEqual(a, b):
    epsilon = 0.0000001
    return abs(a-b) < epsilon


def _plot_point_cov(points, nstd=2, ax=None, **kwargs):
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


def _plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
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


def _count_sig_decimals(digits, maximum=7):
    """Return the number of significant decimals of the input digit string (up to a maximum of 7). """
    _, _, fractional = digits.partition(".")

    if fractional:
        return min(len(fractional), maximum)
    else:
        return 0


def _parse_input_keyword_for_numeric_widgets(inputValue, defaultValueRangeStep, initValueRangeStep, validRange=None, onlyValue=False):
    """Parse an input keyword and set initial range and default values (when the input is a slider-widget).

    Check if the fixed value is not None, otherwise it returns the default value (samewise for ``initRange`` and ``defaultRange``).
    The optional parameter ``validRange`` is use to check if the fixedValue has a usable value.
    If the ``defaultValue`` is out of the ``initRange``, the default value is move to the closest of the initRange extremes.

    Parameters
    ----------
    inputValue : object
        if not ``None`` it indicated the fixed value to use
    defaultValueRangeStep : list of object
        Default set of values in the format ``[val,min,max,step]``
    initValueRangeStep : list of object
        User-specified set of values in the format ``[val,min,max,step]``
    validRange : list of object, optional 
        The min and max accepted values ``[min,max]``
    onlyValue : bool, optional
        If ``True`` then ``defaultValueRangeStep`` and ``initValueRangeStep`` are only a single value

    Returns
    -------
    values : list of object 
        Contains a list of five items (start-value, min-value, max-value,
        step-size, fixed). if onlyValue, it's only two items (start-value,
        fixed). The item fixed is a bool. If ``True`` the value is fixed (partial
        controller active), if ``False`` the widget will be created. 

    """
    outputValues = defaultValueRangeStep if not onlyValue else [defaultValueRangeStep]
    if onlyValue == False:
        if initValueRangeStep is not None and getattr(initValueRangeStep, "__getitem__", None) is None:
            errorMsg = "initValueRangeStep value '" + str(initValueRangeStep) + "' must be specified in the format [val,min,max,step].\n" \
                    "Please, correct the value and retry."
            print(errorMsg)
            raise MuMoTValueError(errorMsg)
    if not inputValue is None:
        if not isinstance(inputValue, numbers.Number):
            errorMsg = "Input value '" + str(inputValue) + "' is not a numeric vaule and must be a number.\n" \
                    "Please, correct the value and retry."
            print(errorMsg)
            raise MuMoTValueError(errorMsg)
        elif validRange and (inputValue < validRange[0] or inputValue > validRange[1]):
            errorMsg = "Input value '" + str(inputValue) + "' has raised out-of-range exception. Valid range is " + str(validRange) + "\n" \
                    "Please, correct the value and retry."
            print(errorMsg)
            raise MuMoTValueError(errorMsg)
        else:
            if onlyValue:
                return [inputValue, True]
            else:
                outputValues[0] = inputValue
                outputValues.append(True)
                # it is not necessary to modify the values [min,max,step] because when last value is True, they should be ignored
                return outputValues
    
    if not initValueRangeStep is None:
        if onlyValue:
            if validRange and (initValueRangeStep < validRange[0] or initValueRangeStep > validRange[1]):
                errorMsg = "Invalid init value=" + str(initValueRangeStep) + ". has raised out-of-range exception. Valid range is " + str(validRange) + "\n" \
                            "Please, correct the value and retry."
                print(errorMsg)
                raise MuMoTValueError(errorMsg)
            else:
                outputValues = [initValueRangeStep]
        else:
            if initValueRangeStep[1] > initValueRangeStep[2] or initValueRangeStep[0] < initValueRangeStep[1] or initValueRangeStep[0] > initValueRangeStep[2]:
                errorMsg = "Invalid init range [val,min,max,step]=" + str(initValueRangeStep) + ". Value must be within min and max values.\n"\
                            "Please, correct the value and retry."
                print(errorMsg)
                raise MuMoTValueError(errorMsg)
            elif validRange and (initValueRangeStep[1] < validRange[0] or initValueRangeStep[2] > validRange[1]):
                errorMsg = "Invalid init range [val,min,max,step]=" + str(initValueRangeStep) + ". has raised out-of-range exception. Valid range is " + str(validRange) + "\n" \
                            "Please, correct the value and retry."
                print(errorMsg)
                raise MuMoTValueError(errorMsg)
            else:
                outputValues = initValueRangeStep
    
    outputValues.append(False)
    return outputValues


def _parse_input_keyword_for_boolean_widgets(inputValue, defaultValue, initValue=None, paramNameForErrorMsg=None):
    """Parse an input keyword and set initial range and default values (when the input is a boolean checkbox)
    check if the fixed value is not None, otherwise it returns the default value.

    Parameters
    ----------
    inputValue : object
        If not None it indicates the fixed value to use
    defaultValue : bool
        dafault boolean value 

    Returns
    -------
    value : object
        The keyword value; 'fixed' is a boolean. 
    fixed : bool 
        If True the value is fixed (partial controller active); if False the widget will be created. 
    
    """
    if inputValue is not None:
        if not isinstance(inputValue, bool):  # terminating the process if the input argument is wrong
            paramNameForErrorMsg = "for " + str(paramNameForErrorMsg) + " = " if paramNameForErrorMsg else ""
            errorMsg = "The specified value " + paramNameForErrorMsg + "'" + str(inputValue) + "' is not valid. \n" \
                        "The value must be a boolean True/False."
            print(errorMsg)
            raise MuMoTValueError(errorMsg)
        return [inputValue, True]
    else:
        if isinstance(initValue, bool):
            return [initValue, False]
        else:  
            return [defaultValue, False]


def _get_item_from_params_list(params, targetName):
    """Params is a list (rather than a dictionary) and this method is necessary to fetch the value by name. """
    for param in params:
        if param[0] == targetName or param[0].replace('\\', '') == targetName or param[0].replace('_','_{') + '}' == targetName:
            return param[1]
    return None


def _format_advanced_option(optionName, inputValue, initValues, extraParam=None, extraParam2=None):
    """Check if the user-specified values are within valid range (appropriate subfunctions are called depending on the parameter).

    parameters for slider widgets return list of length 5 as [value, min, max, step, fixed]

    parameters for boolean, dropbox, or input fields return list of lenght two as [value, fixed]

    values is the initial value, (min,max,step) are for sliders, and fixed is a boolean that indciates if the parameter is fixed or the widget should be displayed

    """
    if (optionName == 'initialState'):
        (allReactants, _) = extraParam
        initialState = {}
        # handle initialState dictionary (either convert or generate a default one)
        if inputValue is not None:
            for i, reactant in enumerate(sorted(inputValue.keys(), key=str)):
                pop = inputValue[reactant]
                initPop = initValues.get(reactant) if initValues is not None else None
                
                # convert string into SymPy symbol
                initialState[process_sympy(reactant)] = _parse_input_keyword_for_numeric_widgets(inputValue=pop,
                                    defaultValueRangeStep=[MuMoTdefault._agents, MuMoTdefault._agentsLimits[0], MuMoTdefault._agentsLimits[1], MuMoTdefault._agentsStep], 
                                    initValueRangeStep=initPop, 
                                    validRange=(0.0, 1.0))
                fixedBool = True
        else:
            first = True
            initValuesSympy = {process_sympy(reactant): pop for reactant, pop in initValues.items()} if initValues is not None else {}
            for i, reactant in enumerate(sorted(allReactants, key=str)):
                defaultV = MuMoTdefault._agents if first else 0
                first = False
                initialState[reactant] = _parse_input_keyword_for_numeric_widgets(inputValue=None,
                                            defaultValueRangeStep=[defaultV, MuMoTdefault._agentsLimits[0], MuMoTdefault._agentsLimits[1], MuMoTdefault._agentsStep], 
                                            initValueRangeStep=initValuesSympy.get(reactant), 
                                            validRange=(0.0, 1.0))
                fixedBool = False
        
        ## check if the initialState values are valid
        sumValues = sum([initialState[reactant][0] for reactant in allReactants])
        minStep = min([initialState[reactant][3] for reactant in allReactants])     
        for i, reactant in enumerate(sorted(allReactants, key=str)):
            if reactant not in allReactants:
                errorMsg = "Reactant '" + str(reactant) + "' does not exist in this model.\n" \
                    "Valid reactants are " + str(allReactants) + ". Please, correct the value and retry."
                print(errorMsg)
                raise MuMoTValueError(errorMsg) 
            
            pop = initialState[reactant]
            # check if the proportions sum to 1 
            if i == 0:
                idleReactant = reactant
                idleValue = pop[0]
                # the idleValue have range min-max reset to [0,1]
                initialState[reactant][1] = 0 
                initialState[reactant][2] = 1
                initialState[reactant][3] = minStep
            else:
                # modify (if necessary) the initial value
                if sumValues > 1:
                    newVal = max(0, pop[0]+(1-sumValues))
                    if not _almostEqual(pop[0], newVal):
                        wrnMsg = "WARNING! the initial value of reactant " + str(reactant) + " has been changed to " + str(newVal) + "\n"
                        print(wrnMsg)
                        sumValues -= pop[0]
                        sumValues += newVal
                        initialState[reactant][0] = newVal
                # modify (if necessary) min-max
                pop = initialState[reactant]
                sumNorm = sumValues if sumValues <= 1 else 1
                if pop[2] > (1-sumNorm+pop[0]+idleValue):  # max
                    if pop[1] > (1-sumNorm+pop[0]+idleValue):  # min
                        initialState[reactant][1] = (1-sumNorm+pop[0]+idleValue)
                    initialState[reactant][2] = (1-sumNorm+pop[0]+idleValue)
                if pop[1] > (1-sumNorm+pop[0]):  # min
                    initialState[reactant][1] = (1-sumNorm+pop[0])
                #initialState[reactant][3] = minStep
        if not _almostEqual(sumValues, 1):
            newVal = 1-sum([initialState[reactant][0] for reactant in allReactants if reactant != idleReactant])
            wrnMsg = "WARNING! the initial value of reactant " + str(idleReactant) + " has been changed to " + str(newVal) + "\n"
            print(wrnMsg)
            initialState[idleReactant][0] = newVal      
        return [initialState, fixedBool]
        #print("Initial State is " + str(initialState) )
    if (optionName == 'maxTime'):
        return _parse_input_keyword_for_numeric_widgets(inputValue=inputValue,
                                    defaultValueRangeStep=[2, 0.1, 3, 0.1] if extraParam is 'asNoise' else [MuMoTdefault._maxTime, MuMoTdefault._timeLimits[0], MuMoTdefault._timeLimits[1], MuMoTdefault._timeStep], 
                                    initValueRangeStep=initValues, 
                                    validRange=(0, float("inf")))
    if (optionName == 'randomSeed'):
        return _parse_input_keyword_for_numeric_widgets(inputValue=inputValue,
                                    defaultValueRangeStep=np.random.randint(MAX_RANDOM_SEED), 
                                    initValueRangeStep=initValues,
                                    validRange=(1, MAX_RANDOM_SEED), onlyValue=True)
    if (optionName == 'motionCorrelatedness'):
        return _parse_input_keyword_for_numeric_widgets(inputValue=inputValue,
                                defaultValueRangeStep=[0.5, 0.0, 1.0, 0.05], 
                                initValueRangeStep=initValues, 
                                validRange=(0, 1)) 
    if (optionName == 'particleSpeed'):
        return _parse_input_keyword_for_numeric_widgets(inputValue=inputValue,
                                defaultValueRangeStep=[0.01, 0.0, 0.1, 0.005], 
                                initValueRangeStep=initValues, 
                                validRange=(0, 1)) 
    
    if (optionName == 'timestepSize'):
        return _parse_input_keyword_for_numeric_widgets(inputValue=inputValue,
                                defaultValueRangeStep=[1, 0.01, 1, 0.01], 
                                initValueRangeStep=initValues, 
                                validRange=(0, float("inf"))) 

    if (optionName == 'netType'):
        # check validity of the network type or init to default
        if inputValue is not None:
            decodedNetType = _decodeNetworkTypeFromString(inputValue)
            if decodedNetType is None:  # terminating the process if the input argument is wrong
                errorMsg = "The specified value for netType =" + str(inputValue) + " is not valid. \n" \
                            "Accepted values are: 'full',  'erdos-renyi', 'barabasi-albert', and 'dynamic'."
                print(errorMsg)
                raise MuMoTValueError(errorMsg)
                    
            return [inputValue, True]
        else:
            decodedNetType = _decodeNetworkTypeFromString(initValues) if initValues is not None else None
            if decodedNetType is not None:  # assigning the init value only if it's a valid value
                return [initValues, False] 
            else:
                return ['full', False]  # as default netType is set to 'full'  
    # @todo: avoid that these value will be overwritten by _update_net_params()
    if (optionName == 'netParam'):
        netType = extraParam
        systemSize = extraParam2
        # if netType is not fixed, netParam cannot be fixed
        if (not netType[-1]) and inputValue is not None:
            errorMsg = "If netType is not fixed, netParam cannot be fixed. Either leave free to widget the 'netParam' or fix the 'netType'."
            print(errorMsg)
            raise MuMoTValueError(errorMsg)
        # check if netParam range is valid or set the correct default range (systemSize is necessary) 
        if _decodeNetworkTypeFromString(netType[0]) == NetworkType.FULLY_CONNECTED:
            return [0, 0, 0, False]
        elif _decodeNetworkTypeFromString(netType[0]) == NetworkType.ERSOS_RENYI:
            return _parse_input_keyword_for_numeric_widgets(inputValue=inputValue,
                                        defaultValueRangeStep=[0.1, 0.1, 1, 0.1], 
                                        initValueRangeStep=initValues, 
                                         validRange=(0.1, 1.0))          
        elif _decodeNetworkTypeFromString(netType[0]) == NetworkType.BARABASI_ALBERT:
            maxEdges = systemSize - 1 
            return _parse_input_keyword_for_numeric_widgets(inputValue=inputValue,
                                        defaultValueRangeStep=[min(maxEdges, 3), 1, maxEdges, 1], 
                                        initValueRangeStep=initValues, 
                                         validRange=(1, maxEdges))  
        elif _decodeNetworkTypeFromString(netType[0]) == NetworkType.SPACE:
            pass  # method is not implemented
        elif _decodeNetworkTypeFromString(netType[0]) == NetworkType.DYNAMIC:
            return _parse_input_keyword_for_numeric_widgets(inputValue=inputValue,
                                        defaultValueRangeStep=[0.1, 0.0, 1.0, 0.05], 
                                        initValueRangeStep=initValues, 
                                         validRange=(0, 1.0)) 
        
        return _parse_input_keyword_for_numeric_widgets(inputValue=inputValue,
                                defaultValueRangeStep=[0.5, 0, 1, 0.1], 
                                initValueRangeStep=initValues, 
                                validRange=(0, float("inf"))) 
    if (optionName == 'plotProportions'):
        return _parse_input_keyword_for_boolean_widgets(inputValue=inputValue,
                                                                           defaultValue=False,
                                                                           initValue=initValues,
                                                                           paramNameForErrorMsg=optionName) 
    if (optionName == 'realtimePlot'):
        return _parse_input_keyword_for_boolean_widgets(inputValue=inputValue,
                                                                           defaultValue=False, 
                                                                           initValue=initValues,
                                                                           paramNameForErrorMsg=optionName)  
    if (optionName == 'showTrace'):
        return _parse_input_keyword_for_boolean_widgets(inputValue=inputValue,
                                                                           defaultValue=False,
                                                                           initValue=initValues, 
                                                                           paramNameForErrorMsg=optionName) 
    if (optionName == 'showInteractions'):
        return _parse_input_keyword_for_boolean_widgets(inputValue=inputValue,
                                                                           defaultValue=False, 
                                                                           initValue=initValues,
                                                                           paramNameForErrorMsg=optionName)  
    if (optionName == 'visualisationType'):
        if extraParam is not None:
            if extraParam == 'multiagent':
                validVisualisationTypes = ['evo', 'graph', 'final', 'barplot']
            elif extraParam == "SSA":
                validVisualisationTypes = ['evo', 'final', 'barplot']
            elif extraParam == "multicontroller":
                validVisualisationTypes = ['evo', 'final']
        else:
            validVisualisationTypes = ['evo', 'graph', 'final']
        if inputValue is not None:
            if inputValue not in validVisualisationTypes:  # terminating the process if the input argument is wrong
                errorMsg = "The specified value for visualisationType = " + str(inputValue) + " is not valid. \n" \
                            "Valid values are: " + str(validVisualisationTypes) + ". Please correct it and retry."
                print(errorMsg)
                raise MuMoTValueError(errorMsg)
            return [inputValue, True]
        else:
            if initValues in validVisualisationTypes:
                return [initValues, False]
            else: 
                return ['evo', False]  # as default visualisationType is set to 'evo'
    
    if (optionName == 'final_x') or (optionName == 'final_y'):
        reactants_str = [str(reactant) for reactant in sorted(extraParam, key=str)]
        if inputValue is not None:
            inputValue = inputValue.replace('\\', '')
            if inputValue not in reactants_str:
                errorMsg = "The specified value for " + optionName + " = " + str(inputValue) + " is not valid. \n" \
                            "Valid values are the reactants: " + str(reactants_str) + ". Please correct it and retry."
                print(errorMsg)
                raise MuMoTValueError(errorMsg)
            else:
                return [inputValue, True]
        else:
            if initValues is not None: initValues = initValues.replace('\\', '')
            if initValues in reactants_str:
                return [initValues, False]
            else: 
                if optionName == 'final_x' or len(reactants_str) == 1:
                    return [reactants_str[0], False]  # as default final_x is set to the first (sorted) reactant
                else: 
                    return [reactants_str[1], False]  # as default final_y is set to the second (sorted) reactant
                
    if (optionName == 'runs'):
        return _parse_input_keyword_for_numeric_widgets(inputValue=inputValue,
                                defaultValueRangeStep=[20, 5, 100, 5] if extraParam is 'asNoise' else [1, 1, 20, 1], 
                                initValueRangeStep=initValues,
                                validRange=(1, float("inf"))) 
    
    if (optionName == 'aggregateResults'):
        return _parse_input_keyword_for_boolean_widgets(inputValue=inputValue,
                                                           defaultValue=True, 
                                                           initValue=initValues,
                                                           paramNameForErrorMsg=optionName)  
        
    if (optionName == 'initBifParam'):
        return _parse_input_keyword_for_numeric_widgets(inputValue=inputValue,
                                    defaultValueRangeStep=[MuMoTdefault._initialRateValue, MuMoTdefault._rateLimits[0], MuMoTdefault._rateLimits[1], MuMoTdefault._rateStep], 
                                    initValueRangeStep=initValues, 
                                    validRange=(0, float("inf")))
        
    
    return [None, False]  # default output for unknown optionName


def _doubleUnderscorify(s):
    """Set underscores in expressions which need two indices to enable proper LaTeX rendering. """
    ind_list = [kk for kk, char in enumerate(s) if char == '_' and s[kk+1] != '{']
    if len(ind_list) == 0:
        return s
    else:
        index_MinCharLength = 1
        index_MaxCharLength_init = 20
        s_list = list(s)
        
        for ind in ind_list:
            ind_diff = len(s_list)-1-ind
            if ind_diff > 5:
                index_MaxCharLength = min(index_MaxCharLength_init, ind_diff-5)
                # the following requires that indices consist of 1 or 2 charcter(s) only
                for nn in range(4+index_MinCharLength, 5+index_MaxCharLength):
                    if s_list[ind+nn] == '}' and s_list[ind+nn+1] != '}':
                        s_list[ind] = '_{'
                        s_list[ind+nn] = '}}'
                        break
        
    return ''.join(s_list)


def _greekPrependify(s):
    """Prepend two backslash symbols in front of Greek letters to enable proper LaTeX rendering. """
    for nn in range(len(GREEK_LETT_LIST_1)):
        if 'eta' in s:
            s = _greekReplace(s, 'eta', '\\eta')
        if GREEK_LETT_LIST_1[nn] in s:
            s = _greekReplace(s, GREEK_LETT_LIST_1[nn], GREEK_LETT_LIST_2[nn]) 
            #if s[s.find(GREEK_LETT_LIST_1[nn]+'_')-1] !='\\':
            #    s = s.replace(GREEK_LETT_LIST_1[nn]+'_',GREEK_LETT_LIST_2[nn]+'_')
    return s


def _greekReplace(s, sub, repl):
    """ Auxiliary function for _greekPrependify() """
    # if find_index is not minus1 we have found at least one match for the substring
    find_index = s.find(sub)
    # loop util we find no (more) match
    while find_index != -1:
        if find_index == 0 or (s[find_index-1] != '\\' and not s[find_index-1].isalpha()):
            if sub != 'eta':
                s = s[:find_index]+repl+s[find_index + len(sub):]
            else:
                if s[find_index-1] != 'b' and s[find_index-1] != 'z':
                    if s[find_index-1] != 'h':
                        s = s[:find_index]+repl+s[find_index + len(sub):]
                    elif s[find_index-1] == 'h':
                        if s[find_index-2] != 't' and s[find_index-2] != 'T':
                            s = s[:find_index]+repl+s[find_index + len(sub):]
        # find + 1 means we start at the last match start index + 1
        find_index = s.find(sub, find_index + 1)
    return s


def _roundNumLogsOut(number):
    """ Round numerical output in Logs to 3 decimal places. """
    # if number is complex
    if type(number) == sympy.Add:
        return str(sympy.re(number).round(4)) + str(sympy.im(number).round(4)) + 'j'
    # if number is real
    else:
        return str(number.round(4))
