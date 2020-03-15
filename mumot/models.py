"""MuMoT model classes."""

import copy
import sympy
import os
import re
import tempfile

from graphviz import Digraph
from IPython.display import display, Math
import numpy as np
from sympy import (
    collect,
    default_sort_key,
    Derivative,
    lambdify,
    latex,
    linsolve,
    numbered_symbols,
    preview,
    simplify,
    solve,
    Symbol,
    symbols,
    Function
)
from sympy.parsing.latex import parse_latex
from warnings import warn

from . import (
    controllers,
    defaults,
    consts,
    exceptions,
    utils,
    views,
)


class MuMoTmodel:
    """Model class."""
    # list of rules
    _rules = None
    # set of reactants
    _reactants = None
    # set of fixed-concentration reactants (boundary conditions)
    _constantReactants = None
    # parameter that determines system size, set by using substitute()
    _systemSize = None
    # is system size constant or not?
    _constantSystemSize = None
    # list of LaTeX strings describing reactants (@todo: deprecated?)
    _reactantsLaTeX = None
    # set of rates
    _rates = None
    # dictionary of LaTeX strings describing rates and constant reactants (@todo: rename)
    _ratesLaTeX = None
    # dictionary of ODE righthand sides with reactant as key
    _equations = None
    # set of solutions to equations
    _solutions = None
    # summary of stoichiometry as nested dictionaries
    _stoichiometry = None
    # dictionary (reagents as keys) with reaction-rate and relative effects of each reaction-rule for each reagent (structure necessary for multiagent simulations)
    _agentProbabilities = None
    # dictionary of lambdified functions for integration, plotting, etc.
    _funcs = None
    # tuple of argument symbols for lambdified functions
    _args = None
    # graphviz visualisation of model
    _dot = None
    # image format used for rendering edge labels for model visualisation
    _renderImageFormat = 'png'
    # local path for creation of temporary storage
    _tmpdirpath = '__mumot_files__'
    # temporary storage for image files, etc. used in visualising model
    _tmpdir = None
    # list of temporary files created
    _tmpfiles = None

    def substitute(self, subsString: str):
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
        size_set_expr = None
        sizeSetKosher = False
        subs = []
        subsStrings = subsString.split(',')
        for subString in subsStrings:
            if '=' not in subString:
                raise exceptions.MuMoTSyntaxError("No '=' in assignment " + subString)
            assignment = parse_latex(subString)
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
                raise exceptions.MuMoTSyntaxError(f"Using substitute to rename reactants not supported: {sub[0]} = {sub[1]}")

        # Creating a new stoichiometry dictionary for every reaction by substituting keys and values when necessary
        for reaction in newModel._stoichiometry:
            for sub in subs:
                newModel._stoichiometry[reaction]['rate'] = newModel._stoichiometry[reaction]['rate'].subs(sub[0], sub[1])
                new_stoichiometry_dict = {}
                for stoich_key, stoich_value in newModel._stoichiometry[reaction].items():
                    new_st_key = stoich_key
                    new_st_value = stoich_value
                    if (stoich_key != 'rate') and stoich_key == sub[0]:  # check if substitution is necessary
                        if '+' not in str(sub[1]) and '-' not in str(sub[1]):  # substitute key
                            new_st_key = sub[1]
                        else:  # substitute value
                            new_st_value.append({stoich_key: sub[1]})
                    new_stoichiometry_dict[new_st_key] = new_st_value
                newModel._stoichiometry[reaction] = new_stoichiometry_dict

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
                            size_set_expr = sub[1] - sub[0]
                            if sub[0] in newModel._reactants:
                                sizeSetKosher = True
                        else:
                            raise exceptions.MuMoTSyntaxError(f"More than one unknown reactant encountered when trying to set system size: {sub[0]} = {sub[1]}")
                    else:
                        sizeSetrhs.append(atom)
                if newModel._systemSize is None:
                    raise exceptions.MuMoTSyntaxError(f"Expected to find system size parameter but failed: {sub[0]} = {sub[1]}")
                # @todo: more thorough error checking for valid system size expression
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
            # newModel._ratesLaTeX[repr(reactant)] = '(' + latexStr + ')'
            newModel._ratesLaTeX[repr(reactant)] = r'\Phi_{' + latexStr + '}'
        if sizeSet:
            # need to check setting system size was done correctly
            candidate_expr = newModel._systemSize
            for reactant in self._equations:
                # check all reactants in original model present in substitution string (first check, to help users explicitly realise they must inlude all reactants)
                if sizeSetKosher and reactant != sub[0] and reactant not in sizeSetrhs:
                    raise exceptions.MuMoTSyntaxError(f"Expected to find reactant {reactant} but failed: {sub[0]} = {sub[1]}")
                candidate_expr = candidate_expr - reactant
            # if reactant not in sizeSetrhs:
            # check substitution format is correct
            diff_expr = candidate_expr - size_set_expr
            if diff_expr != 0:
                raise exceptions.MuMoTSyntaxError(f"System size not set by expression of form <reactant> = <system size> - <reactants>: difference = {diff_expr}")

        # @todo: what else should be copied to new model?

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
                latexrep = '(' + self._ratesLaTeX[repr(reactant)].replace(r'\Phi_{', '').replace('}', '') + ')'
                dot.node(str(reactant), " ", image=self._localLaTeXimageFile(Symbol(latexrep)))
            for rule in self._rules:
                # render LaTeX representation of rule
                latexrep = '$$' + utils._doubleUnderscorify(utils._greekPrependify(self._ratesLaTeX.get(repr(rule.rate), repr(rule.rate)))) + '$$'
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
                out = utils._doubleUnderscorify(utils._greekPrependify(out))
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
            out = utils._doubleUnderscorify(utils._greekPrependify(reactant))
            display(Math(out))

    def showRates(self):
        """Show a sorted LaTeX representation of the model's rate parameters.

        Displays rendered LaTeX in the Jupyter Notebook.

        Returns
        -------
            `None`

        """
        for reaction in self._stoichiometry:
            out = latex(self._stoichiometry[reaction]['rate']) + r"\; (" + latex(reaction) + ")"
            out = utils._doubleUnderscorify(utils._greekPrependify(out))
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
            if agent == consts.EMPTYSET_SYMBOL:
                print("Spontaneous birth from EMPTYSET", end=' ')
            else:
                print(f"Agent {agent}", end=' ')
            if probs:
                print("reacts")
                for prob in probs:
                    print(f"  at rate {prob[1]}", end=' ')
                    if prob[0]:
                        print(f"when encounters {prob[0]}", end=' ')
                    else:
                        print("alone", end=' ')
                    print(f"and becomes {prob[2]}", end=', ')
                    if prob[0]:
                        print("while", end=' ')
                        for i in np.arange(len(prob[0])):
                            print(f"reagent {prob[0][i]} becomes {prob[3][i]}", end=' ')
                    print("")
            else:
                print("does not initiate any reaction.")

    def getODEs(self, method='massAction'):
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
            print("Invalid input for method. Choose either method = 'massAction' or method = 'vanKampen'. Default is 'massAction'.")

    def showODEs(self, method='massAction'):
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
                out = utils._doubleUnderscorify(utils._greekPrependify(out))
                display(Math(out))
        elif method == 'vanKampen':
            ODEdict = _getODEs_vKE(_get_orderedLists_vKE, self._stoichiometry)
            for ode in ODEdict:
                out = latex(ode) + " := " + latex(ODEdict[ode])
                out = utils._doubleUnderscorify(utils._greekPrependify(out))
                display(Math(out))
        else:
            print("Invalid input for method. Choose either method = 'massAction' or method = 'vanKampen'. Default is 'massAction'.")

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
        out = utils._doubleUnderscorify(utils._greekPrependify(out))
        display(Math(out))

    def getMasterEquation(self):
        """Gets Master Equation expressed with step operators, and substitutions.

        Returns
        -------
        :class:`dict`, :class:`dict`
            Dictionary showing all terms of the right hand side of the Master Equation
            Dictionary of substitutions used, this defaults to `None` if no substitutions were made

        """
        t = symbols('t')
        P = Function('P')
        stoich = self._stoichiometry
        nvec = []
        for key1 in stoich:
            for key2 in stoich[key1]:
                if key2 != 'rate' and stoich[key1][key2] != 'const':
                    if key2 not in nvec:
                        nvec.append(key2)
        nvec = sorted(nvec, key=default_sort_key)

        if len(nvec) < 1 or len(nvec) > 4:
            print("Derivation of Master Equation works for 1, 2, 3 or 4 different reactants only")

            return
        # assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'
        rhs_dict, substring = views._deriveMasterEquation(stoich)

        return rhs_dict, substring

    def showMasterEquation(self):
        """Displays Master equation expressed with step operators.

        Displays rendered LaTeX in the Jupyter Notebook.

        Returns
        -------
            `None`

        """

        t = symbols('t')
        P = Function('P')
        out_rhs = ""
        stoich = self._stoichiometry
        nvec = []
        for key1 in stoich:
            for key2 in stoich[key1]:
                if key2 != 'rate' and stoich[key1][key2] != 'const':
                    if key2 not in nvec:
                        nvec.append(key2)
        nvec = sorted(nvec, key=default_sort_key)

        if len(nvec) < 1 or len(nvec) > 4:
            print("Derivation of Master Equation works for 1, 2, 3 or 4 different reactants only")

            return
        # assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'
        rhs_dict, substring = views._deriveMasterEquation(stoich)

        # rhs_ME = 0
        term_count = 0
        for key in rhs_dict:
            # rhs_ME += terms_gcd(key*(rhs_dict[key][0]-1)*rhs_dict[key][1]*rhs_dict[key][2], deep=True)
            if term_count == 0:
                rhs_plus = ""
            else:
                rhs_plus = " + "
            out_rhs += rhs_plus + latex(rhs_dict[key][3]) + " ( " + latex((rhs_dict[key][0] - 1)) + " ) " + latex(rhs_dict[key][1]) + " " + latex(rhs_dict[key][2])
            term_count += 1

        if len(nvec) == 1:
            lhs_ME = Derivative(P(nvec[0], t), t)
        elif len(nvec) == 2:
            lhs_ME = Derivative(P(nvec[0], nvec[1], t), t)
        elif len(nvec) == 3:
            lhs_ME = Derivative(P(nvec[0], nvec[1], nvec[2], t), t)
        else:
            lhs_ME = Derivative(P(nvec[0], nvec[1], nvec[2], nvec[3], t), t)

        # return {lhs_ME: rhs_ME}
        out = latex(lhs_ME) + ":= " + out_rhs
        out = utils._doubleUnderscorify(out)
        out = utils._greekPrependify(out)
        display(Math(out))
        # substring is a dictionary
        if substring is not None:
            for subKey, subVal in substring.items():
                subK = utils._greekPrependify(utils._doubleUnderscorify(str(subKey)))
                subV = utils._greekPrependify(utils._doubleUnderscorify(str(subVal)))
                display(Math(r"With \; substitution:\;" + latex(subK) + ":= " + latex(subV)))

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

        rhs_vke, lhs_vke, substring = views._doVanKampenExpansion(views._deriveMasterEquation, self._stoichiometry)

        return lhs_vke, rhs_vke, substring

    def showVanKampenExpansion(self):
        """Show van Kampen expansion when the operators are expanded up to
        second order.

        Displays rendered LaTeX in the Jupyter Notebook.

        Returns
        -------
            `None`

        """
        rhs_vke, lhs_vke, substring = views._doVanKampenExpansion(views._deriveMasterEquation, self._stoichiometry)
        out = latex(lhs_vke) + " := \n" + latex(rhs_vke)
        out = utils._doubleUnderscorify(utils._greekPrependify(out))
        display(Math(out))
        # substring is a dictionary
        if substring is not None:
            for subKey, subVal in substring.items():
                subK = utils._greekPrependify(utils._doubleUnderscorify(str(subKey)))
                subV = utils._greekPrependify(utils._doubleUnderscorify(str(subVal)))
                display(Math(r"With \; substitution:\;" + latex(subK) + ":= " + latex(subV)))

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
            out = utils._doubleUnderscorify(utils._greekPrependify(out))
            display(Math(out))
        # substring is a dictionary
        if substring is not None:
            for subKey, subVal in substring.items():
                subK = utils._greekPrependify(utils._doubleUnderscorify(str(subKey)))
                subV = utils._greekPrependify(utils._doubleUnderscorify(str(subVal)))
                display(Math(r"With \; substitution:\;" + latex(subK) + ":= " + latex(subV)))

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
        EQsys1stOrdMom, EOM_1stOrderMom, NoiseSubs1stOrder, EQsys2ndOrdMom, EOM_2ndOrderMom, NoiseSubs2ndOrder = \
            _getNoiseEOM(_getFokkerPlanckEquation, _get_orderedLists_vKE, self._stoichiometry)

        return EOM_1stOrderMom, NoiseSubs1stOrder, EOM_2ndOrderMom, NoiseSubs2ndOrder

    def showNoiseEquations(self):
        """Display equations of motion of first and second order moments of noise.

        Displays rendered LaTeX in the Jupyter Notebook.

        Returns
        -------
            `None`

        """
        EQsys1stOrdMom, EOM_1stOrderMom, NoiseSubs1stOrder, EQsys2ndOrdMom, EOM_2ndOrderMom, NoiseSubs2ndOrder = \
            _getNoiseEOM(_getFokkerPlanckEquation, _get_orderedLists_vKE, self._stoichiometry)
        for eom1 in EOM_1stOrderMom:
            out = "\\displaystyle \\frac{\\textrm{d}" + latex(eom1.subs(NoiseSubs1stOrder)) + "}{\\textrm{d}t} := " + latex(EOM_1stOrderMom[eom1].subs(NoiseSubs1stOrder))
            out = utils._doubleUnderscorify(out)
            out = utils._greekPrependify(out)
            display(Math(out))
        for eom2 in EOM_2ndOrderMom:
            out = "\\displaystyle \\frac{\\textrm{d}" + latex(eom2.subs(NoiseSubs2ndOrder)) + "}{\\textrm{d}t} := " + latex(EOM_2ndOrderMom[eom2].subs(NoiseSubs2ndOrder))
            out = utils._doubleUnderscorify(out)
            out = utils._greekPrependify(out)
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
        SOL_1stOrderMom, NoiseSubs1stOrder, SOL_2ndOrdMomDict, NoiseSubs2ndOrder = \
            _getNoiseStationarySol(_getNoiseEOM, _getFokkerPlanckEquation, _get_orderedLists_vKE, self._stoichiometry)

        print('Stationary solutions of first and second order moments of noise:')
        if SOL_1stOrderMom is None:
            print('Noise 1st-order moments could not be calculated analytically.')
            return None
        else:
            for sol1 in SOL_1stOrderMom:
                out = latex(sol1.subs(NoiseSubs1stOrder)) + latex(r'(t \to \infty)') + ":= " + latex(SOL_1stOrderMom[sol1].subs(NoiseSubs1stOrder))
                out = utils._doubleUnderscorify(utils._greekPrependify(out))
                display(Math(out))
        if SOL_2ndOrdMomDict is None:
            print('Noise 2nd-order moments could not be calculated analytically.')
            return None
        else:
            for sol2 in SOL_2ndOrdMomDict:
                out = latex(sol2.subs(NoiseSubs2ndOrder)) + latex(r'(t \to \infty)') + " := " + latex(SOL_2ndOrdMomDict[sol2].subs(NoiseSubs2ndOrder))
                out = utils._doubleUnderscorify(utils._greekPrependify(out))
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
                if reactant == consts.EMPTYSET_SYMBOL:
                    reactant = Symbol(r'\emptyset')
                if reactant in self._constantReactants:
                    out += "("
                out += latex(reactant)
                if reactant in self._constantReactants:
                    out += ")"
                out += " + "
            out = out[0:len(out) - 2]  # delete the last ' + '
            out += " \\xrightarrow{" + latex(rule.rate) + "}"
            for reactant in rule.rhsReactants:
                if reactant == consts.EMPTYSET_SYMBOL:
                    reactant = Symbol(r'\emptyset')
                if reactant in self._constantReactants:
                    out += "("
                out += latex(reactant)
                if reactant in self._constantReactants:
                    out += ")"
                out += " + "
            out = out[0:len(out) - 2]  # delete the last ' + '
            out = utils._doubleUnderscorify(utils._greekPrependify(out))
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
        legend_loc : str, optional
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

        paramValuesDict = self._create_free_param_dictionary_for_controller(
            inputParams=kwargs.get('params', []),
            initWidgets=initWidgets,
            showSystemSize=True,
            showPlotLimits=False)

        IntParams = {}
        # read input parameters
        IntParams['substitutedReactant'] = [[react for react in self._getAllReactants()[0] if react not in self._reactants][0] if self._systemSize is not None else None, True]
        # the first item of extraParam2 for intial state is fixSumTo1, the second is the idle reactant
        extraParam2initS = [True, IntParams['substitutedReactant'][0]]
        IntParams['initialState'] = utils._format_advanced_option(
            optionName='initialState',
            inputValue=kwargs.get('initialState'),
            initValues=initWidgets.get('initialState'),
            extraParam=self._getAllReactants(),
            extraParam2 = extraParam2initS )
        IntParams['maxTime'] = utils._format_advanced_option(
            optionName='maxTime',
            inputValue=kwargs.get('maxTime'),
            initValues=initWidgets.get('maxTime'))
        IntParams['plotProportions'] = utils._format_advanced_option(
            optionName='plotProportions',
            inputValue=kwargs.get('plotProportions'),
            initValues=initWidgets.get('plotProportions'))
        IntParams['conserved'] = [kwargs.get('conserved', False), True]

        # construct controller
        viewController = controllers.MuMoTtimeEvolutionController(
            paramValuesDict=paramValuesDict,
            paramLabelDict=self._ratesLaTeX,
            continuousReplot=False,
            advancedOpts=IntParams,
            showSystemSize=True,
            **kwargs)

        # if showStateVars:
        #    showStateVars = [r'' + showStateVars[kk] for kk in range(len(showStateVars))]

        modelView = views.MuMoTintegrateView(self, viewController, IntParams, showStateVars, **kwargs)
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

        paramValuesDict = self._create_free_param_dictionary_for_controller(
            inputParams=kwargs.get('params', []),
            initWidgets=initWidgets,
            showSystemSize=True,
            showPlotLimits=False)

        NCParams = {}
        # read input parameters
        NCParams['substitutedReactant'] = [[react for react in self._getAllReactants()[0] if react not in self._reactants][0] if self._systemSize is not None else None, True]
        # the first item of extraParam2 for intial state is fixSumTo1, the second is the idle reactant
        extraParam2initS = [True, NCParams['substitutedReactant'][0]]
        NCParams['initialState'] = utils._format_advanced_option(
            optionName='initialState',
            inputValue=kwargs.get('initialState'),
            initValues=initWidgets.get('initialState'),
            extraParam=self._getAllReactants(),
            extraParam2=extraParam2initS)
        NCParams['maxTime'] = utils._format_advanced_option(
            optionName='maxTime',
            inputValue=kwargs.get('maxTime'),
            initValues=initWidgets.get('maxTime'))
        NCParams['conserved'] = [kwargs.get('conserved', False), True]

        EQsys1stOrdMom, EOM_1stOrderMom, NoiseSubs1stOrder, EQsys2ndOrdMom, EOM_2ndOrderMom, NoiseSubs2ndOrder = \
            _getNoiseEOM(_getFokkerPlanckEquation, _get_orderedLists_vKE, self._stoichiometry)

        # construct controller
        viewController = controllers.MuMoTtimeEvolutionController(
            paramValuesDict=paramValuesDict,
            paramLabelDict=self._ratesLaTeX,
            continuousReplot=False,
            advancedOpts=NCParams,
            showSystemSize=True,
            **kwargs)

        modelView = views.MuMoTnoiseCorrelationsView(self, viewController,
                                                     NCParams, EOM_1stOrderMom,
                                                     EOM_2ndOrderMom, **kwargs)

        viewController._setView(modelView)

        viewController._setReplotFunction(modelView._plot_NumSolODE)

        return viewController

    def _check2ndOrderMom(self, showNoise=False):
        """Check if 2nd Order moments of noise-noise correlations can be calculated via Master equation and Fokker-Planck equation"""

        if showNoise is True:
            substitutions = False
            for reaction in self._stoichiometry:
                for key in self._stoichiometry[reaction]:
                    if key != 'rate':
                        if self._stoichiometry[reaction][key] != 'const':
                            if len(self._stoichiometry[reaction][key]) > 2:
                                substitutions = True

            if substitutions is False:
                SOL_1stOrderMomDict, NoiseSubs1stOrder, SOL_2ndOrdMomDict, NoiseSubs2ndOrder = \
                    _getNoiseStationarySol(_getNoiseEOM, _getFokkerPlanckEquation, _get_orderedLists_vKE, self._stoichiometry)
                # SOL_2ndOrdMomDict is second order solution and will be used by MuMoTnoiseView
            else:
                SOL_2ndOrdMomDict = None

                # if SOL_2ndOrdMomDict is None:
                #     if substitutions == True:
                #         print('Noise in stream plots is only available for systems with exactly two time dependent reactants. Stream plot only works WITHOUT noise in case of reducing number of state variables from 3 to 2 via substitute() - method.')
                #     print('Warning: Noise in the system could not be calculated: \'showNoise\' automatically disabled.')
                #     kwargs['showNoise'] = False
        else:
            SOL_2ndOrdMomDict = None

        return SOL_2ndOrdMomDict

    # construct interactive stream plot with the option to show noise around
    # fixed points
    def stream(self, stateVariable1, stateVariable2=None, stateVariable3=None,
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
        setNumPoints : int, optional
             Used for 3d stream plots to specify the number of streams plotted
        maxTime : float, optional
             Must be strictly positive. Used for numerical integration of ODE system in 3d stream plots. Default value is 1.0.

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

        if self._systemSize is None and self._constantSystemSize is True:  # duplicate check in view and controller required for speedy error reporting, plus flexibility to instantiate view independent of controller
            print("Cannot construct field-based plot until system size is set, using substitute()")
            return None

        if self._check_state_variables(stateVariable1, stateVariable2, stateVariable3):
            if stateVariable2 is None:
                SOL_2ndOrdMomDict = self._check2ndOrderMom(showNoise=kwargs.get('showNoise', False))
            elif stateVariable3 is None:
                SOL_2ndOrdMomDict = self._check2ndOrderMom(showNoise=kwargs.get('showNoise', False))
            else:
                SOL_2ndOrdMomDict = None

            continuous_update = not (kwargs.get('showNoise', False) or kwargs.get('showFixedPoints', False))
            showNoise = kwargs.get('showNoise', False)
            showSystemSize = showNoise
            plotLimitsSlider = not(self._constantSystemSize)
            paramValuesDict = self._create_free_param_dictionary_for_controller(
                inputParams=params if params is not None else [],
                initWidgets=initWidgets,
                showSystemSize=showSystemSize,
                showPlotLimits=plotLimitsSlider)

            advancedOpts = {}
            # read input parameters
            advancedOpts['maxTime'] = utils._format_advanced_option(
                optionName='maxTime',
                inputValue=kwargs.get('maxTime'),
                initValues=initWidgets.get('maxTime'), extraParam="asNoise")
            advancedOpts['randomSeed'] = utils._format_advanced_option(
                optionName='randomSeed',
                inputValue=kwargs.get('randomSeed'),
                initValues=initWidgets.get('randomSeed'))
            # next line to address issue #283
            # advancedOpts['plotProportions'] = utils._format_advanced_option(optionName='plotProportions', inputValue=kwargs.get('plotProportions'), initValues=initWidgets.get('plotProportions'))
            # next lines useful to address issue #95
            # advancedOpts['final_x'] = utils._format_advanced_option(optionName='final_x', inputValue=kwargs.get('final_x'), initValues=initWidgets.get('final_x'), extraParam=self._getAllReactants()[0])
            # advancedOpts['final_y'] = utils._format_advanced_option(optionName='final_y', inputValue=kwargs.get('final_y'), initValues=initWidgets.get('final_y'), extraParam=self._getAllReactants()[0])
            advancedOpts['runs'] = utils._format_advanced_option(
                optionName='runs',
                inputValue=kwargs.get('runs'),
                initValues=initWidgets.get('runs'),
                extraParam="asNoise")
            advancedOpts['aggregateResults'] = utils._format_advanced_option(
                optionName='aggregateResults',
                inputValue=kwargs.get('aggregateResults'),
                initValues=initWidgets.get('aggregateResults'))

            # construct controller
            viewController = controllers.MuMoTfieldController(
                paramValuesDict=paramValuesDict,
                paramLabelDict=self._ratesLaTeX,
                continuousReplot=continuous_update,
                showPlotLimits=plotLimitsSlider,
                advancedOpts=advancedOpts,
                showSystemSize=showSystemSize, **kwargs)

            # construct view
            modelView = views.MuMoTstreamView(
                self, viewController,
                advancedOpts, SOL_2ndOrdMomDict,
                stateVariable1, stateVariable2, stateVariable3,
                params=params, **kwargs)

            viewController._setView(modelView)
            viewController._setReplotFunction(modelView._plot_field)

            return viewController
        else:
            return None

    # construct interactive vector plot with the option to show noise around
    # fixed points
    def vector(self, stateVariable1, stateVariable2, stateVariable3=None,
               params=None, initWidgets=None, **kwargs):
        """Display interactive vector plot of ``stateVariable1`` (x-axis),
        ``stateVariable2`` (y-axis), and optionally ``stateVariable3`` (z-axis)

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

        if self._systemSize is None and self._constantSystemSize is True:  # duplicate check in view and controller required for speedy error reporting, plus flexibility to instantiate view independent of controller
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
            paramValuesDict = self._create_free_param_dictionary_for_controller(
                inputParams=params if params is not None else [],
                initWidgets=initWidgets,
                showSystemSize=showSystemSize,
                showPlotLimits=plotLimitsSlider)

            advancedOpts = {}
            # read input parameters
            advancedOpts['maxTime'] = utils._format_advanced_option(
                optionName='maxTime',
                inputValue=kwargs.get('maxTime'),
                initValues=initWidgets.get('maxTime'),
                extraParam="asNoise")
            advancedOpts['randomSeed'] = utils._format_advanced_option(
                optionName='randomSeed',
                inputValue=kwargs.get('randomSeed'),
                initValues=initWidgets.get('randomSeed'))
            # next line to address issue #283
            # advancedOpts['plotProportions'] = utils._format_advanced_option(optionName='plotProportions', inputValue=kwargs.get('plotProportions'), initValues=initWidgets.get('plotProportions'))
            # next lines useful to address issue #95
            # advancedOpts['final_x'] = utils._format_advanced_option(optionName='final_x', inputValue=kwargs.get('final_x'), initValues=initWidgets.get('final_x'), extraParam=self._getAllReactants()[0])
            # advancedOpts['final_y'] = utils._format_advanced_option(optionName='final_y', inputValue=kwargs.get('final_y'), initValues=initWidgets.get('final_y'), extraParam=self._getAllReactants()[0])
            advancedOpts['runs'] = utils._format_advanced_option(
                optionName='runs',
                inputValue=kwargs.get('runs'),
                initValues=initWidgets.get('runs'),
                extraParam="asNoise")
            advancedOpts['aggregateResults'] = utils._format_advanced_option(
                optionName='aggregateResults',
                inputValue=kwargs.get('aggregateResults'),
                initValues=initWidgets.get('aggregateResults'))

            # construct controller
            viewController = controllers.MuMoTfieldController(
                paramValuesDict=paramValuesDict,
                paramLabelDict=self._ratesLaTeX,
                continuousReplot=continuous_update,
                showPlotLimits=plotLimitsSlider,
                advancedOpts=advancedOpts,
                showSystemSize=showSystemSize, **kwargs)

            # construct view
            modelView = views.MuMoTvectorView(
                self, viewController, advancedOpts, SOL_2ndOrdMomDict,
                stateVariable1, stateVariable2, stateVariable3, params=params,
                **kwargs)

            viewController._setView(modelView)
            viewController._setReplotFunction(modelView._plot_field)

            return viewController
        else:
            return None

    def bifurcation(self, bifurcationParameter, stateVariable1,
                    stateVariable2=None, initWidgets=None, **kwargs):
        r"""Construct and display bifurcation plot of ``stateVariable1``
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
        # Check for substitutions of state variables in conserved systems
        stoich = self._stoichiometry
        for key1 in stoich:
            for key2 in stoich[key1]:
                if key2 != 'rate' and stoich[key1][key2] != 'const':
                    if len(stoich[key1][key2]) == 3:
                        conserved = True

        # if bifurcationParameter[0]=='\\':
        #         bifPar = bifurcationParameter[1:]
        # else:
        #     bifPar = bifurcationParameter
        bifPar = bifurcationParameter
        # if self._systemSize:
        #     kwargs['conserved'] = True

        paramValuesDict = self._create_free_param_dictionary_for_controller(
            inputParams=kwargs.get('params', []),
            initWidgets=initWidgets,
            showSystemSize=False,
            showPlotLimits=False)

        if str(parse_latex(bifPar)) in paramValuesDict:
            del paramValuesDict[str(parse_latex(bifPar))]

        BfcParams = {}
        # read input parameters
        BfcParams['initBifParam'] = utils._format_advanced_option(
            optionName='initBifParam',
            inputValue=kwargs.get('initBifParam'),
            initValues=initWidgets.get('initBifParam'))
        BfcParams['substitutedReactant'] = [[react for react in self._getAllReactants()[0] if react not in self._reactants][0] if self._systemSize is not None else None, True]
        # the first item of extraParam2 for intial state is fixSumTo1, the second is the idle reactant
        extraParam2initS = [True, BfcParams['substitutedReactant'][0]]
        BfcParams['initialState'] = utils._format_advanced_option(
            optionName='initialState',
            inputValue=kwargs.get('initialState'),
            initValues=initWidgets.get('initialState'),
            extraParam=self._getAllReactants(),
            extraParam2=extraParam2initS)
        BfcParams['bifurcationParameter'] = [bifPar, True]
        BfcParams['conserved'] = [conserved, True]

        # construct controller
        viewController = controllers.MuMoTbifurcationController(
            paramValuesDict=paramValuesDict,
            paramLabelDict=self._ratesLaTeX,
            continuousReplot=False,
            advancedOpts=BfcParams,
            showSystemSize=False,
            **kwargs)

        # if showStateVars:
        #     showStateVars = [r'' + showStateVars[kk] for kk in range(len(showStateVars))]

        modelView = views.MuMoTbifurcationView(self, viewController, BfcParams, bifurcationParameter, stateVariable1, stateVariable2, **kwargs)
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
                legend_fontsize: int, optional
            Specify fontsize of legend.  Defaults to 14.
        legend_loc : str, optional
            Specify legend location: combinations like 'upper left' (default), 'lower right', or 'center center' are allowed (9 options in total).
        fontsize : integer, optional
            Specify fontsize for axis-labels.  If not specified, the fontsize is automatically derived from the length of axis label.
        xlab : str, optional
            Specify label on x-axis.   Defaults to 'time t'.
        ylab : str, optional
            Specify label on y-axis.   Defaults to 'reactants'.
        choose_xrange : list of float, optional
            Specify range plotted on x-axis as a two-element iterable of the form [xmin, xmax]. If not given uses data values to set axis limits.
        silent : bool, optional
            Switch on/off widgets and plot. Important for use with multicontrollers. Defaults to False.

        Returns
        -------
        :class:`MuMoTmultiagentController`
            A MuMoT controller object
        """
        if initWidgets is None:
            initWidgets = dict()

        paramValuesDict = self._create_free_param_dictionary_for_controller(
            inputParams=kwargs.get('params', []),
            initWidgets=initWidgets,
            showSystemSize=True,
            showPlotLimits=False)

        MAParams = {}
        # Read input parameters
        MAParams['substitutedReactant'] = [[react for react in self._getAllReactants()[0] if react not in self._reactants][0] if self._systemSize is not None else None, True]
        # the first item of extraParam2 for intial state is fixSumTo1, the second is the idle reactant
        # in the multiagent() view, the sum of the initial states is fixed to 1. If this wants to be changed, remember to change also the callback function of the widgets _updateInitialStateWidgets in controllers.py
        if MAParams['substitutedReactant'][0] is None and len(self._getAllReactants()[0]) > 1:
            MAParams['substitutedReactant'][0] = sorted(self._getAllReactants()[0], key=str)[0]
        extraParam2initS = [True, MAParams['substitutedReactant'][0]]
        MAParams['initialState'] = utils._format_advanced_option(
            optionName='initialState',
            inputValue=kwargs.get('initialState'),
            initValues=initWidgets.get('initialState'),
            extraParam=self._getAllReactants(),
            extraParam2=extraParam2initS)
        MAParams['maxTime'] = utils._format_advanced_option(
            optionName='maxTime',
            inputValue=kwargs.get('maxTime'),
            initValues=initWidgets.get('maxTime'))
        MAParams['randomSeed'] = utils._format_advanced_option(
            optionName='randomSeed',
            inputValue=kwargs.get('randomSeed'),
            initValues=initWidgets.get('randomSeed'))
        MAParams['motionCorrelatedness'] = utils._format_advanced_option(
            optionName='motionCorrelatedness',
            inputValue=kwargs.get('motionCorrelatedness'),
            initValues=initWidgets.get('motionCorrelatedness'))
        MAParams['particleSpeed'] = utils._format_advanced_option(
            optionName='particleSpeed',
            inputValue=kwargs.get('particleSpeed'),
            initValues=initWidgets.get('particleSpeed'))
        MAParams['timestepSize'] = utils._format_advanced_option(
            optionName='timestepSize',
            inputValue=kwargs.get('timestepSize'),
            initValues=initWidgets.get('timestepSize'))
        MAParams['netType'] = utils._format_advanced_option(
            optionName='netType',
            inputValue=kwargs.get('netType'),
            initValues=initWidgets.get('netType'))
        systemSize = paramValuesDict["systemSize"][0]
        MAParams['netParam'] = utils._format_advanced_option(
            optionName='netParam',
            inputValue=kwargs.get('netParam'),
            initValues=initWidgets.get('netParam'),
            extraParam=MAParams['netType'],
            extraParam2=systemSize)
        MAParams['plotProportions'] = utils._format_advanced_option(
            optionName='plotProportions',
            inputValue=kwargs.get('plotProportions'),
            initValues=initWidgets.get('plotProportions'))
        MAParams['realtimePlot'] = utils._format_advanced_option(
            optionName='realtimePlot',
            inputValue=kwargs.get('realtimePlot'),
            initValues=initWidgets.get('realtimePlot'))
        MAParams['showTrace'] = utils._format_advanced_option(
            optionName='showTrace',
            inputValue=kwargs.get('showTrace'),
            initValues=initWidgets.get('showTrace', MAParams['netType'] == consts.NetworkType.DYNAMIC))
        MAParams['showInteractions'] = utils._format_advanced_option(
            optionName='showInteractions',
            inputValue=kwargs.get('showInteractions'),
            initValues=initWidgets.get('showInteractions'))
        MAParams['visualisationType'] = utils._format_advanced_option(
            optionName='visualisationType',
            inputValue=kwargs.get('visualisationType'),
            initValues=initWidgets.get('visualisationType'),
            extraParam="multiagent")
        MAParams['final_x'] = utils._format_advanced_option(
            optionName='final_x',
            inputValue=kwargs.get('final_x'),
            initValues=initWidgets.get('final_x'),
            extraParam=self._getAllReactants()[0])
        MAParams['final_y'] = utils._format_advanced_option(
            optionName='final_y',
            inputValue=kwargs.get('final_y'),
            initValues=initWidgets.get('final_y'),
            extraParam=self._getAllReactants()[0])
        MAParams['runs'] = utils._format_advanced_option(
            optionName='runs',
            inputValue=kwargs.get('runs'),
            initValues=initWidgets.get('runs'))
        MAParams['aggregateResults'] = utils._format_advanced_option(
            optionName='aggregateResults',
            inputValue=kwargs.get('aggregateResults'),
            initValues=initWidgets.get('aggregateResults'))

        # if the netType is a fixed-param and its value is not 'DYNAMIC', all useless parameter become fixed (and widgets are never displayed)
        if MAParams['netType'][-1]:
            decodedNetType = utils._decodeNetworkTypeFromString(MAParams['netType'][0])
            if decodedNetType != consts.NetworkType.DYNAMIC:
                MAParams['motionCorrelatedness'][-1] = True
                MAParams['particleSpeed'][-1] = True
                MAParams['showTrace'][-1] = True
                MAParams['showInteractions'][-1] = True
                if decodedNetType == consts.NetworkType.FULLY_CONNECTED:
                    MAParams['netParam'][-1] = True

        # Construct controller
        viewController = controllers.MuMoTmultiagentController(
            paramValuesDict=paramValuesDict,
            paramLabelDict=self._ratesLaTeX,
            continuousReplot=False,
            advancedOpts=MAParams,
            showSystemSize=True, **kwargs)

        # Get the default network values assigned from the controller
        modelView = views.MuMoTmultiagentView(self, viewController, MAParams, **kwargs)
        viewController._setView(modelView)
        # viewController._setReplotFunction(modelView._computeAndPlotSimulation(self._reactants, self._rules))
        viewController._setReplotFunction(modelView._computeAndPlotSimulation,
                                          modelView._redrawOnly)
        # viewController._widgetsExtraParams['netType'].value.observe(modelView._update_net_params, 'value') #netType is special

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
        legend_loc : str, optional
            Specify legend location: combinations like 'upper left' (default), 'lower right', or 'center center' are allowed (9 options in total).
        fontsize : integer, optional
            Specify fontsize for axis-labels.  If not specified, the fontsize is automatically derived from the length of axis label.
        xlab : str, optional
            Specify label on x-axis.   Defaults to 'time t'.
        ylab : str, optional
            Specify label on y-axis.   Defaults to 'reactants'.
        choose_xrange : list of float, optional
            Specify range plotted on x-axis as a two-element iterable of the form [xmin, xmax]. If not given uses data values to set axis limits.
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
        # Read input parameters
        ssaParams['substitutedReactant'] = [[react for react in self._getAllReactants()[0] if react not in self._reactants][0] if self._systemSize is not None else None, True]
        # the first item of extraParam2 for intial state is fixSumTo1, the second is the idle reactant
        # in the SSA() view, the sum of the initial states is fixed to 1. If this wants to be changed, remember to change also the callback function of the widgets _updateInitialStateWidgets in controllers.py
        if ssaParams['substitutedReactant'][0] is None and len(self._getAllReactants()[0]) > 1:
            ssaParams['substitutedReactant'][0] = sorted(self._getAllReactants()[0], key=str)[0]
        extraParam2initS = [True, ssaParams['substitutedReactant'][0]]
        ssaParams['initialState'] = utils._format_advanced_option(
            optionName='initialState',
            inputValue=kwargs.get('initialState'),
            initValues=initWidgets.get('initialState'),
            extraParam=self._getAllReactants(),
            extraParam2=extraParam2initS)
        ssaParams['maxTime'] = utils._format_advanced_option(
            optionName='maxTime',
            inputValue=kwargs.get('maxTime'),
            initValues=initWidgets.get('maxTime'))
        ssaParams['randomSeed'] = utils._format_advanced_option(
            optionName='randomSeed',
            inputValue=kwargs.get('randomSeed'),
            initValues=initWidgets.get('randomSeed'))
        ssaParams['plotProportions'] = utils._format_advanced_option(
            optionName='plotProportions',
            inputValue=kwargs.get('plotProportions'),
            initValues=initWidgets.get('plotProportions'))
        ssaParams['realtimePlot'] = utils._format_advanced_option(
            optionName='realtimePlot',
            inputValue=kwargs.get('realtimePlot'),
            initValues=initWidgets.get('realtimePlot'))
        ssaParams['visualisationType'] = utils._format_advanced_option(
            optionName='visualisationType',
            inputValue=kwargs.get('visualisationType'),
            initValues=initWidgets.get('visualisationType'),
            extraParam="SSA")
        ssaParams['final_x'] = utils._format_advanced_option(
            optionName='final_x',
            inputValue=kwargs.get('final_x'),
            initValues=initWidgets.get('final_x'),
            extraParam=self._getAllReactants()[0])
        ssaParams['final_y'] = utils._format_advanced_option(
            optionName='final_y',
            inputValue=kwargs.get('final_y'),
            initValues=initWidgets.get('final_y'),
            extraParam=self._getAllReactants()[0])
        ssaParams['runs'] = utils._format_advanced_option(
            optionName='runs',
            inputValue=kwargs.get('runs'),
            initValues=initWidgets.get('runs'))
        ssaParams['aggregateResults'] = utils._format_advanced_option(
            optionName='aggregateResults',
            inputValue=kwargs.get('aggregateResults'),
            initValues=initWidgets.get('aggregateResults'))

        # construct controller
        viewController = controllers.MuMoTstochasticSimulationController(
            paramValuesDict=paramValuesDict,
            paramLabelDict=self._ratesLaTeX,
            continuousReplot=False,
            advancedOpts=ssaParams,
            showSystemSize=True,
            **kwargs)

        modelView = views.MuMoTSSAView(self, viewController, ssaParams, **kwargs)
        viewController._setView(modelView)

        viewController._setReplotFunction(modelView._computeAndPlotSimulation, modelView._redrawOnly)

        return viewController

    def _getAllReactants(self):
        """Get the pair of set (reactants, constantReactants).

        This method is necessary to have all reactants (to set the system size) also after a substitution has occurred.

        """
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

    def _get_rates_from_stoichiometry(self):
        rates = set()
        for reaction in self._stoichiometry.values():
            if reaction['rate']: 
                for symb in reaction['rate'].atoms():
                    if isinstance(symb, Symbol):
                        rates.add( symb )
        return rates

    def _create_free_param_dictionary_for_controller(self, inputParams, initWidgets=None, showSystemSize=False, showPlotLimits=False):
        initWidgetsSympy = {parse_latex(paramName): paramValue for paramName, paramValue in initWidgets.items()} if initWidgets is not None else {}

        paramValuesDict = {}
        for freeParam in self._get_rates_from_stoichiometry().union(self._constantReactants):
            paramValuesDict[str(freeParam)] = utils._parse_input_keyword_for_numeric_widgets(
                inputValue=utils._get_item_from_params_list(inputParams, str(freeParam)),
                defaultValueRangeStep=[defaults.MuMoTdefault._initialRateValue, defaults.MuMoTdefault._rateLimits[0], defaults.MuMoTdefault._rateLimits[1], defaults.MuMoTdefault._rateStep],
                initValueRangeStep=initWidgetsSympy.get(freeParam),
                validRange=(-float("inf"), float("inf")))

        if showSystemSize:
            paramValuesDict['systemSize'] = utils._parse_input_keyword_for_numeric_widgets(
                inputValue=utils._get_item_from_params_list(inputParams, 'systemSize'),
                defaultValueRangeStep=[defaults.MuMoTdefault._systemSize,
                                       defaults.MuMoTdefault._systemSizeLimits[0],
                                       defaults.MuMoTdefault._systemSizeLimits[1],
                                       defaults.MuMoTdefault._systemSizeStep],
                initValueRangeStep=initWidgets.get('systemSize'),
                validRange=(1, float("inf")))
        if showPlotLimits:
            paramValuesDict['plotLimits'] = utils._parse_input_keyword_for_numeric_widgets(
                inputValue=utils._get_item_from_params_list(inputParams, 'plotLimits'),
                defaultValueRangeStep=[defaults.MuMoTdefault._plotLimits,
                                       defaults.MuMoTdefault._plotLimitsLimits[0],
                                       defaults.MuMoTdefault._plotLimitsLimits[1],
                                       defaults.MuMoTdefault._plotLimitsStep],
                initValueRangeStep=initWidgets.get('plotLimits'),
                validRange=(-float("inf"), float("inf")))

        return paramValuesDict

    def _getSingleAgentRules(self):
        """Derive the single-agent rules (which are used in the multiagent simulation) from the reaction rules"""
        # Create the empty structure
        self._agentProbabilities = {}
        (allReactants, allConstantReactants) = self._getAllReactants()
        for reactant in allReactants | allConstantReactants | {consts.EMPTYSET_SYMBOL}:
            self._agentProbabilities[reactant] = []

        # Populate the created structure
        for rule in self._rules:
            targetReact = []
            # Check if constant reactants are not changing state
            # WARNING! if the checks hereafter are changed, it might be necessary to modify the way in which network-simulations are disabled (atm only on consts.EMPTYSET_SYMBOL presence because (A) -> B are not allowed)
            # for idx, reactant in enumerate(rule.lhsReactants):
            #     if reactant in allConstantReactants: #self._constantReactants:
            #         if not (rule.rhsReactants[idx] == reactant or rule.rhsReactants[idx] == consts.EMPTYSET_SYMBOL):
            #             errorMsg = 'In rule ' + str(rule.lhsReactants) + ' -> '  + str(rule.rhsReactants) + ' constant reactant are not properly used. ' \
            #                         'Constant reactants must either match the same constant reactant or the EMPTYSET on the right-handside. \n' \
            #                         'NOTE THAT ORDER MATTERS: MuMoT assumes that first reactant on left-handside becomes first reactant on right-handside and so on for sencond and third...'
            #             print(errorMsg)
            #             raise exceptions.MuMoTValueError(errorMsg)
            #     elif rule.rhsReactants[idx] in allConstantReactants:
            #         errorMsg = 'In rule ' + str(rule.lhsReactants) + ' -> '  + str(rule.rhsReactants) + ' constant reactant are not properly used.' \
            #                         'Constant reactants appears on the right-handside and not on the left-handside. \n' \
            #                         'NOTE THAT ORDER MATTERS: MuMoT assumes that first reactant on left-handside becomes first reactant on right-handside and so on for sencond and third...'
            #         print(errorMsg)
            #         raise exceptions.MuMoTValueError(errorMsg)

            #     if reactant == consts.EMPTYSET_SYMBOL:
            #         targetReact.append(rule.rhsReactants[idx])

            for reactant in rule.rhsReactants:
                if reactant in allConstantReactants:
                    warningMsg = 'WARNING! Constant reactants appearing on the right-handside are ignored. Every constant reactant on the left-handside (implicitly) corresponds to the same constant reactant on the right-handside.\n'\
                                 f'E.g., in rule ' + str(rule.lhsReactants) + ' -> ' + str(rule.rhsReactants) + ' constant reactants should not appear on the right-handside.'
                    warn(warningMsg, exceptions.MuMoTWarning)
                    break  # print maximum one warning

            # Add to the target of the first non-empty item the new born coming from empty-set or constant reactants
            for idx, reactant in enumerate(rule.lhsReactants):
                if reactant == consts.EMPTYSET_SYMBOL or reactant in allConstantReactants:
                    # Constant reactants on the right-handside are ignored
                    if rule.rhsReactants[idx] not in allConstantReactants:
                        targetReact.append(rule.rhsReactants[idx])

            # Creating a rule for the first non-empty element (on the lhs) of the rule (if all empty, it uses an empty)
            idx = 0
            while idx < len(rule.lhsReactants) - 1:
                reactant = rule.lhsReactants[idx]
                if not reactant == consts.EMPTYSET_SYMBOL:
                    break
                else:
                    idx += 1

            # create list of other reactants on the left-handside that reacts with the considered reactant
            otherReact = []
            otherTargets = []
            for idx2, react2 in enumerate(rule.lhsReactants):
                if idx == idx2 or react2 == consts.EMPTYSET_SYMBOL:
                    continue
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
            elif not reactant == consts.EMPTYSET_SYMBOL:  # if empty it's not added because it has been already added in the initial loop
                targetReact.append(rule.rhsReactants[idx])

            # create a new entry
            self._agentProbabilities[reactant].append([otherReact, rule.rate, targetReact, otherTargets])
        # print(self._agentProbabilities)

    def _check_state_variables(self, stateVariable1, stateVariable2, stateVariable3=None):
        if parse_latex(stateVariable1) in self._reactants and (stateVariable2 is None or parse_latex(stateVariable2) in self._reactants) and (stateVariable3 is None or parse_latex(stateVariable3) in self._reactants):
            if (stateVariable1 != stateVariable2 and stateVariable1 != stateVariable3 and stateVariable2 != stateVariable3) or (stateVariable2 is None and stateVariable3 is None):
                return True
            else:
                print('State variables cannot be the same')
                return False
        else:
            print('Invalid reactant provided as state variable')
            return False

    def _getFuncs(self):
        """Lambdify sympy equations for numerical integration, plotting, etc."""
        # if self._systemSize is None:
        #     assert false ## @todo is this necessary?
        if self._funcs is None:
            argList = []
            for reactant in self._reactants:
                argList.append(reactant)
            for reactant in self._constantReactants:
                argList.append(reactant)
            for rate in self._get_rates_from_stoichiometry():
                argList.append(rate)
            if self._systemSize is not None:
                argList.append(self._systemSize)
            self._args = tuple(argList)
            self._funcs = {}
            for equation in self._equations:
                f = lambdify(self._args, self._equations[equation], "math")
                self._funcs[equation] = f

        return self._funcs

    def _getArgTuple1d(self, argDict, stateVariable1, X):
        """Get tuple to evalute functions returned by _getFuncs with, for 2d field-based plots."""
        argList = []
        for arg in self._args:
            if arg == stateVariable1:
                argList.append(X)
            elif arg == self._systemSize:
                argList.append(1)  # @todo: system size set to 1
            else:
                try:
                    argList.append(argDict[arg])
                except KeyError:
                    raise exceptions.MuMoTValueError('Unexpected reactant \'' + str(arg) + '\': system size > 1?')
        return tuple(argList)

    def _getArgTuple2d(self, argDict, stateVariable1, stateVariable2, X, Y):
        """Get tuple to evalute functions returned by _getFuncs with, for 2d field-based plots."""
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
                    raise exceptions.MuMoTValueError('Unexpected reactant \'' + str(arg) + '\': system size > 2?')
        return tuple(argList)

    def _getArgTuple3d(self, argDict, stateVariable1, stateVariable2, stateVariable3, X, Y, Z):
        """Get tuple to evalute functions returned by _getFuncs with, for 3d field-based plots."""
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
                    raise exceptions.MuMoTValueError('Unexpected reactant: system size > 3?')

        return tuple(argList)

    def _getArgTuple(self, argDict, reactants, reactantValues):
        """Get tuple to evalute functions returned by _getFuncs with."""
        assert False  # need to work this out
        argList = []
        # for arg in self._args:
        #     if arg == stateVariable1:
        #         argList.append(X)
        #     elif arg == stateVariable2:
        #         argList.append(Y)
        #     elif arg == stateVariable3:
        #         argList.append(Z)
        #     elif arg == self._systemSize:
        #         argList.append(1) ## @todo: system size set to 1
        #     else:
        #         argList.append(argDict[arg])

        return tuple(argList)

    def _localLaTeXimageFile(self, source):
        """Render LaTeX source to local image file."""
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
        # @todo: check when this is invoked
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
        raise exceptions.MuMoTWarning("Loading from file not currently supported")

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
                if token in consts.GREEK_LETT_RESERVED_LIST:
                    raise exceptions.MuMoTSyntaxError(f"Reserved letter {token} encountered: the list of reserved letters is {consts.GREEK_LETT_RESERVED_LIST_PRINT}")
                constantReactant = False

                if state == 'A':
                    if token not in ("+", "->", ":"):
                        state = 'B'
                        if '^' in token:
                            raise exceptions.MuMoTSyntaxError(f"Reactants cannot contain '^' :{token} in {rule}")
                        reactantCount += 1
                        if (token[0] == '(' and token[-1] == ')'):
                            constantReactant = True
                            token = token.replace('(', '')
                            token = token.replace(')', '')
                        if token == r'\emptyset':
                            constantReactant = True
                            token = '1'
                        expr = parse_latex(token)
                        reactantAtoms = expr.atoms()
                        if len(reactantAtoms) != 1:
                            raise exceptions.MuMoTSyntaxError(f"Non-singleton symbol set in token {token} in rule {rule}")
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
                        exceptions._raiseModelError("reactant", token, rule)
                        return
                elif state == 'B':
                    if token == "->":
                        state = 'C'
                    elif token == '+':
                        state = 'A'
                    else:
                        exceptions._raiseModelError("'->' or '+'", token, rule)
                        return
                elif state == 'C':
                    if token != "+" and token != "->" and token != ":":
                        state = 'D'
                        if '^' in token:
                            raise exceptions.MuMoTSyntaxError(f"Reactants cannot contain '^' : {token} in {rule}")
                        reactantCount -= 1
                        if (token[0] == '(' and token[-1] == ')'):
                            constantReactant = True
                            token = token.replace('(', '')
                            token = token.replace(')', '')
                        if token == r'\emptyset':
                            constantReactant = True
                            token = '1'
                        expr = parse_latex(token)
                        reactantAtoms = expr.atoms()
                        if len(reactantAtoms) != 1:
                            raise exceptions.MuMoTSyntaxError(f"Non-singleton symbol set in token {token} in rule {rule}")
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
                        exceptions._raiseModelError("reactant", token, rule)
                        return
                elif state == 'D':
                    if token == ":":
                        state = 'E'
                    elif token == '+':
                        state = 'C'
                    else:
                        exceptions._raiseModelError("':' or '+'", token, rule)
                        return
                elif state == 'E':
                    rate += token
                    # state = 'F'
                else:
                    # should never reach here
                    assert False

            newRule.rate = parse_latex(rate)
            rateAtoms = newRule.rate.atoms(Symbol)
            for atom in rateAtoms:
                if atom not in rates:
                    rates.add(atom)

            if reactantCount == 0:
                rules.append(newRule)
            else:
                raise exceptions.MuMoTSyntaxError(f"Unequal number of reactants on lhs and rhs of rule {rule}")

            if constantReactantCount != 0:
                model._constantSystemSize = False

    model._rules = rules
    model._reactants = reactants
    model._constantReactants = constantReactants
    # check intersection of reactants and constantReactants is empty
    intersect = model._reactants.intersection(model._constantReactants)
    if len(intersect) != 0:
        raise exceptions.MuMoTSyntaxError("Following reactants defined as both constant and variable: {intersect}")
    model._rates = rates
    model._equations = views._deriveODEsFromRules(model._reactants, model._rules)
    model._ratesLaTeX = {}
    rates = map(latex, list(model._rates))
    for (rate, latex_str) in zip(model._rates, rates):
        model._ratesLaTeX[repr(rate)] = latex_str
    constantReactants = map(latex, list(model._constantReactants))
    for (reactant, latexStr) in zip(model._constantReactants, constantReactants):
        # model._ratesLaTeX[repr(reactant)] = '(' + latexStr + ')'
        model._ratesLaTeX[repr(reactant)] = r'\Phi_{' + latexStr + '}'

    model._stoichiometry = _getStoichiometry(model._rules, model._constantReactants)

    return model


def _get_orderedLists_vKE(stoich):
    """Create list of dictionaries where the key is the system size order."""
    V = Symbol(r'\overline{V}', real=True, constant=True)
    stoichiometry = stoich
    rhs_vke, lhs_vke, substring = views._doVanKampenExpansion(views._deriveMasterEquation, stoichiometry)
    Vlist_lhs = []
    Vlist_rhs = []
    for jj in range(len(rhs_vke.args)):
        try:
            Vlist_rhs.append(simplify(rhs_vke.args[jj]).collect(V, evaluate=False))
        except NotImplementedError:
            prod = 1
            for nn in range(len(rhs_vke.args[jj].args) - 1):
                prod *= rhs_vke.args[jj].args[nn]
            tempdict = prod.collect(V, evaluate=False)
            for key in tempdict:
                Vlist_rhs.append({key: prod / key * rhs_vke.args[jj].args[-1]})

    for jj in range(len(lhs_vke.args)):
        try:
            Vlist_lhs.append(simplify(lhs_vke.args[jj]).collect(V, evaluate=False))
        except NotImplementedError:
            prod = 1
            for nn in range(len(lhs_vke.args[jj].args) - 1):
                prod *= lhs_vke.args[jj].args[nn]
            tempdict = prod.collect(V, evaluate=False)
            for key in tempdict:
                Vlist_lhs.append({key: prod / key * lhs_vke.args[jj].args[-1]})
    return Vlist_lhs, Vlist_rhs, substring


def _getFokkerPlanckEquation(_get_orderedLists_vKE, stoich):
    """Return the Fokker-Planck equation."""
    t = symbols('t')
    P = Function('P')
    V = Symbol(r'\overline{V}', real=True, constant=True)
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

    FPE = lhsFPE - rhsFPE

    nvec = []
    for key1 in stoich:
        for key2 in stoich[key1]:
            if key2 != 'rate' and stoich[key1][key2] != 'const':
                if key2 not in nvec:
                    nvec.append(key2)
    nvec = sorted(nvec, key=default_sort_key)
    if len(nvec) < 1 or len(nvec) > 4:
        print("Derivation of Fokker Planck equation works for 1, 2, 3 or 4 different reactants only")

        return None, None
    # assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'

    NoiseDict = {}
    for kk in range(len(nvec)):
        NoiseDict[nvec[kk]] = Symbol('eta_' + str(nvec[kk]))

    if len(Vlist_lhs) - 1 == 1:
        SOL_FPE = solve(FPE, Derivative(P(NoiseDict[nvec[0]], t), t), dict=True)[0]
    elif len(Vlist_lhs) - 1 == 2:
        SOL_FPE = solve(FPE, Derivative(P(NoiseDict[nvec[0]], NoiseDict[nvec[1]], t), t), dict=True)[0]
    elif len(Vlist_lhs) - 1 == 3:
        SOL_FPE = solve(FPE, Derivative(P(NoiseDict[nvec[0]], NoiseDict[nvec[1]], NoiseDict[nvec[2]], t), t), dict=True)[0]
    elif len(Vlist_lhs) - 1 == 4:
        SOL_FPE = solve(FPE, Derivative(P(NoiseDict[nvec[0]], NoiseDict[nvec[1]], NoiseDict[nvec[2]], NoiseDict[nvec[3]], t), t), dict=True)[0]
    else:
        print('Not implemented yet.')

    return SOL_FPE, substring


def _getNoiseEOM(_getFokkerPlanckEquation, _get_orderedLists_vKE, stoich):
    """Calculates noise in the system.

    Returns equations of motion for noise.

    """
    t = symbols('t')
    P = Function('P')
    M_1 = Function('M_1')
    M_2 = Function('M_2')

    # A,B, alpha, beta, gamma = symbols('A B alpha beta gamma')
    # custom_stoich= {'reaction1': {'rate': alpha, A: [0,1]}, 'reaction2': {'rate': gamma, A: [2,0], B: [0,1]},
    #                  'reaction3': {'rate': beta, B: [1,0]}}
    # stoich = custom_stoich

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
        NoiseDict[nvec[kk]] = Symbol('eta_' + str(nvec[kk]))
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
        NoiseSub1stOrder[NoiseDict[noise1] * Pdim] = M_1(NoiseDict[noise1])
        for noise2 in NoiseDict:
            for noise3 in NoiseDict:
                key = NoiseDict[noise1] * NoiseDict[noise2] * Derivative(Pdim, NoiseDict[noise3])
                if key not in NoiseSub1stOrder:
                    if NoiseDict[noise1] == NoiseDict[noise2] and NoiseDict[noise3] == NoiseDict[noise1]:
                        NoiseSub1stOrder[key] = -2 * M_1(NoiseDict[noise1])
                    elif NoiseDict[noise1] != NoiseDict[noise2] and NoiseDict[noise3] == NoiseDict[noise1]:
                        NoiseSub1stOrder[key] = -M_1(NoiseDict[noise2])
                    elif NoiseDict[noise1] != NoiseDict[noise2] and NoiseDict[noise3] == NoiseDict[noise2]:
                        NoiseSub1stOrder[key] = -M_1(NoiseDict[noise1])
                    elif NoiseDict[noise1] != NoiseDict[noise3] and NoiseDict[noise3] != NoiseDict[noise2]:
                        NoiseSub1stOrder[key] = 0
                    else:
                        NoiseSub1stOrder[key] = 0
                key2 = NoiseDict[noise1] * Derivative(Pdim, NoiseDict[noise2], NoiseDict[noise3])
                if key2 not in NoiseSub1stOrder:
                    NoiseSub1stOrder[key2] = 0

    for noise1 in NoiseDict:
        for noise2 in NoiseDict:
            key = NoiseDict[noise1] * NoiseDict[noise2] * Pdim
            if key not in NoiseSub2ndOrder:
                NoiseSub2ndOrder[key] = M_2(NoiseDict[noise1] * NoiseDict[noise2])
            for noise3 in NoiseDict:
                for noise4 in NoiseDict:
                    key2 = NoiseDict[noise1] * NoiseDict[noise2] * NoiseDict[noise3] * Derivative(Pdim, NoiseDict[noise4])
                    if key2 not in NoiseSub2ndOrder:
                        if noise1 == noise2 and noise2 == noise3 and noise3 == noise4:
                            NoiseSub2ndOrder[key2] = -3 * M_2(NoiseDict[noise1] * NoiseDict[noise1])
                        elif noise1 == noise2 and noise2 != noise3 and noise1 == noise4:
                            NoiseSub2ndOrder[key2] = -2 * M_2(NoiseDict[noise1] * NoiseDict[noise3])
                        elif noise1 == noise2 and noise2 != noise3 and noise3 == noise4:
                            NoiseSub2ndOrder[key2] = -M_2(NoiseDict[noise1] * NoiseDict[noise2])
                        elif noise1 != noise2 and noise2 == noise3 and noise1 == noise4:
                            NoiseSub2ndOrder[key2] = -M_2(NoiseDict[noise2] * NoiseDict[noise3])
                        elif noise1 != noise2 and noise2 == noise3 and noise3 == noise4:
                            NoiseSub2ndOrder[key2] = -2 * M_2(NoiseDict[noise1] * NoiseDict[noise2])
                        elif noise1 != noise2 and noise2 != noise3 and noise1 != noise3:
                            if noise1 == noise4:
                                NoiseSub2ndOrder[key2] = -M_2(NoiseDict[noise2] * NoiseDict[noise3])
                            elif noise2 == noise4:
                                NoiseSub2ndOrder[key2] = -M_2(NoiseDict[noise1] * NoiseDict[noise3])
                            elif noise3 == noise4:
                                NoiseSub2ndOrder[key2] = -M_2(NoiseDict[noise1] * NoiseDict[noise2])
                            else:
                                NoiseSub2ndOrder[key2] = 0
                        else:
                            NoiseSub2ndOrder[key2] = 0
                    key3 = NoiseDict[noise1] * NoiseDict[noise2] * Derivative(Pdim, NoiseDict[noise3], NoiseDict[noise4])
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
            eq1stOrderMoment = (NoiseDict[noise] * FPEdict[fpe_lhs]).expand()
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
                eq2ndOrderMoment = (NoiseDict[noise1] * NoiseDict[noise2] * FPEdict[fpe_lhs]).expand()
                eq2ndOrderMoment = eq2ndOrderMoment.subs(NoiseSub2ndOrder)
                eq2ndOrderMoment = eq2ndOrderMoment.subs(NoiseSub1stOrder)
                # eq2ndOrderMoment = eq2ndOrderMoment.subs(SOL_1stOrderMom[0])
                if len(NoiseDict) == 1:
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[0]]))
                elif len(NoiseDict) == 2:
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[0]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[1]] * NoiseDict[nvec[1]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[1]]))
                elif len(NoiseDict) == 3:
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[0]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[1]] * NoiseDict[nvec[1]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[2]] * NoiseDict[nvec[2]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[1]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[2]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[1]] * NoiseDict[nvec[2]]))
                else:
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[0]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[1]] * NoiseDict[nvec[1]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[2]] * NoiseDict[nvec[2]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[3]] * NoiseDict[nvec[3]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[1]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[2]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[3]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[1]] * NoiseDict[nvec[2]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[1]] * NoiseDict[nvec[3]]))
                    eq2ndOrderMoment = collect(eq2ndOrderMoment, M_2(NoiseDict[nvec[2]] * NoiseDict[nvec[3]]))

                eq2ndOrderMoment = eq2ndOrderMoment.simplify()
                if eq2ndOrderMoment not in EQsys2ndOrdMom:
                    EQsys2ndOrdMom.append(eq2ndOrderMoment)
                if M_2(NoiseDict[noise1] * NoiseDict[noise2]) not in EOM_2ndOrderMom:
                    EOM_2ndOrderMom[M_2(NoiseDict[noise1] * NoiseDict[noise2])] = eq2ndOrderMoment
                    NoiseSubs2ndOrder[M_2(NoiseDict[noise1] * NoiseDict[noise2])] = (
                        r'\left< \vphantom{Dg}\right.' + latex(NoiseDict[noise1] * NoiseDict[noise2]) + r'\left. \vphantom{Dg}\right>')

    return EQsys1stOrdMom, EOM_1stOrderMom, NoiseSubs1stOrder, EQsys2ndOrdMom, EOM_2ndOrderMom, NoiseSubs2ndOrder


def _getNoiseStationarySol(_getNoiseEOM, _getFokkerPlanckEquation, _get_orderedLists_vKE, stoich):
    """Calculate noise in the system.

    Returns analytical solution for stationary noise.

    """
    t = symbols('t')
    P = Function('P')
    M_1 = Function('M_1')
    M_2 = Function('M_2')

    EQsys1stOrdMom, EOM_1stOrderMom, NoiseSubs1stOrder, EQsys2ndOrdMom, EOM_2ndOrderMom, NoiseSubs2ndOrder = _getNoiseEOM(_getFokkerPlanckEquation, _get_orderedLists_vKE, stoich)

    nvec = []
    for key1 in stoich:
        for key2 in stoich[key1]:
            if key2 != 'rate' and stoich[key1][key2] != 'const':
                if key2 not in nvec:
                    nvec.append(key2)
    nvec = sorted(nvec, key=default_sort_key)
    if len(nvec) < 1 or len(nvec) > 4:
        print("showNoiseSolutions works for 1, 2, 3 or 4 different reactants only")

        return
    # assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'

    NoiseDict = {}
    for kk in range(len(nvec)):
        NoiseDict[nvec[kk]] = Symbol('eta_' + str(nvec[kk]))
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
        SOL_2ndOrderMom = list(linsolve(EQsys2ndOrdMom, [M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[0]])]))[0]  # only one set of solutions (if any) in linear system of equations
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[0]])] = SOL_2ndOrderMom[0]

    elif len(NoiseDict) == 2:
        SOL_2ndOrderMom = list(linsolve(EQsys2ndOrdMom, [M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[0]]),
                                                         M_2(NoiseDict[nvec[1]] * NoiseDict[nvec[1]]),
                                                         M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[1]])]))[0]  # only one set of solutions (if any) in linear system of equations

        if M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[1]]) in SOL_2ndOrderMom:
            print('Solution for 2nd order noise moments NOT unique!')
            return None, None, None, None

        # ZsubDict = {M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[1]]): 0}
        # SOL_2ndOrderMomMod = []
        # for nn in range(len(SOL_2ndOrderMom)):
        #     SOL_2ndOrderMomMod.append(SOL_2ndOrderMom[nn].subs(ZsubDict))
        # SOL_2ndOrderMom = SOL_2ndOrderMomMod
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[0]])] = SOL_2ndOrderMom[0]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[1]] * NoiseDict[nvec[1]])] = SOL_2ndOrderMom[1]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[1]])] = SOL_2ndOrderMom[2]

    elif len(NoiseDict) == 3:
        SOL_2ndOrderMom = list(linsolve(EQsys2ndOrdMom, [M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[0]]),
                                                         M_2(NoiseDict[nvec[1]] * NoiseDict[nvec[1]]),
                                                         M_2(NoiseDict[nvec[2]] * NoiseDict[nvec[2]]),
                                                         M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[1]]),
                                                         M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[2]]),
                                                         M_2(NoiseDict[nvec[1]] * NoiseDict[nvec[2]])]))[0]  # only one set of solutions (if any) in linear system of equations; hence index [0]
        # ZsubDict = {}
        for noise1 in NoiseDict:
            for noise2 in NoiseDict:
                if M_2(NoiseDict[noise1] * NoiseDict[noise2]) in SOL_2ndOrderMom:
                    print('Solution for 2nd order noise moments NOT unique!')
                    return None, None, None, None
        # ZsubDict[M_2(NoiseDict[noise1] * NoiseDict[noise2])] = 0
        # if len(ZsubDict) > 0:
        #     SOL_2ndOrderMomMod = []
        #     for nn in range(len(SOL_2ndOrderMom)):
        #         SOL_2ndOrderMomMod.append(SOL_2ndOrderMom[nn].subs(ZsubDict))
        # SOL_2ndOrderMom = SOL_2ndOrderMomMod
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[0]])] = SOL_2ndOrderMom[0]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[1]] * NoiseDict[nvec[1]])] = SOL_2ndOrderMom[1]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[2]] * NoiseDict[nvec[2]])] = SOL_2ndOrderMom[2]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[1]])] = SOL_2ndOrderMom[3]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[2]])] = SOL_2ndOrderMom[4]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[1]] * NoiseDict[nvec[2]])] = SOL_2ndOrderMom[5]

    else:
        SOL_2ndOrderMom = list(linsolve(EQsys2ndOrdMom, [M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[0]]),
                                                         M_2(NoiseDict[nvec[1]] * NoiseDict[nvec[1]]),
                                                         M_2(NoiseDict[nvec[2]] * NoiseDict[nvec[2]]),
                                                         M_2(NoiseDict[nvec[3]] * NoiseDict[nvec[3]]),
                                                         M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[1]]),
                                                         M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[2]]),
                                                         M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[3]]),
                                                         M_2(NoiseDict[nvec[1]] * NoiseDict[nvec[2]]),
                                                         M_2(NoiseDict[nvec[1]] * NoiseDict[nvec[3]]),
                                                         M_2(NoiseDict[nvec[2]] * NoiseDict[nvec[3]])]))[0]  # only one set of solutions (if any) in linear system of equations; hence index [0]
        # ZsubDict = {}
        for noise1 in NoiseDict:
            for noise2 in NoiseDict:
                if M_2(NoiseDict[noise1] * NoiseDict[noise2]) in SOL_2ndOrderMom:
                    print('Solution for 2nd order noise moments NOT unique!')
                    return None, None, None, None
        # ZsubDict[M_2(NoiseDict[noise1] * NoiseDict[noise2])] = 0
        # if len(ZsubDict) > 0:
        #     SOL_2ndOrderMomMod = []
        #     for nn in range(len(SOL_2ndOrderMom)):
        #         SOL_2ndOrderMomMod.append(SOL_2ndOrderMom[nn].subs(ZsubDict))
        # SOL_2ndOrderMom = SOL_2ndOrderMomMod
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[0]])] = SOL_2ndOrderMom[0]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[1]] * NoiseDict[nvec[1]])] = SOL_2ndOrderMom[1]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[2]] * NoiseDict[nvec[2]])] = SOL_2ndOrderMom[2]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[3]] * NoiseDict[nvec[3]])] = SOL_2ndOrderMom[3]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[1]])] = SOL_2ndOrderMom[4]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[2]])] = SOL_2ndOrderMom[5]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[0]] * NoiseDict[nvec[3]])] = SOL_2ndOrderMom[6]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[1]] * NoiseDict[nvec[2]])] = SOL_2ndOrderMom[7]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[1]] * NoiseDict[nvec[3]])] = SOL_2ndOrderMom[8]
        SOL_2ndOrdMomDict[M_2(NoiseDict[nvec[2]] * NoiseDict[nvec[3]])] = SOL_2ndOrderMom[9]

    return SOL_1stOrderMom[0], NoiseSubs1stOrder, SOL_2ndOrdMomDict, NoiseSubs2ndOrder


def _getODEs_vKE(_get_orderedLists_vKE, stoich):
    """Return the ODE system derived from Master equation."""
    t = symbols('t')
    P = Function('P')
    V = Symbol(r'\overline{V}', real=True, constant=True)
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

    ODE = lhsODE - rhsODE

    nvec = []
    for key1 in stoich:
        for key2 in stoich[key1]:
            if key2 != 'rate' and stoich[key1][key2] != 'const':
                if key2 not in nvec:
                    nvec.append(key2)
    # for reactant in reactants:
    #    nvec.append(reactant)
    nvec = sorted(nvec, key=default_sort_key)
    if len(nvec) < 1 or len(nvec) > 4:
        print("van Kampen expansions works for 1, 2, 3 or 4 different reactants only")

        return
    # assert (len(nvec)==2 or len(nvec)==3 or len(nvec)==4), 'This module works for 2, 3 or 4 different reactants only'

    PhiDict = {}
    NoiseDict = {}
    for kk in range(len(nvec)):
        NoiseDict[nvec[kk]] = Symbol('eta_' + str(nvec[kk]))
        PhiDict[nvec[kk]] = Symbol('Phi_' + str(nvec[kk]))

    PhiSubDict = None
    if substring is not None:
        PhiSubDict = {}
        for sub in substring:
            PhiSubSym = Symbol('Phi_' + str(sub))
            PhiSubDict[PhiSubSym] = substring[sub]
        for key in PhiSubDict:
            for sym in PhiSubDict[key].atoms(Symbol):
                phisub = Symbol('Phi_' + str(sym))
                if sym in nvec:
                    symSub = phisub
                    PhiSubDict[key] = PhiSubDict[key].subs({sym: symSub})
                else:
                    # here we assume that if a reactant in the substitution
                    # string is not a time-dependent reactant it can only be
                    # the total number of reactants which is constant,
                    # i.e. 1=N / N
                    PhiSubDict[key] = PhiSubDict[key].subs({sym: 1})

    if len(Vlist_lhs) - 1 == 1:
        ode1 = 0
        for kk in range(len(ODE.args)):
            prod = 1
            for nn in range(len(ODE.args[kk].args) - 1):
                prod *= ODE.args[kk].args[nn]
            if ODE.args[kk].args[-1] == Derivative(P(NoiseDict[nvec[0]], t), NoiseDict[nvec[0]]):
                ode1 += prod
            else:
                print('Check ODE.args!')

        if PhiSubDict:
            ode1 = ode1.subs(PhiSubDict)

            for key in PhiSubDict:
                ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]], t), dict=True)
                ODEsys = {**ODE_1[0]}
        else:
            ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]], t), dict=True)
            ODEsys = {**ODE_1[0]}

    elif len(Vlist_lhs) - 1 == 2:
        ode1 = 0
        ode2 = 0
        for kk in range(len(ODE.args)):
            prod = 1
            for nn in range(len(ODE.args[kk].args) - 1):
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
                    ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]], t), dict=True)
                    ODEsys = {**ODE_2[0]}
                elif key == PhiDict[nvec[1]]:
                    ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]], t), dict=True)
                    ODEsys = {**ODE_1[0]}
                else:
                    ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]], t), dict=True)
                    ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]], t), dict=True)
                    ODEsys = {**ODE_1[0], **ODE_2[0]}
        else:
            ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]], t), dict=True)
            ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]], t), dict=True)
            ODEsys = {**ODE_1[0], **ODE_2[0]}

    elif len(Vlist_lhs) - 1 == 3:
        ode1 = 0
        ode2 = 0
        ode3 = 0
        for kk in range(len(ODE.args)):
            prod = 1
            for nn in range(len(ODE.args[kk].args) - 1):
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
                    ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]], t), dict=True)
                    ODE_3 = solve(ode3, Derivative(PhiDict[nvec[2]], t), dict=True)
                    ODEsys = {**ODE_2[0], **ODE_3[0]}
                elif key == PhiDict[nvec[1]]:
                    ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]], t), dict=True)
                    ODE_3 = solve(ode3, Derivative(PhiDict[nvec[2]], t), dict=True)
                    ODEsys = {**ODE_1[0], **ODE_3[0]}
                elif key == PhiDict[nvec[2]]:
                    ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]], t), dict=True)
                    ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]], t), dict=True)
                    ODEsys = {**ODE_1[0], **ODE_2[0]}
                else:
                    ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]], t), dict=True)
                    ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]], t), dict=True)
                    ODE_3 = solve(ode3, Derivative(PhiDict[nvec[2]], t), dict=True)
                    ODEsys = {**ODE_1[0], **ODE_2[0], **ODE_3[0]}

        else:
            ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]], t), dict=True)
            ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]], t), dict=True)
            ODE_3 = solve(ode3, Derivative(PhiDict[nvec[2]], t), dict=True)
            ODEsys = {**ODE_1[0], **ODE_2[0], **ODE_3[0]}

    elif len(Vlist_lhs) - 1 == 4:
        ode1 = 0
        ode2 = 0
        ode3 = 0
        ode4 = 0
        for kk in range(len(ODE.args)):
            prod = 1
            for nn in range(len(ODE.args[kk].args) - 1):
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
                    ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]], t), dict=True)
                    ODE_3 = solve(ode3, Derivative(PhiDict[nvec[2]], t), dict=True)
                    ODE_4 = solve(ode4, Derivative(PhiDict[nvec[3]], t), dict=True)
                    ODEsys = {**ODE_2[0], **ODE_3[0], **ODE_4[0]}
                elif key == PhiDict[nvec[1]]:
                    ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]], t), dict=True)
                    ODE_3 = solve(ode3, Derivative(PhiDict[nvec[2]], t), dict=True)
                    ODE_4 = solve(ode4, Derivative(PhiDict[nvec[3]], t), dict=True)
                    ODEsys = {**ODE_1[0], **ODE_3[0], **ODE_4[0]}
                elif key == PhiDict[nvec[2]]:
                    ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]], t), dict=True)
                    ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]], t), dict=True)
                    ODE_4 = solve(ode4, Derivative(PhiDict[nvec[3]], t), dict=True)
                    ODEsys = {**ODE_1[0], **ODE_2[0], **ODE_4[0]}
                elif key == PhiDict[nvec[3]]:
                    ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]], t), dict=True)
                    ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]], t), dict=True)
                    ODE_3 = solve(ode3, Derivative(PhiDict[nvec[2]], t), dict=True)
                    ODEsys = {**ODE_1[0], **ODE_2[0], **ODE_3[0]}
                else:
                    ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]], t), dict=True)
                    ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]], t), dict=True)
                    ODE_3 = solve(ode3, Derivative(PhiDict[nvec[2]], t), dict=True)
                    ODE_4 = solve(ode4, Derivative(PhiDict[nvec[3]], t), dict=True)
                    ODEsys = {**ODE_1[0], **ODE_2[0], **ODE_3[0], **ODE_4[0]}

        else:
            ODE_1 = solve(ode1, Derivative(PhiDict[nvec[0]], t), dict=True)
            ODE_2 = solve(ode2, Derivative(PhiDict[nvec[1]], t), dict=True)
            ODE_3 = solve(ode3, Derivative(PhiDict[nvec[2]], t), dict=True)
            ODE_4 = solve(ode4, Derivative(PhiDict[nvec[3]], t), dict=True)
            ODEsys = {**ODE_1[0], **ODE_2[0], **ODE_3[0], **ODE_4[0]}

    else:
        print('Not implemented yet.')

    return ODEsys


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
                    reactDict[reactant] = [rule.lhsReactants.count(reactant),
                                           rule.rhsReactants.count(reactant)]
        for reactant in rule.rhsReactants:
            if reactant != 1:
                if reactant not in rule.lhsReactants:
                    if reactant not in const_reactants:
                        reactDict[reactant] = [rule.lhsReactants.count(reactant),
                                               rule.rhsReactants.count(reactant)]
        stoich[ReactionNr.__next__()] = reactDict

    return stoich
