"""Symbolic derivation of model equations (ODEs, Master equation, van Kampen expansion).

These functions only depend on SymPy and are used by :mod:`mumot.models`.
"""

import sympy
from sympy import (
    default_sort_key,
    Derivative,
    factorial,
    Function,
    lambdify,
    Symbol,
    symbols,
)

from ._sympy_compat import simplify


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
