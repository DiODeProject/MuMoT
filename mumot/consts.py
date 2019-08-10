"""Constants (inc. symbols) used in MuMoT."""

from .process_latex.process_latex import process_sympy

MAX_RANDOM_SEED = 2147483647

EMPTYSET_SYMBOL = process_sympy('1')

GREEK_LETT_LIST_1 = ['alpha', 'beta', 'gamma', 'Gamma', 'delta', 'Delta',
                     'epsilon', 'zeta', 'theta', 'Theta', 'iota', 'kappa',
                     'lambda', 'Lambda', 'mu', 'xi', 'Xi', 'pi', 'Pi', 'rho',
                     'sigma', 'Sigma', 'tau', 'upsilon', 'Upsilon', 'phi',
                     'Phi', 'chi', 'psi', 'Psi', 'omega', 'Omega', 'varrho',
                     'vartheta', 'varepsilon', 'varphi']
GREEK_LETT_LIST_2 = ['\\' + GreekLett for GreekLett in GREEK_LETT_LIST_1]
GREEK_LETT_RESERVED_LIST = ['\\eta', '\\nu', '\\Phi', '(\\eta)', '(\\nu)',
                            '(\\Phi)']
GREEK_LETT_RESERVED_LIST_PRINT = ['eta', 'nu', 'Phi']

INITIAL_RATE_VALUE = 0.5
RATE_BOUND = 10.0
RATE_STEP = 0.1

INITIAL_COND_INIT_VAL = 0.0
INITIAL_COND_INIT_BOUND = 1.0


