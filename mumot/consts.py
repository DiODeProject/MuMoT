"""Constants (inc. symbols) used in MuMoT."""

from enum import Enum

from sympy.parsing.latex import parse_latex

MAX_RANDOM_SEED = 2147483647

EMPTYSET_SYMBOL = parse_latex('1')

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

LINE_COLOR_LIST = ['b', 'g', 'r', 'c', 'm', 'y', 'grey', 'orange', 'k']
MULTIPLOT_COLUMNS = 2


class NetworkType(Enum):
    """Enumeration of possible network types."""

    FULLY_CONNECTED = 0
    ERSOS_RENYI = 1
    BARABASI_ALBERT = 2
    SPACE = 3
    DYNAMIC = 4
