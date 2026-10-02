"""Symbolic results must match those of MuMoT 1.2.2 with SymPy 1.4.

``symbolic_reference.json`` holds ``srepr`` dumps of the model getters'
results produced by the last release (Python 3.8, SymPy 1.4, pinned
dependencies).  Newer SymPy versions order terms differently and changed
the behaviour of e.g. ``simplify`` and ``subs``, so results are compared for
mathematical equality rather than textual identity.
"""
import json
import pathlib

import pytest
import sympy

import mumot

REFERENCE = json.loads((pathlib.Path(__file__).parent / 'symbolic_reference.json').read_text())
SYMPY_NAMESPACE = {name: getattr(sympy, name) for name in dir(sympy)}

MODELS = {
    'm1': r"""
U -> A : g_1
U -> B : g_2
A -> U : a_1
B -> U : a_2
A + U -> A + A : r_1
B + U -> B + B : r_2
A + B -> A + U : s
A + B -> B + U : s
""",
    'm7': r"""
U -> A : g_1
U -> B : g_2
U -> C : g_3
A -> U : a_1
B -> U : a_2
C -> U : a_3
A + U -> A + A : r_1
B + U -> B + B : r_2
C + U -> C + C : r_3
A + B -> A + U : s
A + B -> B + U : s
A + C -> A + U : s
A + C -> C + U : s
B + C -> B + U : s
B + C -> C + U : s
""",
    'm9': r"""
(\alpha) -> X : \gamma
X + X + Y -> X + X + X : \chi
(\beta) + X -> Y + \emptyset : \delta
X -> \emptyset : \xi
""",
    'o1': r"""
\emptyset + \alpha_\beta -> \alpha_\beta + \alpha_\beta : r
\alpha_\beta + \alpha_\beta + \alpha_\beta -> \emptyset + \emptyset + \emptyset : b
""",
    'o2': r"""
(A) -> X : k
X + X -> \emptyset + \emptyset : h
""",
    'sir': r"""
S + I -> I + I : \beta
I -> R : \gamma
R -> S : \alpha
""",
}


def _model(name):
    if name == 'm4':
        model = mumot.parseModel(MODELS['m1'])
        model = model.substitute('a_1 = 1/v_1, a_2 = 1/v_2, g_1 = v_1, g_2 = v_2, r_1 = v_1, r_2 = v_2')
        model = model.substitute('v_1 = \\mu + \\Delta/2, v_2 = \\mu - \\Delta/2')
        return model.substitute('U = N - \\A - \\B')
    if name == 'm8':
        return mumot.parseModel(MODELS['m7']).substitute('U = N - A - B - C')
    return mumot.parseModel(MODELS[name])


def _equal(a, b):
    if isinstance(a, dict):
        return (isinstance(b, dict) and a.keys() == b.keys()
                and all(_equal(v, b[k]) for k, v in a.items()))
    if isinstance(a, (list, tuple)):
        return (isinstance(b, (list, tuple)) and len(a) == len(b)
                and all(_equal(x, y) for x, y in zip(a, b)))
    if isinstance(a, str) and isinstance(b, sympy.Symbol):
        # LaTeX display substitutions are now Symbols (SymPy 1.4 silently ignored strings)
        return a == b.name
    if isinstance(a, sympy.Basic) and isinstance(b, sympy.Basic):
        if a == b:
            return True
        if isinstance(a, sympy.Equality) and isinstance(b, sympy.Equality):
            a, b = a.lhs - a.rhs, b.lhs - b.rhs
        return sympy.simplify(a - b, doit=False) == 0
    return a == b


@pytest.mark.parametrize('key', sorted(REFERENCE))
def test_matches_reference(key):
    model_name, getter = key.split('.')
    kwargs = {'method': 'vanKampen'} if getter.endswith('vanKampen') else {}
    getter = getter.removesuffix('vanKampen')
    result = getattr(_model(model_name), getter)(**kwargs)
    expected = eval(REFERENCE[key], SYMPY_NAMESPACE)
    assert _equal(expected, result)
