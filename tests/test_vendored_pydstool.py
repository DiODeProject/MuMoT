"""Tests for the parts of the bundled PyDSTool that were modernised."""
import pytest

from mumot._vendor.pydstool import _cst
from mumot._vendor.pydstool.parseUtils import convertPowers, string2ast, ast2string


def _chain(leaf):
    """Full grammar chain from `test` down to an atom, as the old parser produced."""
    node = ['atom', leaf]
    for name in ('atom_expr', 'power', 'factor', 'term', 'arith_expr', 'shift_expr',
                 'and_expr', 'xor_expr', 'expr', 'comparison', 'not_test', 'and_test',
                 'or_test', 'test'):
        node = [name, node]
    return node


def test_cst_matches_old_parser_module():
    # reference output of sym2name(parser.expr('x').tolist()) under Python 3.8
    assert _cst.expr_tolist('x') == [
        'eval_input', ['testlist', _chain(['NAME', 'x'])], ['NEWLINE', ''], ['ENDMARKER', '']]


@pytest.mark.parametrize('expr, shortlist', [
    ('a**2', ['power', ['NAME', 'a'], ['DOUBLESTAR', '**'], ['NUMBER', '2']]),
    ('-x', ['factor', ['MINUS', '-'], ['NAME', 'x']]),
    ('a+b*c', ['arith_expr', ['NAME', 'a'], ['PLUS', '+'],
               ['term', ['NAME', 'b'], ['STAR', '*'], ['NAME', 'c']]]),
    ('f(x)', ['atom_expr', ['NAME', 'f'], ['trailer', ['LPAR', '('], ['NAME', 'x'], ['RPAR', ')']]]),
    ('a not in b', ['comparison', ['NAME', 'a'], ['comp_op', ['NAME', 'not'], ['NAME', 'in']], ['NAME', 'b']]),
])
def test_string2ast(expr, shortlist):
    assert string2ast(expr) == shortlist
    assert ast2string(string2ast(expr)).replace(' ', '') == expr.replace(' ', '')


@pytest.mark.parametrize('bad', [' x', 'lambda: 1', 'a.True', '(', 'x = 1'])
def test_cst_rejects_what_parser_rejected(bad):
    with pytest.raises(SyntaxError):
        _cst.expr_tolist(bad)


def test_convert_powers():
    assert convertPowers('X000**2*P001', 'pow') == 'pow(X000,2)*P001'
    assert convertPowers('pow(x,3)', '**') == 'x**3'
