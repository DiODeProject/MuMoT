"""Concrete syntax trees for Python expressions.

PyDSTool's expression manipulation code (``parseUtils``, ``Symbolic``) was
written against the concrete syntax trees produced by the standard library
``parser`` module, which was removed in Python 3.10.  This module provides
:func:`expr_tolist`, a small recursive-descent parser for the expression
subset of the Python 3.8 grammar that returns the same nested-list structure
as ``sym2name(parser.expr(source).tolist())`` did, i.e. with grammar symbol
and token *names* (``'arith_expr'``, ``'NAME'``, ...) rather than numeric ids.

Only expressions are supported; lambdas, comprehensions, ``yield``, ``await``
and assignment expressions raise :class:`SyntaxError`.
"""

import io
import keyword
import token
import tokenize

__all__ = ['expr_tolist']

_COMP_OPS = {'<', '>', '==', '>=', '<=', '!=', 'in', 'not', 'is'}
_SKIP = {tokenize.NL, tokenize.COMMENT}


def _tokens(source):
    toks = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type in _SKIP:
                continue
            if tok.type == tokenize.INDENT:
                raise IndentationError('unexpected indent')
            if tok.type == tokenize.ERRORTOKEN and not tok.string.isspace():
                raise SyntaxError('invalid token %r' % tok.string)
            if tok.type in (tokenize.ERRORTOKEN, tokenize.DEDENT):
                continue
            if tok.type == tokenize.OP:
                name = token.tok_name[tok.exact_type]
            elif tok.type in (tokenize.NEWLINE, tokenize.ENDMARKER):
                toks.append((token.tok_name[tok.type], ''))
                continue
            else:
                name = token.tok_name[tok.type]
            toks.append((name, tok.string))
    except tokenize.TokenError as err:
        raise SyntaxError(str(err)) from None
    if source.endswith('\n'):
        # the parser module saw an extra NEWLINE in this case
        toks.insert(-1, ('NEWLINE', ''))
    return toks


class _Parser:

    def __init__(self, source):
        self.toks = _tokens(source)
        self.pos = 0

    # -- token helpers ----------------------------------------------------
    def peek(self, offset=0):
        return self.toks[self.pos + offset]

    def at(self, *values):
        return self.peek()[1] in values and self.peek()[0] != 'STRING'

    def at_name(self, *values):
        typ, val = self.peek()
        return typ == 'NAME' and val in values

    def take(self):
        tok = self.peek()
        self.pos += 1
        return [tok[0], tok[1]]

    def expect(self, value):
        if not self.at(value):
            raise SyntaxError('expected %r, got %r' % (value, self.peek()[1]))
        return self.take()

    def at_test_start(self):
        typ, val = self.peek()
        if typ in ('NAME', 'NUMBER', 'STRING'):
            return not (typ == 'NAME' and keyword.iskeyword(val)
                        and val not in ('not', 'None', 'True', 'False',
                                        'lambda', 'await'))
        return val in ('(', '[', '{', '-', '+', '~', '...')

    # -- grammar ----------------------------------------------------------
    def eval_input(self):
        node = ['eval_input', self.testlist()]
        while self.peek()[0] == 'NEWLINE':
            node.append(self.take())
        if self.peek()[0] != 'ENDMARKER':
            raise SyntaxError('invalid syntax at %r' % self.peek()[1])
        node.append(self.take())
        return node

    def testlist(self):
        node = ['testlist', self.test()]
        while self.at(','):
            node.append(self.take())
            if not self.at_test_start():
                break
            node.append(self.test())
        return node

    def test(self):
        if self.at_name('lambda'):
            raise SyntaxError('lambda expressions are not supported')
        node = ['test', self.or_test()]
        if self.at_name('if'):
            node.append(self.take())
            node.append(self.or_test())
            if not self.at_name('else'):
                raise SyntaxError("expected 'else'")
            node.append(self.take())
            node.append(self.test())
        return node

    def namedexpr_test(self):
        node = ['namedexpr_test', self.test()]
        if self.at(':='):
            raise SyntaxError('assignment expressions are not supported')
        return node

    def or_test(self):
        node = ['or_test', self.and_test()]
        while self.at_name('or'):
            node.append(self.take())
            node.append(self.and_test())
        return node

    def and_test(self):
        node = ['and_test', self.not_test()]
        while self.at_name('and'):
            node.append(self.take())
            node.append(self.not_test())
        return node

    def not_test(self):
        if self.at_name('not'):
            return ['not_test', self.take(), self.not_test()]
        return ['not_test', self.comparison()]

    def comparison(self):
        node = ['comparison', self.expr()]
        while self.at(*_COMP_OPS) and (self.peek()[0] != 'NAME' or self.peek()[1] in _COMP_OPS):
            op = ['comp_op', self.take()]
            if op[1][1] == 'not':
                if not self.at_name('in'):
                    raise SyntaxError("expected 'in' after 'not'")
                op.append(self.take())
            elif op[1][1] == 'is' and self.at_name('not'):
                op.append(self.take())
            node.append(op)
            node.append(self.expr())
        return node

    def _binary(self, name, child, ops):
        node = [name, child()]
        while self.at(*ops) and self.peek()[0] != 'NAME':
            node.append(self.take())
            node.append(child())
        return node

    def expr(self):
        return self._binary('expr', self.xor_expr, ('|',))

    def xor_expr(self):
        return self._binary('xor_expr', self.and_expr, ('^',))

    def and_expr(self):
        return self._binary('and_expr', self.shift_expr, ('&',))

    def shift_expr(self):
        return self._binary('shift_expr', self.arith_expr, ('<<', '>>'))

    def arith_expr(self):
        return self._binary('arith_expr', self.term, ('+', '-'))

    def term(self):
        return self._binary('term', self.factor, ('*', '@', '/', '%', '//'))

    def factor(self):
        if self.at('+', '-', '~'):
            return ['factor', self.take(), self.factor()]
        return ['factor', self.power()]

    def power(self):
        node = ['power', self.atom_expr()]
        if self.at('**'):
            node.append(self.take())
            node.append(self.factor())
        return node

    def atom_expr(self):
        if self.at_name('await'):
            raise SyntaxError('await is not supported')
        node = ['atom_expr', self.atom()]
        while self.at('(', '[', '.'):
            node.append(self.trailer())
        return node

    def atom(self):
        typ, val = self.peek()
        if typ == 'STRING':
            node = ['atom']
            while self.peek()[0] == 'STRING':
                node.append(self.take())
            return node
        if typ == 'NUMBER' or (typ == 'NAME' and (not keyword.iskeyword(val)
                                                  or val in ('None', 'True', 'False'))):
            return ['atom', self.take()]
        if self.at('...'):
            return ['atom', self.take()]
        if self.at('('):
            node = ['atom', self.take()]
            if self.at_name('yield'):
                raise SyntaxError('yield is not supported')
            if not self.at(')'):
                node.append(self.testlist_comp())
            node.append(self.expect(')'))
            return node
        if self.at('['):
            node = ['atom', self.take()]
            if not self.at(']'):
                node.append(self.testlist_comp())
            node.append(self.expect(']'))
            return node
        if self.at('{'):
            raise SyntaxError('dict and set displays are not supported')
        raise SyntaxError('invalid syntax at %r' % val)

    def _star_or_named(self):
        if self.at('*'):
            return ['star_expr', self.take(), self.expr()]
        return self.namedexpr_test()

    def testlist_comp(self):
        node = ['testlist_comp', self._star_or_named()]
        if self.at_name('for', 'async'):
            raise SyntaxError('comprehensions are not supported')
        while self.at(','):
            node.append(self.take())
            if not (self.at_test_start() or self.at('*')):
                break
            node.append(self._star_or_named())
        return node

    def trailer(self):
        if self.at('('):
            node = ['trailer', self.take()]
            if not self.at(')'):
                node.append(self.arglist())
            node.append(self.expect(')'))
            return node
        if self.at('['):
            node = ['trailer', self.take(), self.subscriptlist()]
            node.append(self.expect(']'))
            return node
        node = ['trailer', self.expect('.')]
        if self.peek()[0] != 'NAME' or keyword.iskeyword(self.peek()[1]):
            raise SyntaxError('expected name after "."')
        node.append(self.take())
        return node

    def arglist(self):
        node = ['arglist', self.argument()]
        while self.at(','):
            node.append(self.take())
            if not (self.at_test_start() or self.at('*', '**')):
                break
            node.append(self.argument())
        return node

    def argument(self):
        if self.at('*', '**'):
            return ['argument', self.take(), self.test()]
        node = ['argument', self.test()]
        if self.at_name('for', 'async'):
            raise SyntaxError('comprehensions are not supported')
        if self.at('='):
            node.append(self.take())
            node.append(self.test())
        elif self.at(':='):
            raise SyntaxError('assignment expressions are not supported')
        return node

    def subscriptlist(self):
        node = ['subscriptlist', self.subscript()]
        while self.at(','):
            node.append(self.take())
            if not (self.at_test_start() or self.at(':')):
                break
            node.append(self.subscript())
        return node

    def subscript(self):
        node = ['subscript']
        if not self.at(':'):
            node.append(self.test())
            if not self.at(':'):
                return node
        node.append(self.take())
        if self.at_test_start():
            node.append(self.test())
        if self.at(':'):
            sliceop = ['sliceop', self.take()]
            if self.at_test_start():
                sliceop.append(self.test())
            node.append(sliceop)
        return node


def expr_tolist(source):
    """Return the concrete syntax tree of the expression ``source``.

    Equivalent to ``sym2name(parser.expr(source).tolist())`` under Python 3.8.
    """
    return _Parser(source).eval_input()
