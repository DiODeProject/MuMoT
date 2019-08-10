from textwrap import dedent

from mumot.models import parseModel


def test_dummy_1():
    """A brief description of this test.

    More information about this test.
    """
    assert 1 == 1


def test_dummy_2():
    """A brief description of this test.

    More information about this test.
    """
    assert 2 == 2


def test_parse_model_from_cell_contents():
    """Assert we can instantiate a MuMoTmodel from the contents of a Notebook cell that uses the %%model cell magic."""
    s = "get_ipython().run_cell_magic('model', '', '$\\nU -> A : g_A\\nU -> B : g_B\\nA -> U : a_A\\nB -> U : a_B\\nA + U -> A + A : r_A\\nB + U -> B + B : r_B\\nA + B -> A + U : s\\nA + B -> B + U : s\\n$\\n')"
    parseModel(s)


def test_parse_model_from_str():
    """Assert we can instantiate a MuMoTmodel from a multi-line string."""
    s = dedent("""U -> A : g_A
        U -> B : g_B
        A -> U : a_A
        B -> U : a_B
        A + U -> A + A : r_A
        B + U -> B + B : r_B
        A + B -> A + U : s
        A + B -> B + U : s""")
    parseModel(s)
