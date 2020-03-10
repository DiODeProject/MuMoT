import os

from mumot.models import parseModel

EXPRESSION_STRS = [
    "U -> A : g_A",
    "U -> B : g_B",
    "A -> U : a_A",
    "B -> U : a_B",
    "A + U -> A + A : r_A",
    "B + U -> B + B : r_B",
    "A + B -> A + U : s",
    "A + B -> B + U : s"]


def test_parse_model_from_cell_contents():
    """Assert we can instantiate a MuMoTmodel from the contents of a Notebook
    cell that uses the %%model cell magic."""
    parseModel(r"\n".join(
        ["get_ipython().run_cell_magic('model', '', '$"] +
        EXPRESSION_STRS +
        ["$", "')"]))


def test_parse_model_from_str():
    """Assert we can instantiate a MuMoTmodel from a multi-line string."""
    parseModel(os.linesep.join(EXPRESSION_STRS))
