import base64
import os
import shutil

import pytest

from mumot import exceptions
from mumot.models import _embedSVGimages, parseModel

MODEL = os.linesep.join([
    "U -> A : g_A",
    "A -> U : a_A",
    "A + U -> A + A : r_A"])


@pytest.mark.parametrize('subsString', [
    'g_A = \beta', 'g_A = \nu', 'g_A = \theta', 'g_A = \rho', 'g_A = \alpha'])
def test_substitute_rejects_latex_mangled_by_python_escapes(subsString):
    """In an ordinary string e.g. '\\beta' starts with a backspace: say so
    rather than silently substituting 'eta'."""
    with pytest.raises(exceptions.MuMoTSyntaxError, match='raw string'):
        parseModel(MODEL).substitute(subsString)


def test_substitute_accepts_latex_in_raw_strings():
    model = parseModel(MODEL).substitute(r'g_A = \beta + \nu, r_A = \theta')
    assert {str(s) for s in model._rates} == {'a_{A}', 'beta', 'nu', 'theta'}


def test_embed_svg_images(tmp_path, monkeypatch):
    """Links to local image files become data URIs; anything else is kept."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'label.png').write_bytes(b'not really a png')
    svg = ('<svg><image xlink:href="label.png" width="5px"/>'
           '<image href="missing.png"/><a xlink:href="https://example.com/"/></svg>')
    data = base64.b64encode(b'not really a png').decode()
    assert _embedSVGimages(svg) == (
        f'<svg><image xlink:href="data:image/png;base64,{data}" width="5px"/>'
        '<image href="missing.png"/><a xlink:href="https://example.com/"/></svg>')


@pytest.mark.skipif(not (shutil.which('latex') and shutil.which('dvipng') and shutil.which('dot')),
                    reason='needs LaTeX, dvipng and Graphviz')
def test_visualise_svg_embeds_label_images(tmp_path, monkeypatch):
    """JupyterLab and Notebook 7 show SVG output as an image, which cannot
    load the separate label image files, so they must be embedded."""
    monkeypatch.chdir(tmp_path)
    model = parseModel(MODEL)  # keep it: deleting a model deletes its image files
    svg = model.visualise()._repr_mimebundle_()["image/svg+xml"]
    assert svg.count('href="data:image/png;base64,') >= 5  # 2 nodes, 3 edges
    assert '__mumot_files__' not in svg
