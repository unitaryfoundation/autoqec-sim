"""Griffe extension for the API reference: Sphinx roles as inline code.

The package's docstrings carry reStructuredText roles such as
``:class:`autoqec_sim.graphs.Graph``` and ``:func:`ldpc_feasible```.
mkdocstrings does not resolve them, so they would render literally. This
rewrites each role to plain inline code at documentation build time; the
source docstrings are left as they are.
"""

from __future__ import annotations

import re

from griffe import Extension, Object

_ROLE = re.compile(r":(?:class|func|mod|meth|attr|data|exc|obj):`~?([^`]+)`")


def _strip_roles(text: str) -> str:
    return _ROLE.sub(lambda m: f"`{m.group(1)}`", text)


class SphinxRolesToCode(Extension):
    def on_instance(self, *, obj: Object, **kwargs: object) -> None:
        docstring = obj.docstring
        if docstring is not None and ":" in docstring.value:
            docstring.value = _strip_roles(docstring.value)
