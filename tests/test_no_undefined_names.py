"""No script may reference a name that does not exist.

This is a small static check, and it exists because of a real bug: when the
learned-model maths was moved into `src/windmill/nbm.py`, the module-level
`ALL_COLUMNS` declaration was dropped from the script while the line that
wrote to it stayed. `scripts/pipeline/07_lightgbm_nbm.py` then died with a
NameError the first time it was run.

Nothing caught it, because:
  * `compileall` only checks syntax, and a NameError is a RUNTIME error;
  * pytest does not import the pipeline scripts;
  * CI ran the tests but not the pipeline.

So the arm's committed result was still the pre-refactor number, and the
script that should have reproduced it was broken. That is the worst kind of
failure: the artefact looks right and the code that made it does not run.

This check finds names that are read but never bound anywhere in the file and
are not builtins. It is not a full linter -- it is deliberately narrow, so it
has no false positives on this codebase.
"""

from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PIPELINE = SCRIPTS / "pipeline"
DIAGNOSTICS = SCRIPTS / "diagnostics"
#: The shared library too. The R_DRY bug lived HERE, not in a script: a
#: function was moved into conditioning.py without the constants it used, and
#: only the scripts that happened to call it ever found out.
LIBRARY = ROOT / "src" / "windmill"

BUILTINS = set(dir(builtins))
#: Module dunders, which always exist at runtime but are not "defined" in the
#: file -- __file__ above all.
DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__",
           "__loader__", "__builtins__", "__debug__"}


def _bound_names(tree: ast.Module) -> set[str]:
    """Every name the module binds somewhere: imports, assignments, defs."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                bound.add(a.asname or a.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, (ast.comprehension,)):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    bound.add(n.id)
    return bound


def _read_names(tree: ast.Module) -> set[str]:
    """Every name the module READS."""
    read: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            read.add(node.id)
    return read


@pytest.mark.parametrize("script", sorted(PIPELINE.glob("*.py")) +
                         sorted(DIAGNOSTICS.glob("*.py")) +
                         [ROOT / "app.py"],
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_script_defines_every_name_it_uses(script: Path):
    """Every name a script reads must be bound in that script (or a builtin)."""
    tree = ast.parse(script.read_text(), filename=str(script))
    bound = _bound_names(tree) | BUILTINS | DUNDERS
    undefined = sorted(n for n in _read_names(tree) if n not in bound)
    assert not undefined, (
        f"{script.relative_to(ROOT)} reads names it never defines: {undefined}. "
        f"This is the class of bug that left the learned-model arm's script "
        f"broken while its committed result still looked right."
    )


@pytest.mark.parametrize("script", sorted(PIPELINE.glob("*.py")) +
                         sorted(DIAGNOSTICS.glob("*.py")) +
                         [ROOT / "app.py"],
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_script_is_syntactically_valid(script: Path):
    ast.parse(script.read_text(), filename=str(script))


@pytest.mark.parametrize("module", sorted(LIBRARY.glob("*.py")),
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_library_module_defines_every_name_it_uses(module: Path):
    """The shared library gets the same check as the scripts.

    A constant left behind when a function was moved into `src/windmill/`
    is invisible to a check that only looks at `scripts/`, and it only
    surfaces when a caller happens to reach that line.
    """
    tree = ast.parse(module.read_text(), filename=str(module))
    bound = _bound_names(tree) | BUILTINS | DUNDERS
    undefined = sorted(n for n in _read_names(tree) if n not in bound)
    assert not undefined, (
        f"{module.relative_to(ROOT)} reads names it never defines: {undefined}"
    )
