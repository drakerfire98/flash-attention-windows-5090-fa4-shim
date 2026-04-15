"""Helpers for isolated native FA4 import probing on Windows.

These shims are intentionally permissive and import-focused. They are used to
advance the real native import chain far enough to reveal the next honest
blocker without changing the stable fallback implementation under ``shims/``.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
import sys


class ProbePlaceholder:
    """A lightweight object that tolerates chained attribute access and calls."""

    def __init__(self, name: str):
        self._name = name

    def __repr__(self) -> str:
        return f"<ProbePlaceholder {self._name}>"

    def __getattr__(self, name: str) -> "ProbePlaceholder":
        return ProbePlaceholder(f"{self._name}.{name}")

    def __call__(self, *args, **kwargs):
        del args, kwargs
        return ProbePlaceholder(f"{self._name}()")

    def __getitem__(self, item):
        return ProbePlaceholder(f"{self._name}[{item!r}]")

    def __iter__(self):
        return iter(())

    def __bool__(self) -> bool:
        return False

    def __or__(self, other):
        del other
        return self

    def __ror__(self, other):
        del other
        return self


def passthrough_decorator(*args, **kwargs):
    """No-op decorator used for import-time probe shims."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def _decorate(func):
        return func

    return _decorate


def module_getattr(prefix: str):
    def _getattr(name: str):
        return ProbePlaceholder(f"{prefix}.{name}")

    return _getattr


def try_dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def find_package_init_candidates(
    package_name: str,
    *,
    exclude_roots: list[Path] | None = None,
) -> list[Path]:
    candidates: list[Path] = []
    exclude_roots = [root.resolve() for root in (exclude_roots or [])]
    package_parts = package_name.split(".")
    relative_init = Path(*package_parts) / "__init__.py"

    for entry in sys.path:
        try:
            base = Path(entry).resolve()
        except OSError:
            continue
        candidate = (base / relative_init).resolve()
        if not candidate.is_file():
            continue
        if any(_is_relative_to(candidate, root) for root in exclude_roots):
            continue
        candidates.append(candidate)

    return candidates
