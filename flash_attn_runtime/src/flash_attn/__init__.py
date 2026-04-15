"""Repo-owned FlashAttention namespace overlay for the Windows FA4 probe path."""

from __future__ import annotations

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "2.8.4"

