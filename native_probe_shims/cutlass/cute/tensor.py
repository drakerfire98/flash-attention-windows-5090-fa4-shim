"""Minimal tensor placeholders for native FA4 import probing."""

from __future__ import annotations


class TensorSSA:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

