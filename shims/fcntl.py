"""Minimal Windows fallback for Unix-only fcntl used by quack cache_utils.

This keeps import-time behavior alive on Windows. It does not provide real file
locking semantics.
"""

LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8


def flock(_fd, _operation):
    return None
