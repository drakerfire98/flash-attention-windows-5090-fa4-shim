"""Minimal Windows fallback for Unix-only fcntl used by the local runtime package."""

LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8


def flock(_fd, _operation):
    return None
