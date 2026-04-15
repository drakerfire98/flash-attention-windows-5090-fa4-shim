"""CUTLASS utils wrapper for native FA4 import probing."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


_SHIM_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
_REAL_PYTHON_ROOT = (
    _WORKSPACE_ROOT / "third_party" / "flash-attention-for-windows" / "csrc" / "cutlass" / "python"
)
_REAL_PACKAGE_DIR = _REAL_PYTHON_ROOT / "cutlass" / "utils"
_REAL_INIT = _REAL_PACKAGE_DIR / "__init__.py"

if str(_REAL_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_REAL_PYTHON_ROOT))

__path__ = [str(_SHIM_DIR), str(_REAL_PACKAGE_DIR)]
__file__ = str(_REAL_INIT)

exec(compile(_REAL_INIT.read_text(encoding="utf-8"), __file__, "exec"), globals())


class LayoutEnum(Enum):
    ROW_MAJOR = "row_major"
    COLUMN_MAJOR = "column_major"

    @classmethod
    def from_tensor(cls, tensor):
        del tensor
        return cls.ROW_MAJOR

    def is_m_major_c(self):
        return True

    def is_n_major_c(self):
        return False


class HardwareInfo:
    def get_max_active_clusters(self, cluster_size):
        del cluster_size
        return 1


class SmemAllocator:
    def __init__(self, *args, **kwargs):
        del args, kwargs


class TmemAllocator:
    def __init__(self, *args, **kwargs):
        del args, kwargs


class TensorMapManager:
    def __init__(self, *args, **kwargs):
        del args, kwargs


@dataclass
class WorkTileInfo:
    tile_coord_mnkl: object
    is_valid: object


def get_smem_capacity_in_bytes(arch):
    del arch
    return 0


def get_num_tmem_alloc_cols(tensor):
    del tensor
    return 0

