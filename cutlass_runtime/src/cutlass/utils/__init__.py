"""Runtime-owned CUTLASS utils compatibility surface for Windows FA4 probing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from _probe_helpers import module_getattr


class LayoutEnum(Enum):
    ROW_MAJOR = "row_major"
    COLUMN_MAJOR = "column_major"

    @classmethod
    def from_tensor(cls, tensor):
        del tensor
        return cls.ROW_MAJOR

    def is_m_major_c(self):
        return self is type(self).ROW_MAJOR

    def is_n_major_c(self):
        return self is type(self).COLUMN_MAJOR


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


def alignment_or_default(*args, **kwargs):
    del args, kwargs
    return 1


def calculate_smem_usage(*args, **kwargs):
    del args, kwargs
    return 0


def calculate_smem_usage_per_stage(*args, **kwargs):
    del args, kwargs
    return 0


def valid_cluster_shape(*args, **kwargs):
    del args, kwargs
    return True


def valid_schedule(*args, **kwargs):
    del args, kwargs
    return True


def valid_stage_count(*args, **kwargs):
    del args, kwargs
    return True


def update_alignment(*args, **kwargs):
    del args, kwargs
    return None


def get_smem_capacity_in_bytes(arch):
    del arch
    return 0


def get_num_tmem_alloc_cols(tensor):
    del tensor
    return 0


__all__ = [
    "HardwareInfo",
    "LayoutEnum",
    "SmemAllocator",
    "TensorMapManager",
    "TmemAllocator",
    "WorkTileInfo",
    "alignment_or_default",
    "calculate_smem_usage",
    "calculate_smem_usage_per_stage",
    "get_num_tmem_alloc_cols",
    "get_smem_capacity_in_bytes",
    "update_alignment",
    "valid_cluster_shape",
    "valid_schedule",
    "valid_stage_count",
]

__getattr__ = module_getattr("cutlass.utils")

