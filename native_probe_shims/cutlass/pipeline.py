"""Minimal pipeline class scaffolding for native FA4 import probing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from _probe_helpers import module_getattr


class Agent(Enum):
    Thread = "thread"
    ThreadBlock = "thread_block"
    ThreadBlockCluster = "thread_block_cluster"


class PipelineUserType(Enum):
    Producer = "producer"
    Consumer = "consumer"


@dataclass(frozen=True)
class PipelineState:
    stages: int
    count: object
    index: object
    phase: object

    def __post_init__(self):
        object.__setattr__(self, "_count", self.count)
        object.__setattr__(self, "_index", self.index)
        object.__setattr__(self, "_phase", self.phase)


@dataclass(frozen=True)
class NamedBarrier:
    barrier_id: int = 0
    num_threads: int = 0

    @staticmethod
    def create(*args, **kwargs):
        return NamedBarrier(*args, **kwargs)


@dataclass(frozen=True)
class CooperativeGroup:
    agent: object = None
    group_size: int = 1

    @property
    def size(self) -> int:
        return self.group_size


@dataclass(frozen=True)
class MbarrierArray:
    size: int = 0


@dataclass(frozen=True)
class PipelineOp:
    pass


@dataclass(frozen=True)
class _BasePipeline:
    @staticmethod
    def create(*args, **kwargs):
        del args, kwargs
        return _BasePipeline()

    def producer_acquire(self, *args, **kwargs):
        del args, kwargs

    def producer_commit(self, *args, **kwargs):
        del args, kwargs

    def consumer_wait(self, *args, **kwargs):
        del args, kwargs

    def consumer_release(self, *args, **kwargs):
        del args, kwargs


class PipelineAsync(_BasePipeline):
    pass


class PipelineCpAsync(_BasePipeline):
    pass


class PipelineTmaAsync(_BasePipeline):
    pass


class PipelineTmaUmma(_BasePipeline):
    pass


class PipelineUmmaAsync(_BasePipeline):
    pass


class PipelineAsyncUmma(_BasePipeline):
    pass


def make_pipeline_state(user_type: PipelineUserType, stages: int):
    if user_type is PipelineUserType.Producer:
        return PipelineState(stages=stages, count=0, index=0, phase=1)
    return PipelineState(stages=stages, count=0, index=0, phase=0)


def agent_sync(*args, **kwargs):
    del args, kwargs


def pipeline_init_arrive(*args, **kwargs):
    del args, kwargs


def pipeline_init_wait(*args, **kwargs):
    del args, kwargs


__getattr__ = module_getattr("cutlass.pipeline")
