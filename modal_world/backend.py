from __future__ import annotations

from abc import ABC, abstractmethod

from .contracts import Capability, WorldRequest, WorldResult


class WorldBackendError(RuntimeError):
    pass


class UnsupportedOperation(WorldBackendError):
    pass


class BackendUnavailable(WorldBackendError):
    pass


class WorldBackend(ABC):
    name: str

    @property
    @abstractmethod
    def capability(self) -> Capability:
        raise NotImplementedError

    @abstractmethod
    def run(self, request: WorldRequest) -> WorldResult:
        raise NotImplementedError
