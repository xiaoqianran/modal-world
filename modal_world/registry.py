from __future__ import annotations

from collections.abc import Callable

from .backend import WorldBackend

BackendFactory = Callable[[], WorldBackend]
_REGISTRY: dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory, *, replace: bool = False) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("backend name must not be empty")
    if key in _REGISTRY and not replace:
        raise ValueError(f"backend already registered: {key}")
    _REGISTRY[key] = factory


def get_backend(name: str) -> WorldBackend:
    key = name.strip().lower()
    try:
        factory = _REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"unknown world backend {name!r}; available: {available}") from exc
    return factory()


def list_backends() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
