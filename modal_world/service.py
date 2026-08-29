from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import Operation, WorldRequest
from .providers import register_builtin_backends
from .registry import get_backend, list_backends

register_builtin_backends()


def capabilities() -> list[dict[str, Any]]:
    result = []
    for name in list_backends():
        capability = get_backend(name).capability
        result.append(
            {
                "backend": capability.backend,
                "operations": sorted(op.value for op in capability.operations),
                "inputs": sorted(capability.inputs),
                "outputs": sorted(capability.outputs),
                "notes": list(capability.notes),
            }
        )
    return result


def execute(
    *,
    backend: str,
    operation: str,
    input_path: str,
    output_dir: str,
    options: dict[str, Any] | None = None,
):
    request = WorldRequest(
        operation=Operation(operation),
        input_path=Path(input_path),
        output_dir=Path(output_dir),
        options=options or {},
    )
    return get_backend(backend).run(request)
