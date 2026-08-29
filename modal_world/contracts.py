from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Operation(str, Enum):
    RECONSTRUCT = "reconstruct"
    GENERATE = "generate"


@dataclass(frozen=True)
class Capability:
    backend: str
    operations: frozenset[Operation]
    inputs: frozenset[str]
    outputs: frozenset[str]
    notes: tuple[str, ...] = ()

    def supports(self, operation: Operation) -> bool:
        return operation in self.operations


@dataclass(frozen=True)
class WorldRequest:
    operation: Operation
    input_path: Path
    output_dir: Path
    options: Mapping[str, Any] = field(default_factory=dict)

    def normalized(self) -> WorldRequest:
        return WorldRequest(
            operation=self.operation,
            input_path=self.input_path.expanduser().resolve(),
            output_dir=self.output_dir.expanduser().resolve(),
            options=dict(self.options),
        )


@dataclass(frozen=True)
class Artifact:
    kind: str
    path: Path
    media_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldResult:
    backend: str
    operation: Operation
    artifacts: tuple[Artifact, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
