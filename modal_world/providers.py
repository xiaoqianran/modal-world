from .backends.hyworld2 import HYWorld2Backend
from .registry import register_backend


def register_builtin_backends() -> None:
    register_backend("hyworld2", HYWorld2Backend, replace=True)
