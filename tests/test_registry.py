from modal_world.backends.hyworld2 import HYWorld2Backend
from modal_world.providers import register_builtin_backends
from modal_world.registry import get_backend, list_backends


def test_builtin_registry_contains_hyworld2():
    register_builtin_backends()
    assert "hyworld2" in list_backends()
    assert isinstance(get_backend("HYWORLD2"), HYWorld2Backend)
