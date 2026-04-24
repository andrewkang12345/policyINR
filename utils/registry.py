from typing import Callable, Dict


class Registry:
    """Tiny registry for decoupling config names from implementations."""

    def __init__(self, name: str):
        self.name = name
        self._store: Dict[str, Callable] = {}

    def register(self, key: str):
        def deco(fn: Callable):
            if key in self._store:
                raise KeyError(f"{self.name}: '{key}' already registered")
            self._store[key] = fn
            return fn
        return deco

    def get(self, key: str) -> Callable:
        if key not in self._store:
            raise KeyError(
                f"{self.name}: '{key}' not found. Available: {sorted(self._store)}"
            )
        return self._store[key]

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def keys(self):
        return self._store.keys()


DATASETS = Registry("datasets")
MODELS = Registry("models")
SHIFTS = Registry("shifts")
