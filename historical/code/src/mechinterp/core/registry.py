"""
Registry pattern for pluggable components.

Allows registering and retrieving models, tasks, and methods by name.
"""

from typing import Any, Callable, Dict, Optional, Type, TypeVar

T = TypeVar("T")


class Registry:
    """Generic registry for components."""

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Any] = {}

    def register(self, name: str, obj: Any = None) -> Callable:
        """
        Register a component.

        Can be used as decorator:
            @registry.register("my_component")
            class MyComponent:
                pass

        Or directly:
            registry.register("my_component", MyComponent)
        """
        if obj is not None:
            self._registry[name] = obj
            return obj

        def decorator(cls: T) -> T:
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Any:
        """Get a registered component by name."""
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"'{name}' not found in {self.name} registry. "
                f"Available: {available}"
            )
        return self._registry[name]

    def build(self, name: str, *args, **kwargs) -> Any:
        """Build (instantiate) a registered component."""
        cls = self.get(name)
        return cls(*args, **kwargs)

    def list(self) -> list:
        """List all registered component names."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"Registry({self.name}, components={self.list()})"


# Global registries
MODEL_REGISTRY = Registry("models")
TASK_REGISTRY = Registry("tasks")
TIER_REGISTRY = Registry("tiers")
BASELINE_REGISTRY = Registry("baselines")


def register_model(name: str):
    """Decorator to register a model wrapper."""
    return MODEL_REGISTRY.register(name)


def register_task(name: str):
    """Decorator to register a task."""
    return TASK_REGISTRY.register(name)


def register_tier(name: str):
    """Decorator to register a tier analyzer."""
    return TIER_REGISTRY.register(name)


def register_baseline(name: str):
    """Decorator to register a baseline method."""
    return BASELINE_REGISTRY.register(name)


def get_model(name: str, *args, **kwargs):
    """Get and instantiate a model wrapper."""
    return MODEL_REGISTRY.build(name, *args, **kwargs)


def get_task(name: str, *args, **kwargs):
    """Get and instantiate a task."""
    return TASK_REGISTRY.build(name, *args, **kwargs)


def get_tier(name: str, *args, **kwargs):
    """Get and instantiate a tier analyzer."""
    return TIER_REGISTRY.build(name, *args, **kwargs)


def get_baseline(name: str, *args, **kwargs):
    """Get and instantiate a baseline method."""
    return BASELINE_REGISTRY.build(name, *args, **kwargs)


def list_models() -> list:
    """List available models."""
    return MODEL_REGISTRY.list()


def list_tasks() -> list:
    """List available tasks."""
    return TASK_REGISTRY.list()


def list_tiers() -> list:
    """List available tiers."""
    return TIER_REGISTRY.list()


def list_baselines() -> list:
    """List available baselines."""
    return BASELINE_REGISTRY.list()
