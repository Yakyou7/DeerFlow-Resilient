"""Aimem hierarchical memory backend package."""

from .aimem_manager import AimemMemoryManager

MANAGER_CLASS = AimemMemoryManager

__all__ = ["MANAGER_CLASS", "AimemMemoryManager"]
