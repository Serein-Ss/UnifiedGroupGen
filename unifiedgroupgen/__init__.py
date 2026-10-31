"""Unified group-constrained materials generation."""
from .group_constraints.compiler import Descriptor, OrbitSpec, compile_descriptor

__all__ = ["Descriptor", "OrbitSpec", "compile_descriptor"]
