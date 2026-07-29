"""Mojo implementation of the in-memory two-dimensional ``rtree`` API."""

from . import core, index
from .core import RTreeError
from .index import Index, Item, Property

Rtree = Index

__all__ = ["Index", "Rtree", "Item", "Property", "RTreeError", "core", "index"]
__version__ = "0.1.0"
