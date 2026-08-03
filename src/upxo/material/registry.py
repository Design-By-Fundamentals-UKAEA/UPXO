"""
registry.py
============
Extensible, typed, provenance-tracking container for material data.

Replaces :class:`upxo._sup.ODDict.ObjectDataDictionary`, which flattened
every appended dataclass to a plain dict via ``dataclasses.asdict()`` the
instant it was stored, discarding all type information immediately and
offering no validation. :class:`MaterialRegistry` keeps typed instances and
lets new data categories be declared via :meth:`register_category` rather
than requiring a central ``build()`` function to be edited (import + append)
every time a category is added.

Validation is deliberately soft: unrecognized field values warn but never
block, so the registry stays usable for exploratory work and materials/
processing routes not yet on any curated list -- the point is catching
likely typos, not gatekeeping legitimate but unanticipated input.

Usage
-----
>>> from dataclasses import dataclass
>>> from upxo.material.registry import MaterialRegistry
>>>
>>> @dataclass
... class MaterialIdentity:
...     name: str = "cu"
...     alloy: str = "value"
>>>
>>> registry = MaterialRegistry()
>>> registry.register_category(
...     "MaterialIdentity", MaterialIdentity,
...     known_values={"name": {"OFHC-Cu", "CuCrZr"}})
>>> registry.ingest(
...     "MaterialIdentity", MaterialIdentity(name="OFHC-Cu"),
...     source="user input")
>>> registry.get("MaterialIdentity").name
'OFHC-Cu'
>>> registry.get_provenance("MaterialIdentity").source
'user input'
"""

import warnings
from dataclasses import is_dataclass
from typing import Any, Dict, List, Optional, Set, Type

from upxo.material.provenance import Provenance


class MaterialRegistry:
    """Typed, extensible container for material data categories."""

    def __init__(self) -> None:
        self._types: Dict[str, Type] = {}
        self._known_values: Dict[str, Dict[str, Set[Any]]] = {}
        self._instances: Dict[str, Any] = {}
        self._provenance: Dict[str, Provenance] = {}

    def register_category(
        self,
        name: str,
        dataclass_type: Type,
        known_values: Optional[Dict[str, Set[Any]]] = None,
    ) -> None:
        """Declare that ``name`` maps to ``dataclass_type``.

        Parameters
        ----------
        name : str
            Category name (e.g. ``"MaterialIdentity"``). Conventionally
            the dataclass's own class name, but not required to be --
            multiple categories of the same underlying type can be
            registered under different names if ever needed.
        dataclass_type : type
            The dataclass every instance ingested under this category must
            be an instance of.
        known_values : dict, optional
            Maps field name -> a set of recognized values for that field.
            Used only for soft-validation warnings on :meth:`ingest`
            (unknown values warn, never block).

        Notes
        -----
        Calling this again for an already-registered ``name`` replaces its
        type/known_values -- it does not merge with a prior registration.
        """
        if not is_dataclass(dataclass_type):
            raise TypeError(
                f"register_category: {dataclass_type!r} is not a dataclass.")
        self._types[name] = dataclass_type
        self._known_values[name] = dict(known_values or {})

    def ingest(
        self,
        category: str,
        instance: Any,
        *,
        source: Optional[str] = None,
        method: Optional[str] = None,
        params: Optional[dict] = None,
    ) -> None:
        """Store ``instance`` under ``category``, with provenance.

        Parameters
        ----------
        category : str
            Must already be registered via :meth:`register_category`.
        instance : Any
            Must be an instance of that category's registered dataclass
            type.
        source, method, params
            Forwarded into a new :class:`Provenance` record attached to
            this ingestion -- see that class for what each means.

        Raises
        ------
        KeyError
            If ``category`` hasn't been registered yet.
        TypeError
            If ``instance`` isn't an instance of the registered dataclass
            type for ``category``.

        Warns
        -----
        UserWarning
            For every field named in this category's ``known_values``
            (from registration) whose current value on ``instance`` isn't
            in the recognized set. The value is still stored -- this is
            soft validation, not a block.
        """
        if category not in self._types:
            raise KeyError(
                f'ingest: category "{category}" is not registered. '
                f'Call register_category() first. '
                f'Registered categories: {list(self._types.keys())}')
        expected_type = self._types[category]
        if not isinstance(instance, expected_type):
            raise TypeError(
                f'ingest: expected an instance of {expected_type.__name__} '
                f'for category "{category}", got {type(instance).__name__}.')

        for field_name, recognized in self._known_values[category].items():
            value = getattr(instance, field_name, None)
            if value is not None and value not in recognized:
                warnings.warn(
                    f'MaterialRegistry.ingest: "{value}" is not a '
                    f'recognized value for {category}.{field_name} '
                    f'(known: {sorted(recognized)}). Accepted anyway -- '
                    f'soft validation only.',
                    stacklevel=2)

        self._instances[category] = instance
        self._provenance[category] = Provenance(
            source=source, method=method, params=dict(params or {}))

    def get(self, category: str) -> Any:
        """Return the stored instance for ``category``.

        Raises
        ------
        KeyError
            If nothing has been ingested yet for ``category``.
        """
        if category not in self._instances:
            raise KeyError(
                f'get: no instance ingested yet for category "{category}".')
        return self._instances[category]

    def get_provenance(self, category: str) -> Optional[Provenance]:
        """Return the :class:`Provenance` record for ``category``, or
        ``None`` if nothing has been ingested yet for it."""
        return self._provenance.get(category)

    def categories(self) -> List[str]:
        """Return every registered category name, in registration order."""
        return list(self._types.keys())
