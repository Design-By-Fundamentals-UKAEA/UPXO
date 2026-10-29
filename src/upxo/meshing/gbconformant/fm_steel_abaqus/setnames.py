"""Rules for the names of element sets, node sets and surfaces in a deck.

Abaqus reads names without regard to case, so ``es_pag_1`` and ``ES_PAG_1`` are the same set. A name here
starts with a letter, holds only letters, digits, ``_`` and ``-``, and is at most 80 characters long. Element
sets and surfaces live in the Part, node sets in the Assembly; within each of these a name may be used once.
"""
from __future__ import annotations

import re

MAX_LENGTH = 80
_NAME = re.compile(r'[A-Za-z][A-Za-z0-9_-]*\Z')
_PREFIX = re.compile(r'[A-Za-z][A-Za-z0-9_-]*\Z')


class SetNameError(ValueError):
    """One or more set names break the naming rules or are used twice."""

    def __init__(self, problems):
        self.problems = list(problems)
        shown = '; '.join(self.problems[:8]) + (f'; and {len(self.problems) - 8} more' if len(self.problems) > 8 else '')
        super().__init__(f'Set names cannot be used: {shown}.')


def name_problem(name, what='name') -> str | None:
    """Why ``name`` cannot name a set, or ``None`` when it can."""
    text = '' if name is None else str(name)
    if not text:
        return f'the {what} is empty'
    if len(text) > MAX_LENGTH:
        return f'{text[:20]}... is {len(text)} characters long (at most {MAX_LENGTH})'
    if not _NAME.match(text):
        return (f'{text!r} must start with a letter and hold only letters, digits, underscores and hyphens')
    return None


def prefix_problem(prefix, longest_key=12, what='prefix') -> str | None:
    """Why ``prefix`` cannot start set names, or ``None``; ``longest_key`` is the longest text appended to it."""
    text = '' if prefix is None else str(prefix)
    if not text:
        return f'the {what} is empty'
    if not _PREFIX.match(text):
        return f'the {what} {text!r} must start with a letter and hold only letters, digits, underscores and hyphens'
    if len(text) + int(longest_key) > MAX_LENGTH:
        return f'the {what} {text!r} leaves no room for the rest of the name ({MAX_LENGTH} characters at most)'
    return None


class SetRegistry:
    """Collects every name a deck defines and the problems found; :meth:`raise_problems` stops the export."""

    SCOPES = {'Elset': 'Part', 'Surface': 'Part', 'Nset': 'Assembly'}

    def __init__(self):
        self._seen = {}            # (scope, keyword, lower name) -> where it came from
        self.problems = []

    def add(self, keyword, name, origin):
        problem = name_problem(name, f'name of the {origin}')
        if problem:
            self.problems.append(f'{origin}: {problem}')
            return
        key = (self.SCOPES[keyword], keyword, str(name).lower())
        if key in self._seen:
            first = self._seen[key]
            same_case = first[1] == name
            self.problems.append(f'{keyword.lower()} {name!r} ({origin}) is already used by the {first[0]}'
                                 + ('' if same_case else f' as {first[1]!r} (Abaqus ignores case)'))
            return
        self._seen[key] = (origin, str(name))

    def names(self, keyword):
        return [name for (_scope, kw, _low), (_origin, name) in self._seen.items() if kw == keyword]

    def raise_problems(self):
        if self.problems:
            raise SetNameError(self.problems)


def settings_problems(settings) -> list[str]:
    """Problems with the names and prefixes typed into ``settings`` (:class:`.export.DeckSettings`), and with
    those a set provider checks.

    Checks every typed name and prefix against the rules, and the typed full names against each other and
    against the fixed names of the deck. Names built from a prefix and a key are checked when the deck is
    written (they depend on the structure).
    """
    from . import constants as C
    from .extra_sets import provider_problems

    problems = []
    for level, prefix in settings.elset_prefixes.items():
        p = prefix_problem(prefix, what=f'{level} element-set prefix')
        if p:
            problems.append(p)
    registry = SetRegistry()
    for face, name in settings.nset_names.items():
        registry.add('Nset', name, f'{face} face node set')
    registry.add('Nset', C.GRAIN_BOUNDARY_NSET, 'grain-boundary node set')
    registry.add('Nset', C.JUNCTION_NSET, 'junction node set')
    if settings.layer_sets:
        registry.add('Elset', settings.prefix_gb_layer, 'grain-boundary layer set')
        registry.add('Elset', settings.prefix_interior, 'interior set')
    if settings.output_elset:
        p = name_problem(settings.output_elset, 'output element set')
        if p:
            problems.append(p)
    return problems + registry.problems + provider_problems(settings)
