"""Which raw feature each meshed grain stands for.

Cleaning only reassigns voxels between labels that already exist, so every
meshed grain traces back to one or more raw labels. When whole grains merge,
the merged grain keeps the identity (and so the orientation and element-set
name) of the largest member by raw voxel count; equal sizes are decided by a
seeded random pick that does not depend on dictionary order.
"""
from dataclasses import dataclass, field
import numpy as np

RULES = ('larger', 'survivor')


@dataclass
class IdentityMap:
    """Meshed label -> winning raw label, with the merge history."""
    winner: dict
    members: dict
    survivor_raw: dict
    rule: str
    seed: int
    report: dict = field(default_factory=dict)

    def lookup(self):
        """Array ``lut`` with ``lut[meshed_label]`` = raw label (-1 if unused)."""
        lut = np.full(max(self.winner) + 1, -1, dtype=np.int64)
        for meshed, raw in self.winner.items():
            lut[meshed] = raw
        return lut


def _invert_unique(mapping, name):
    inverse = {}
    for key, value in mapping.items():
        if value in inverse:
            raise ValueError(f'{name} is not one-to-one at {value}.')
        inverse[value] = key
    return inverse


def resolve_identities(raw_sizes, *, shuffle_map=None, survivor_map=None,
                       rule='larger', seed=0):
    """Resolve the raw identity of every meshed grain.

    Parameters
    ----------
    raw_sizes : mapping raw label -> voxel count before any cleaning.
    shuffle_map : mapping raw label -> pipeline label (grain-ID shuffle);
        ``None`` means the pipeline used the raw labels unchanged.
    survivor_map : mapping pipeline label -> pipeline label of the grain it
        was merged into (the small-grain removal's ``original_to_survivor``);
        ``None`` means no grain was merged.
    rule : ``'larger'`` keeps the largest member's identity; ``'survivor'``
        keeps the identity of the grain that absorbed the others.
    seed : seeds the pick between equally large members.
    """
    if rule not in RULES:
        raise ValueError(f'rule must be one of {RULES}')
    raw_sizes = {int(k): int(v) for k, v in raw_sizes.items()}
    shuffle = ({int(k): int(v) for k, v in shuffle_map.items()}
               if shuffle_map is not None else {r: r for r in raw_sizes})
    if set(shuffle) != set(raw_sizes):
        raise ValueError('shuffle_map must cover exactly the raw labels.')
    pipeline_to_raw = _invert_unique(shuffle, 'shuffle_map')
    survivor = ({int(k): int(v) for k, v in survivor_map.items()}
                if survivor_map is not None else {p: p for p in pipeline_to_raw})
    if set(survivor) != set(pipeline_to_raw):
        raise ValueError('survivor_map must cover exactly the pipeline labels.')
    for target in set(survivor.values()):
        if survivor[target] != target:
            raise ValueError(f'Survivor {target} was itself merged away.')

    members = {}
    for raw, pipeline in shuffle.items():
        members.setdefault(survivor[pipeline], []).append(raw)
    winner, survivor_raw, entries = {}, {}, []
    for meshed in sorted(members):
        group = sorted(members[meshed])
        own = pipeline_to_raw[meshed]
        survivor_raw[meshed] = own
        if rule == 'survivor':
            choice = own
        else:
            top = max(raw_sizes[r] for r in group)
            tied = [r for r in group if raw_sizes[r] == top]
            choice = tied[0] if len(tied) == 1 else int(
                np.random.default_rng([int(seed), int(meshed)]).choice(tied))
        winner[meshed] = choice
        if len(group) > 1:
            entries.append({
                'meshed_label': meshed, 'winner': choice, 'survivor': own,
                'differs_from_survivor': choice != own,
                'members': {int(r): raw_sizes[r] for r in group}})
    absorbed = sorted(r for g in members.values() for r in g
                      if r != winner[survivor[shuffle[r]]])
    report = {
        'rule': rule, 'seed': int(seed),
        'raw_labels': len(raw_sizes), 'meshed_labels': len(winner),
        'lost_raw_labels': absorbed,
        'merged_groups': entries,
        'winner_differs_from_survivor': sum(e['differs_from_survivor'] for e in entries)}
    return IdentityMap(winner=winner, members={m: tuple(sorted(g)) for m, g in members.items()},
                       survivor_raw=survivor_raw, rule=rule, seed=int(seed), report=report)
