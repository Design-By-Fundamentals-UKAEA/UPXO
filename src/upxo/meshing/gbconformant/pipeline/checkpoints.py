"""Stage results saved in a run folder, and restored into a fresh pipeline state.

Each finished stage writes ``<run>/checkpoints/<stage_id>/`` holding the
results it produced (portable files, see :mod:`.serialize`), its report and
status, and a signature of the settings and inputs up to and including that
stage. A restore trusts a stage only when its files read back, their format
version is known and the signature still matches the current settings, so a
changed setting upstream makes everything after it stale instead of silently
reusing results built on other settings.

``state`` is any object with ``artefacts``, ``artefact_writer``, ``reports``,
``config`` and ``stage_ids``; statuses are set through a callback.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from .serialize import FORMAT, VERSION, SerializeError, read_value, safe_name, write_value

FOLDER = 'checkpoints'
META = 'meta.json'


def _retry(action, attempts=40, pause=0.1):
    """Run a file operation, retrying while Windows holds a handle (scanners, indexers)."""
    for attempt in range(attempts):
        try:
            return action()
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(pause)


def _remove_tree(path):
    if Path(path).exists():
        _retry(lambda: shutil.rmtree(path))


def _move_tree(source, destination):
    _retry(lambda: os.replace(source, destination))


def _json_safe(value):
    """Plain JSON types: tuples and sets become lists, numpy numbers become Python numbers."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if hasattr(value, 'item') and callable(value.item):
        return value.item()
    return value


def _normal(value):
    """Same hash for ``1`` and ``1.0``, for a tuple and a list, and for an array and a list."""
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _normal(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normal(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_normal(v) for v in value), key=repr)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def canonical_signature(parts):
    """Short hash of ``parts`` (numbers, text, lists and dictionaries in any order)."""
    text = json.dumps(_normal(parts), sort_keys=True, default=_plain, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf8')).hexdigest()[:20]


def _plain(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if hasattr(value, 'tolist'):
        return value.tolist()
    return str(value)


def chain_signatures(stage_ids, own):
    """Signature per stage that also depends on every earlier stage's own part."""
    out, previous = {}, ''
    for stage_id in stage_ids:
        previous = canonical_signature([previous, own(stage_id)])
        out[stage_id] = previous
    return out


class CheckpointStore:
    """Stage results in ``<run_folder>/checkpoints``.

    With ``keep_history`` (the default) every stage keeps the results it wrote, so a stage can
    be run again from the results of the stage before it. Without it a newer stage's result
    replaces the older copy of the same artefact, which takes less disk; a restore then
    returns only a prefix of stages whose results are all still complete.
    """

    def __init__(self, run_folder, keep_history=True):
        self.root = Path(run_folder) / FOLDER
        self.keep_history = keep_history

    # --- reading -------------------------------------------------------------------
    def stage_dir(self, stage_id):
        return self.root / stage_id

    def meta(self, stage_id):
        path = self.stage_dir(stage_id) / META
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding='utf8'))
        except (OSError, ValueError):
            return None

    def stages(self):
        return sorted(p.name for p in self.root.glob('*')
                      if p.is_dir() and not p.name.startswith('.') and (p / META).is_file()) \
            if self.root.is_dir() else []

    # --- writing ---------------------------------------------------------------------
    def save_stage(self, state, stage_id, signature, status='OK', report=None, config_changes=None):
        """Write the results ``stage_id`` produced; return its meta dictionary.

        ``config_changes`` are the settings the stage itself changed while running (the
        hierarchy level it found, the spacing after an upscale); a restore applies them again.
        """
        keys = [k for k, w in state.artefact_writer.items()
                if w == stage_id and k in state.artefacts]
        tmp = self.root / f'.{stage_id}.writing'
        _remove_tree(tmp)
        tmp.mkdir(parents=True)
        written, skipped, size = {}, {}, 0
        for key in sorted(keys):
            try:
                files = write_value(tmp / safe_name(key), state.artefacts[key], where=key)
            except SerializeError as error:
                skipped[key] = str(error)
                continue
            written[key] = files
        report_files = None
        try:
            report_files = write_value(tmp / '_report',
                                      (report if report is not None else state.reports.get(stage_id)) or {},
                                      where='report')
        except SerializeError as error:
            skipped['_report'] = str(error)
        for path in tmp.iterdir():
            size += path.stat().st_size
        meta = {
            'format': FORMAT, 'version': VERSION, 'stage_id': stage_id, 'status': status,
            'signature': signature, 'artefacts': written, 'report': report_files,
            'skipped': skipped, 'bytes': size, 'config': _json_safe(config_changes or {}),
            'written_utc': datetime.now(timezone.utc).isoformat(),
        }
        (tmp / META).write_text(json.dumps(meta, indent=1), encoding='utf8')
        dest = self.stage_dir(stage_id)
        _remove_tree(dest)
        _move_tree(tmp, dest)
        if not self.keep_history:
            self._drop_superseded(stage_id, written)
        return meta

    def save_marker(self, stage_id, signature, status, report=None, config_changes=None):
        """A stage with no results of its own (for instance a skipped one)."""
        empty = SimpleNamespace(artefacts={}, artefact_writer={}, reports={})
        return self.save_stage(empty, stage_id, signature, status, report=report or {},
                               config_changes=config_changes)

    def _drop_superseded(self, stage_id, keys):
        """A newer stage's result replaces the older copy of the same artefact."""
        for other in self.stages():
            if other == stage_id:
                continue
            meta = self.meta(other)
            stale = [k for k in (meta or {}).get('artefacts', {}) if k in keys]
            if stale:
                self._remove_copies(other, meta, stale, stage_id)

    def _remove_copies(self, stage_id, meta, keys, replaced_by):
        """Delete the files of ``keys`` from ``stage_id``'s copy and note which stage replaced them; returns bytes freed."""
        freed = 0
        for key in keys:
            for name in meta['artefacts'].pop(key).values():
                if name:
                    path = self.stage_dir(stage_id) / name
                    if path.is_file():
                        freed += path.stat().st_size
                        path.unlink()
            meta.setdefault('superseded', {})[key] = replaced_by
        meta['bytes'] = sum(p.stat().st_size for p in self.stage_dir(stage_id).iterdir() if p.name != META)
        (self.stage_dir(stage_id) / META).write_text(json.dumps(meta, indent=1), encoding='utf8')
        return freed

    def _older_copies(self, order):
        """``[(stage, meta, keys, replaced_by)]``: the artefacts a later stage of ``order`` wrote again."""
        present = set(self.stages())
        stages = [s for s in order if s in present]
        last = {}
        for stage in stages:
            for key in (self.meta(stage) or {}).get('artefacts', {}):
                last[key] = stage
        out = []
        for stage in stages:
            meta = self.meta(stage) or {}
            stale = [k for k in meta.get('artefacts', {}) if last[k] != stage]
            if stale:
                out.append((stage, meta, stale, last))
        return out

    def history_bytes(self, order):
        """Disk the older copies of artefacts take: every artefact kept by more than one stage, less its newest copy."""
        total = 0
        for stage, meta, keys, _last in self._older_copies(order):
            for key in keys:
                for name in meta['artefacts'][key].values():
                    path = self.stage_dir(stage) / name if name else None
                    if path is not None and path.is_file():
                        total += path.stat().st_size
        return total

    def drop_history(self, order):
        """Keep each artefact only in the last stage of ``order`` that wrote it; returns the bytes freed.

        A run that finished still restores as a whole afterwards (its results are the newest copies); what is
        lost is running a stage again from the results of the stage before it.
        """
        freed = 0
        for stage, meta, keys, last in self._older_copies(order):
            for key in keys:
                freed += self._remove_copies(stage, meta, [key], last[key])
        return freed

    def discard(self, stage_ids):
        for stage_id in stage_ids:
            _remove_tree(self.stage_dir(stage_id))

    # --- restoring ----------------------------------------------------------------------
    def restore(self, state, stage_ids, signatures, set_status, set_config=None):
        """Load every stage that is still valid, in order; later stages follow earlier ones.

        ``set_status(stage_id, text)`` receives ``'OK'``, ``'Skipped'`` or ``'Stale'``;
        ``set_config(stage_id, changes)`` receives the settings a restored stage changed.
        Returns ``{'restored': [...], 'stale': [...], 'bytes': n, 'problems': [...]}``.
        """
        stale, problems = [], []
        usable = []                         # (stage_id, meta, artefacts, report), read but not yet in the state
        broken = False
        for stage_id in stage_ids:
            meta = self.meta(stage_id)
            if meta is None:
                broken = True
                continue
            reason = None
            if broken:
                reason = 'an earlier stage is not valid'
            elif meta.get('format') != FORMAT or meta.get('version') != VERSION:
                reason = f"unknown file version ({meta.get('format')} v{meta.get('version')})"
            elif meta.get('signature') != signatures.get(stage_id):
                reason = 'settings or inputs changed since it ran'
            if reason is None:
                try:
                    artefacts, report = self._read_stage(stage_id, meta)
                except (SerializeError, OSError, ValueError, KeyError, ImportError) as error:
                    reason = f'results could not be read ({error})'
            if reason is not None:
                broken = True
                stale.append(stage_id)
                problems.append(f'{stage_id}: {reason}')
                set_status(stage_id, 'Stale')
                continue
            usable.append((stage_id, meta, artefacts, report))
        keep = self._complete_prefix(usable)
        restored, size = [], 0
        for stage_id, meta, artefacts, report in usable[:keep]:
            for key, value in artefacts.items():
                state.artefacts[key] = value
                state.artefact_writer[key] = stage_id
            if report:
                state.reports[stage_id] = report
            if set_config is not None and meta.get('config'):
                set_config(stage_id, meta['config'])
            set_status(stage_id, meta['status'])
            restored.append(stage_id)
            size += int(meta.get('bytes', 0))
        for stage_id, _, _, _ in usable[keep:]:
            stale.append(stage_id)
            problems.append(f'{stage_id}: results it builds on were replaced by a later stage '
                            'and not kept')
            set_status(stage_id, 'Stale')
        stale.sort(key=list(stage_ids).index)
        return {'restored': restored, 'stale': stale, 'bytes': size, 'problems': problems}

    @staticmethod
    def _complete_prefix(usable):
        """How many leading stages of ``usable`` form a complete state.

        A stage's result that a later stage replaced (and that was not kept) is missing; the
        prefix must include the stage that replaced it, or the state would lack that result.
        """
        for count in range(len(usable), 0, -1):
            inside = {stage_id for stage_id, _, _, _ in usable[:count]}
            if all(replacer in inside
                   for _, meta, _, _ in usable[:count]
                   for replacer in (meta.get('superseded') or {}).values()):
                return count
        return 0

    def load_keys(self, stage_id, keys):
        """Read only the artefacts ``keys`` that ``stage_id`` saved (those it did not save are left out)."""
        meta = self.meta(stage_id)
        if meta is None:
            return {}
        folder = self.stage_dir(stage_id)
        return {key: read_value(folder / Path(files['json']).stem)
                for key, files in meta['artefacts'].items() if key in set(keys)}

    def _read_stage(self, stage_id, meta):
        folder = self.stage_dir(stage_id)
        artefacts = {key: read_value(folder / Path(files['json']).stem)
                     for key, files in meta['artefacts'].items()}
        report = read_value(folder / '_report') if meta.get('report') else {}
        return artefacts, report
