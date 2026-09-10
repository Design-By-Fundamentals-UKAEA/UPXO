"""Framework-agnostic per-run report session.

Holds an ordered, appendable list of report entries (images, tables, text,
parameter snapshots, section headers) plus the on-disk layout for one
pipeline run:

    <reports_dir>/
        report.html          -- written by render_html(), regenerated on demand
        report_data.json      -- this session's own save()/load() sidecar
        assets/                -- copied/rendered images, referenced by report.html

No GUI-toolkit import here -- GUI pages call these methods and are
responsible only for handing over the current matplotlib Figure / a
shared_state dict; see the upxo.reporting package docstring and the
project_reporting_module_plan memory for the wider design (Reporting-1..4).
"""
import json
import re
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .entries import ImageEntry, TableEntry, TextEntry, ParamsEntry, SectionHeader

_ENTRY_CLASSES = {
    "section": SectionHeader,
    "image": ImageEntry,
    "table": TableEntry,
    "text": TextEntry,
    "params": ParamsEntry,
}


def _slugify(text, max_len=40):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower()
    return (slug or "entry")[:max_len]


def _normalize_table(data):
    """Accept a DataFrame-like object, a dict of equal-length lists (column
    name -> values), or a list of dict records; return (columns, rows) as
    plain Python lists (ndarray.tolist() already downcasts numpy scalars)."""
    if hasattr(data, "columns") and hasattr(data, "values"):
        return [str(c) for c in data.columns], [list(r) for r in data.values.tolist()]
    if isinstance(data, dict):
        columns = list(data.keys())
        n = len(next(iter(data.values()))) if data else 0
        rows = [[data[c][i] for c in columns] for i in range(n)]
        return columns, rows
    if isinstance(data, (list, tuple)) and (len(data) == 0 or isinstance(data[0], dict)):
        columns = list(data[0].keys()) if data else []
        rows = [[rec.get(c) for c in columns] for rec in data]
        return columns, rows
    raise TypeError(
        "add_table() accepts a DataFrame-like object, a dict of equal-length "
        f"lists, or a list of dict records -- got {type(data).__name__}"
    )


def _json_default(obj):
    """json.dumps(default=...) hook: downcast numpy scalars (np.int64 etc.
    are not json-serializable even though np.float64 is) and Path objects,
    both of which are realistic to end up in table rows / params snapshots
    given pandas/numpy are used throughout the pipelines this module reports on."""
    if hasattr(obj, "item") and hasattr(obj, "dtype"):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class ReportSession:
    """One pipeline run's appendable report."""

    def __init__(self, reports_dir, run_title="UPXO Report", pipeline_name=None,
                 run_metadata=None):
        # No mkdir here -- matches this codebase's convention for the other
        # per-run output folders (images_dir/raw_dir/...), which are created
        # lazily on first actual write, not just because a GUI session opened.
        self.reports_dir = Path(reports_dir)
        self.assets_dir = self.reports_dir / "assets"

        self.run_title = run_title
        self.pipeline_name = pipeline_name
        self.created_at = datetime.now().isoformat(timespec="seconds")

        # Free-form run-level metadata (e.g. researcher/organisation/
        # material/processing-condition) shown once at the top of the
        # rendered report, distinct from the appendable entries list --
        # set via update_run_metadata() so a GUI page can call it on every
        # save_state() without ever duplicating anything (a plain dict
        # merge, not an appended entry).
        self.run_metadata = dict(run_metadata) if run_metadata else {}

        self.current_section = None
        self._entries = []
        self._next_id = 1

    def update_run_metadata(self, **fields):
        """Merge `fields` into run_metadata (overwriting any existing keys
        of the same name) -- idempotent, safe to call every time a page's
        save_state() runs."""
        self.run_metadata.update(fields)

    # -- entry list access ----------------------------------------------------

    @property
    def entries(self):
        """Read-only view of the entries, in append order."""
        return tuple(self._entries)

    def __len__(self):
        return len(self._entries)

    def remove_entry(self, entry_id):
        """Remove the entry with the given id. Returns True if one was removed."""
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.id != entry_id]
        return len(self._entries) != before

    def clear(self):
        """Remove all entries (does not delete already-written asset files)."""
        self._entries = []
        self.current_section = None

    # -- appending --------------------------------------------------------------

    def _new_id(self):
        i = self._next_id
        self._next_id += 1
        return i

    @staticmethod
    def _timestamp():
        return datetime.now().isoformat(timespec="seconds")

    def _maybe_open_section(self, section):
        if section is not None and section != self.current_section:
            self.add_section(section)

    def add_section(self, title):
        """Insert an explicit section-header entry and make it the active
        section for subsequent add_*() calls that don't pass section=."""
        entry = SectionHeader(id=self._new_id(), timestamp=self._timestamp(), title=title)
        self._entries.append(entry)
        self.current_section = title
        return entry

    def add_image(self, fig_or_path, title, caption=None, section=None,
                  source_page=None, params=None, dpi=150):
        """Append an image entry. `fig_or_path` is either a matplotlib Figure
        (saved into assets/ as a PNG) or a path to an existing image file
        (copied into assets/ as-is). `params` is an optional dict of the
        pipeline configuration active when this image was captured."""
        self._maybe_open_section(section)
        entry_id = self._new_id()
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(fig_or_path, "savefig"):
            filename = f"{entry_id:04d}_{_slugify(title)}.png"
            dest = self.assets_dir / filename
            fig_or_path.savefig(str(dest), dpi=dpi, facecolor=fig_or_path.get_facecolor())
        else:
            src = Path(fig_or_path)
            filename = f"{entry_id:04d}_{_slugify(title)}{src.suffix or '.png'}"
            dest = self.assets_dir / filename
            shutil.copy2(src, dest)

        entry = ImageEntry(
            id=entry_id, timestamp=self._timestamp(), title=title,
            image_path=f"assets/{filename}", caption=caption,
            section=self.current_section, source_page=source_page, params=params,
        )
        self._entries.append(entry)
        return entry

    def add_table(self, data, title, caption=None, section=None, source_page=None):
        """Append a table entry. `data` may be a pandas DataFrame (or any
        object exposing .columns/.values), a dict of equal-length lists
        (column -> values), or a list of dict records."""
        self._maybe_open_section(section)
        columns, rows = _normalize_table(data)
        entry = TableEntry(
            id=self._new_id(), timestamp=self._timestamp(), title=title,
            columns=columns, rows=rows, caption=caption,
            section=self.current_section, source_page=source_page,
        )
        self._entries.append(entry)
        return entry

    def add_text(self, text, title, section=None, source_page=None):
        """Append a free-text entry, rendered verbatim (whitespace preserved)."""
        self._maybe_open_section(section)
        entry = TextEntry(
            id=self._new_id(), timestamp=self._timestamp(), title=title,
            text=text, section=self.current_section, source_page=source_page,
        )
        self._entries.append(entry)
        return entry

    def add_params_snapshot(self, shared_state, keys, title="Configuration",
                             section=None, source_page=None):
        """Append a key/value snapshot of `shared_state` restricted to `keys`,
        so a later result can be traced back to the config that produced it."""
        self._maybe_open_section(section)
        params = {k: shared_state.get(k) for k in keys}
        entry = ParamsEntry(
            id=self._new_id(), timestamp=self._timestamp(), title=title,
            params=params, section=self.current_section, source_page=source_page,
        )
        self._entries.append(entry)
        return entry

    # -- persistence --------------------------------------------------------------

    def _json_path(self):
        return self.reports_dir / "report_data.json"

    def save(self, path=None):
        """Write this session's entries + metadata as JSON so it can be
        reloaded later (e.g. across a GUI restart within the same run
        directory) via load()/open()."""
        path = Path(path) if path is not None else self._json_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_title": self.run_title,
            "pipeline_name": self.pipeline_name,
            "created_at": self.created_at,
            "run_metadata": self.run_metadata,
            "next_id": self._next_id,
            "current_section": self.current_section,
            "entries": [{"kind": e.kind, **asdict(e)} for e in self._entries],
        }
        path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        return path

    @classmethod
    def load(cls, reports_dir_or_json):
        """Reload a session previously written by save(). Accepts either the
        run's reports_dir or a direct path to its report_data.json."""
        p = Path(reports_dir_or_json)
        json_path = p if p.suffix == ".json" else p / "report_data.json"

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        session = cls(json_path.parent, run_title=payload.get("run_title", "UPXO Report"),
                      pipeline_name=payload.get("pipeline_name"),
                      run_metadata=payload.get("run_metadata"))
        session.created_at = payload.get("created_at", session.created_at)
        session.current_section = payload.get("current_section")
        session._next_id = payload.get("next_id", 1)

        entries = []
        for raw in payload.get("entries", []):
            raw = dict(raw)
            kind = raw.pop("kind")
            entry_cls = _ENTRY_CLASSES.get(kind)
            if entry_cls is None:
                continue
            entries.append(entry_cls(**raw))
        session._entries = entries
        return session

    @classmethod
    def open(cls, reports_dir, run_title="UPXO Report", pipeline_name=None):
        """Load an existing report_data.json in `reports_dir` if one exists,
        otherwise start a fresh session there. This is the usual GUI entry
        point -- it makes "Append to Report" survive a GUI restart within
        the same pipeline run without callers needing to check first."""
        reports_dir = Path(reports_dir)
        if (reports_dir / "report_data.json").exists():
            return cls.load(reports_dir)
        return cls(reports_dir, run_title=run_title, pipeline_name=pipeline_name)
