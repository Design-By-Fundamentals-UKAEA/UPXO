"""Entry types held by a ReportSession, in append order.

Each entry is a plain dataclass with no GUI/plotting dependency of its own --
ReportSession does the matplotlib/file-copy/JSON work and stores just the
resulting data here. `kind` is a plain class attribute (not an annotated
field) so it is excluded from dataclasses.asdict() -- ReportSession.save()
adds it back explicitly as a discriminator for load().
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class SectionHeader:
    kind = "section"
    id: int
    timestamp: str
    title: str


@dataclass
class ImageEntry:
    kind = "image"
    id: int
    timestamp: str
    title: str
    image_path: str  # relative to the owning ReportSession's reports_dir
    caption: Optional[str] = None
    section: Optional[str] = None
    source_page: Optional[str] = None
    params: Optional[dict] = None


@dataclass
class TableEntry:
    kind = "table"
    id: int
    timestamp: str
    title: str
    columns: list
    rows: list
    caption: Optional[str] = None
    section: Optional[str] = None
    source_page: Optional[str] = None


@dataclass
class TextEntry:
    kind = "text"
    id: int
    timestamp: str
    title: str
    text: str
    section: Optional[str] = None
    source_page: Optional[str] = None


@dataclass
class ParamsEntry:
    kind = "params"
    id: int
    timestamp: str
    title: str
    params: dict
    section: Optional[str] = None
    source_page: Optional[str] = None
