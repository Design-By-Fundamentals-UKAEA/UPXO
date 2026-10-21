"""Centralized reporting module for UPXO pipelines (FM Steel, Twinned GS
FCC, ...).

Framework-agnostic: builds an ordered, appendable log of images/tables/text/
parameter snapshots for one pipeline run and renders it to a single
self-contained HTML file. Pipelines call into ReportSession.
"""
from .session import ReportSession, normalize_table_data
from .entries import ImageEntry, TableEntry, TextEntry, ParamsEntry, SectionHeader
from .render_html import render_html

__all__ = [
    "ReportSession",
    "ImageEntry", "TableEntry", "TextEntry", "ParamsEntry", "SectionHeader",
    "render_html", "normalize_table_data",
]
