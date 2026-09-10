"""Centralized reporting module for UPXO GUI pipelines (FM Steel, Twinned GS
FCC, ...).

Framework-agnostic: builds an ordered, appendable log of images/tables/text/
parameter snapshots for one pipeline run and renders it to a single
self-contained HTML file. Has no tkinter (or any GUI-toolkit) dependency --
GUI wiring ("Append to Report" buttons etc.) lives in each pipeline's own
gui/ package and calls into ReportSession.

See the project_reporting_module_plan memory for the phased design
(Reporting-1..4) this module is part of.
"""
from .session import ReportSession
from .entries import ImageEntry, TableEntry, TextEntry, ParamsEntry, SectionHeader
from .render_html import render_html

__all__ = [
    "ReportSession",
    "ImageEntry", "TableEntry", "TextEntry", "ParamsEntry", "SectionHeader",
    "render_html",
]
