"""Cooperative cancel for long-running pipeline stages."""
from __future__ import annotations


class PipelineCancelled(Exception):
    """Raised when a stage is cancelled during a progress update.

    Cancel is cooperative: it takes effect at the next ``report_progress`` /
    algorithm pass callback. Gmsh kernel work can only stop between its phases.
    """

    def __init__(self, message: str = "Cancelled by user"):
        super().__init__(message)
