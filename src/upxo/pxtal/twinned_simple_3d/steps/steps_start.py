"""Start / Global Configuration -- Part A of the Twinned FCC walkthrough.

Picks the pipeline-wide output directory and opens the
upxo.reporting.ReportSession every later stage appends its key results into.
Every later steps_*.py function that takes a `report` argument appends to
the SAME session object returned here, so the notebook ends up with one
report.html covering the whole run.
"""
from pathlib import Path

import upxo
from upxo.reporting import ReportSession

# <repo_root>/data, where <repo_root> is wherever this upxo install lives
# (upxo/__init__.py -> upxo/ -> src/ -> repo root).
DEFAULT_OUTPUT_DIR = Path(upxo.__file__).resolve().parents[2] / "data"


def open_pipeline_report(output_dir=DEFAULT_OUTPUT_DIR, researcher_name="",
                          organisation="", email="", material_name="",
                          processing_condition="", notebook_tag=None):
    """Opens (or resumes) this run's report session under
    <output_dir>/TwinnedFCC/Reports -- the same "<output base>/TwinnedFCC/<category>"
    layout used for every other output category (raw exports, mesh files, ...)
    this walkthrough writes.

    researcher_name/organisation/email/material_name/processing_condition
    are optional run-metadata fields -- all blank by default, shown once at
    the top of the generated report.html.

    notebook_tag: when multiple notebooks (e.g. a basic/intermediate/advanced
    tier) share this same steps/ package, pass a short tag (e.g. 'bas0',
    'int0', 'adv0') so each gets its own <Reports>/<tag>/report.html instead
    of all of them accumulating into one shared report.

    Returns
    -------
    upxo.reporting.ReportSession
    """
    reports_dir = Path(output_dir) / "TwinnedFCC" / "Reports"
    run_title = "Twinned FCC 3D Report"
    if notebook_tag:
        reports_dir = reports_dir / notebook_tag
        run_title = f"{run_title} ({notebook_tag})"
    report = ReportSession.open(
        reports_dir, run_title=run_title, pipeline_name="Twinned FCC 3D")
    report.update_run_metadata(
        researcher_name=researcher_name, organisation=organisation, email=email,
        material_name=material_name, processing_condition=processing_condition,
    )
    return report
