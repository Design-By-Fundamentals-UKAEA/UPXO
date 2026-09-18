"""Start -- Part A of the FM Steel 3D walkthrough.

Establishes the pipeline output directory and opens the report session
(`upxo.reporting.ReportSession`) to which subsequent stages append results.
Mirrors `upxo.demos.Twinned3D.steps.steps_start` exactly, with the pipeline
name/output folder swapped from "TwinnedFCC" to "FMSteel3D".
"""
from pathlib import Path

import upxo
from upxo.reporting import ReportSession

DEFAULT_OUTPUT_DIR = Path(upxo.__file__).resolve().parents[2] / "data"


def open_pipeline_report(output_dir=DEFAULT_OUTPUT_DIR, researcher_name="",
                          organisation="", email="", material_name="",
                          processing_condition="", notebook_tag=None):
    """Open (or resume) a report session for this notebook run.

    Writes to `<output_dir>/FMSteel3D/Reports[/<notebook_tag>]/`. Passing the
    same `notebook_tag` on a later run resumes that same report (loads the
    existing `report_data.json` if present) rather than starting fresh.

    Returns
    -------
    upxo.reporting.ReportSession
    """
    reports_dir = Path(output_dir) / "FMSteel3D" / "Reports"
    run_title = "FM Steel 3D Report"
    if notebook_tag:
        reports_dir = reports_dir / notebook_tag
        run_title = f"{run_title} ({notebook_tag})"
    report = ReportSession.open(
        reports_dir, run_title=run_title, pipeline_name="FM Steel 3D")
    report.update_run_metadata(
        researcher_name=researcher_name, organisation=organisation, email=email,
        material_name=material_name, processing_condition=processing_condition,
    )
    return report


def finish_report(report):
    """Render `report`'s accumulated entries to a single `report.html` file.

    Returns
    -------
    pathlib.Path : path to the written report.html
    """
    from upxo.reporting.render_html import render_html
    return render_html(report)
