"""Render a ReportSession to a single self-contained HTML file.

No template engine (jinja2 is not a UPXO dependency) -- plain string
building plus stdlib html.escape() is all this module's flat entry list
needs. Images are referenced by the relative path already recorded on each
ImageEntry (assets/...), so the output HTML must stay alongside the
assets/ folder -- both live in session.reports_dir by default.
"""
import html
from pathlib import Path

_CSS = """
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
       max-width: 960px; margin: 0 auto; padding: 24px 32px 64px;
       color: #1a1a1a; background: #fff; line-height: 1.5; }
h1 { font-size: 1.6rem; margin-bottom: 4px; }
.meta { color: #666; font-size: 0.85rem; margin-bottom: 32px; }
section h2 { font-size: 1.2rem; margin-top: 40px; padding-bottom: 6px;
             border-bottom: 2px solid #ddd; }
h3 { font-size: 1rem; margin-bottom: 4px; }
figure { margin: 20px 0; }
figure img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
figcaption { color: #444; font-size: 0.9rem; margin-top: 6px; }
.entry-time { color: #999; font-size: 0.75rem; }
table { border-collapse: collapse; margin: 10px 0 20px; font-size: 0.9rem; }
th, td { border: 1px solid #ddd; padding: 4px 10px; text-align: left; }
th { background: #f5f5f5; }
pre { background: #f5f5f5; border: 1px solid #ddd; border-radius: 4px;
      padding: 10px 14px; white-space: pre-wrap; word-wrap: break-word; }
.params-table td:first-child { font-weight: 600; color: #333; }
.run-metadata { margin: 0 0 28px; }
.run-metadata td:first-child { font-weight: 600; color: #333; white-space: nowrap; }
"""

# Display order + labels for ReportSession.run_metadata's well-known keys --
# any OTHER key present is still rendered (title-cased from its name), this
# just controls the order/label for the ones the Global Configuration page
# populates by default.
_RUN_METADATA_LABELS = [
    ("researcher_name", "Researcher"),
    ("organisation", "Organisation"),
    ("email", "E-Mail Address"),
    ("material_name", "Material"),
    ("processing_condition", "Processing Condition"),
]


def _table_html(columns, rows, css_class=""):
    thead = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape('' if v is None else str(v))}</td>" for v in row)
        body_rows.append(f"<tr>{cells}</tr>")
    cls = f' class="{css_class}"' if css_class else ""
    return (f"<table{cls}><thead><tr>{thead}</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody></table>")


def _render_entry(entry):
    kind = entry.kind
    meta_bits = [html.escape(entry.timestamp)]
    if getattr(entry, "source_page", None):
        meta_bits.append(html.escape(entry.source_page))
    time_html = f'<div class="entry-time">{" &middot; ".join(meta_bits)}</div>'

    if kind == "image":
        caption = f": {html.escape(entry.caption)}" if entry.caption else ""
        params_html = ""
        if entry.params:
            rows = [[k, v] for k, v in entry.params.items()]
            params_html = _table_html(["Parameter", "Value"], rows, css_class="params-table")
        return (
            f'<figure>{time_html}'
            f'<img src="{html.escape(entry.image_path)}" alt="{html.escape(entry.title)}">'
            f'<figcaption><strong>{html.escape(entry.title)}</strong>{caption}</figcaption>'
            f'{params_html}</figure>'
        )

    if kind == "table":
        caption = f"<p>{html.escape(entry.caption)}</p>" if entry.caption else ""
        return (
            f'<div class="table-entry">{time_html}'
            f'<h3>{html.escape(entry.title)}</h3>{caption}'
            f'{_table_html(entry.columns, entry.rows)}</div>'
        )

    if kind == "text":
        return (
            f'<div class="text-entry">{time_html}'
            f'<h3>{html.escape(entry.title)}</h3>'
            f'<pre>{html.escape(entry.text)}</pre></div>'
        )

    if kind == "params":
        rows = [[k, v] for k, v in entry.params.items()]
        return (
            f'<div class="params-entry">{time_html}'
            f'<h3>{html.escape(entry.title)}</h3>'
            f'{_table_html(["Parameter", "Value"], rows, css_class="params-table")}</div>'
        )

    raise ValueError(f"Unknown entry kind: {kind!r}")


def render_html(session, out_path=None):
    """Render `session` to a single HTML file and return its Path.

    Defaults to session.reports_dir / "report.html" -- the location every
    ImageEntry's relative image_path assumes.
    """
    out_path = Path(out_path) if out_path is not None else session.reports_dir / "report.html"

    body_parts = []
    section_open = False
    for entry in session.entries:
        if entry.kind == "section":
            if section_open:
                body_parts.append("</section>")
            body_parts.append(f"<section><h2>{html.escape(entry.title)}</h2>")
            section_open = True
            continue
        body_parts.append(_render_entry(entry))
    if section_open:
        body_parts.append("</section>")

    subtitle_bits = [session.pipeline_name] if session.pipeline_name else []
    subtitle_bits.append(f"generated {session.created_at}")
    subtitle = " &middot; ".join(html.escape(str(b)) for b in subtitle_bits)

    # Run metadata (researcher/organisation/material/...) shown once at
    # the top, before any entries -- blank/absent fields are skipped
    # rather than shown as empty rows.
    metadata_html = ""
    if session.run_metadata:
        known_keys = {k for k, _label in _RUN_METADATA_LABELS}
        rows = [[label, session.run_metadata[key]] for key, label in _RUN_METADATA_LABELS
                if str(session.run_metadata.get(key) or "").strip()]
        rows += [[k.replace("_", " ").title(), v] for k, v in session.run_metadata.items()
                 if k not in known_keys and str(v or "").strip()]
        if rows:
            metadata_html = _table_html(["Field", "Value"], rows, css_class="run-metadata params-table")

    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(session.run_title)}</title>"
        f"<style>{_CSS}</style></head><body>"
        f"<h1>{html.escape(session.run_title)}</h1>"
        f"<div class='meta'>{subtitle}</div>"
        f"{metadata_html}"
        f"{''.join(body_parts)}"
        "</body></html>"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path
