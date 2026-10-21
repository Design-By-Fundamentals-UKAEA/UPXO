"""Optional progress callbacks for long-running stage handlers."""
from __future__ import annotations

from typing import Any, Callable, Optional

ProgressFn = Callable[..., None]

# Keys pulled into the progress message when algorithms emit a dict entry.
_ENTRY_DETAIL_KEYS = (
    "accepted_moves",
    "below_target",
    "minimum_dihedral",
    "swaps_2_3",
    "swaps_3_2",
    "inserted",
    "outside_limits",
)


def emit_progress(
    progress: Optional[ProgressFn],
    message: str,
    current: int | None = None,
    total: int | None = None,
) -> None:
    """Invoke ``progress(message, current, total)`` if provided.

    Swallows errors raised by the callback; re-raises ``PipelineCancelled``.
    """
    if progress is None:
        return
    from upxo.meshing.gbconformant.pipeline.cancel import PipelineCancelled
    try:
        progress(message, current, total)
    except PipelineCancelled:
        raise
    except TypeError:
        try:
            progress(message)
        except PipelineCancelled:
            raise
        except Exception:
            pass
    except Exception:
        pass


def phase_progress(
    progress: Optional[ProgressFn],
    phase: int,
    n_phases: int,
    message: str,
) -> None:
    """Report a coarse multi-phase percent (1-based phase index)."""
    if n_phases <= 0:
        emit_progress(progress, message)
        return
    phase = max(0, min(phase, n_phases))
    emit_progress(progress, message, current=phase, total=n_phases)


def adapt_dict_progress(
    state,
    *,
    label: str,
    total_key: str = "pass_number",
    total: int | None = None,
) -> Callable[[Any], None]:
    """Map algorithm ``progress(entry: dict)`` hooks to ``state.report_progress``.

    Tet smooth / swaps / insert call ``progress(entry)`` once per pass with
    keys such as ``pass_number``, ``below_target``, ``minimum_dihedral``.
    """

    def _cb(entry: Any) -> None:
        state.check_cancel()
        if not isinstance(entry, dict):
            state.report_progress(f"{label}: {entry}")
            return
        current = entry.get(total_key)
        try:
            current_i = int(current) if current is not None else None
        except (TypeError, ValueError):
            current_i = None
        parts = [label]
        if current_i is not None:
            if total is not None and total > 0:
                parts.append(f"pass {current_i}/{int(total)}")
            else:
                parts.append(f"pass {current_i}")
        for key in _ENTRY_DETAIL_KEYS:
            if key in entry and entry[key] is not None:
                val = entry[key]
                if isinstance(val, float):
                    parts.append(f"{key}={val:.3g}")
                else:
                    parts.append(f"{key}={val}")
        message = " · ".join(parts)
        tot = int(total) if total is not None and total > 0 else None
        state.report_progress(message, current=current_i, total=tot)

    return _cb
