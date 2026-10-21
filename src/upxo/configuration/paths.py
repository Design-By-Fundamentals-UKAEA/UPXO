"""Configuration protocol for user-selected UPXO data and export locations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any
import uuid


class ConfigurationError(ValueError):
    """Raised when a UPXO configuration is malformed or unsupported."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _path(value: str | os.PathLike[str], *, base: Path | None = None) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(os.fspath(value)))
    result = Path(expanded)
    if not result.is_absolute() and base is not None:
        result = base / result
    return result.resolve(strict=False)


@dataclass(frozen=True)
class PipelineDefinition:
    directory_name: str
    enabled: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {"directory_name": self.directory_name, "enabled": self.enabled}


@dataclass(frozen=True)
class RunPaths:
    """All directories belonging to one reproducible pipeline run."""

    run_root: Path
    config: Path
    manifest: Path
    exports: Path
    abaqus: Path
    raw: Path
    images: Path
    distributions: Path
    reports: Path
    scripts: Path
    logs: Path
    temporary: Path

    def create(self) -> "RunPaths":
        for directory in (
            self.run_root, self.exports, self.abaqus, self.raw, self.images,
            self.distributions, self.reports, self.scripts, self.logs, self.temporary,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def as_dict(self) -> dict[str, str]:
        return {name: value.as_posix() for name, value in self.__dict__.items()}


@dataclass
class UPXOPaths:
    output_root: Path
    temporary_root: Path | None = None
    default_pipeline: str = "fm_steel"
    pipelines: dict[str, PipelineDefinition] = field(default_factory=lambda: {
        "twinned_fcc": PipelineDefinition("TwinnedFCC"),
        "fm_steel": PipelineDefinition("FmSteels"),
    })
    options: dict[str, Any] = field(default_factory=lambda: {
        "create_missing_directories": True,
        "use_run_directories": True,
    })
    schema_version: int = 1
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    @classmethod
    def defaults(cls) -> "UPXOPaths":
        return cls(output_root=Path.home() / "UPXO")

    def validate(self) -> "UPXOPaths":
        self.output_root = _path(self.output_root)
        if self.temporary_root is not None:
            self.temporary_root = _path(self.temporary_root)
        if self.schema_version != 1:
            raise ConfigurationError(f"Unsupported UPXO paths schema version: {self.schema_version}")
        if self.default_pipeline not in self.pipelines:
            raise ConfigurationError(f"Unknown default pipeline: {self.default_pipeline}")
        for name, definition in self.pipelines.items():
            if not name or not definition.directory_name:
                raise ConfigurationError("Pipeline names and directory names cannot be empty")
        return self

    def pipeline_root(self, pipeline: str | None = None) -> Path:
        self.validate()
        name = pipeline or self.default_pipeline
        try:
            definition = self.pipelines[name]
        except KeyError as exc:
            raise ConfigurationError(f"Unknown pipeline: {name}") from exc
        if not definition.enabled:
            raise ConfigurationError(f"Pipeline is disabled: {name}")
        return self.output_root / definition.directory_name

    def run_paths(self, pipeline: str | None = None, run_id: str | None = None, *, create: bool = False) -> RunPaths:
        name = pipeline or self.default_pipeline
        root = self.pipeline_root(name) / "runs" / (run_id or self.new_run_id(name))
        result = RunPaths(
            run_root=root, config=root / "config.json", manifest=root / "manifest.json",
            exports=root / "exports", abaqus=root / "exports" / "abaqus",
            raw=root / "exports" / "raw", images=root / "exports" / "images",
            distributions=root / "exports" / "distributions", reports=root / "reports",
            scripts=root / "scripts", logs=root / "logs", temporary=root / "temporary",
        )
        return result.create() if create else result

    def new_run_id(self, pipeline: str | None = None) -> str:
        name = pipeline or self.default_pipeline
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{stamp}_{name}_{uuid.uuid4().hex[:8]}"

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": "upxo.paths", "schema_version": self.schema_version,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "output_root": self.output_root.as_posix(),
            "temporary_root": self.temporary_root.as_posix() if self.temporary_root else None,
            "default_pipeline": self.default_pipeline,
            "pipelines": {key: value.as_dict() for key, value in self.pipelines.items()},
            "options": self.options,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base: Path | None = None) -> "UPXOPaths":
        if data.get("schema") != "upxo.paths":
            raise ConfigurationError("Not an UPXO paths configuration")
        if data.get("schema_version") != 1:
            raise ConfigurationError(f"Unsupported UPXO paths schema version: {data.get('schema_version')}")
        pipelines = {
            name: PipelineDefinition(str(value["directory_name"]), bool(value.get("enabled", True)))
            for name, value in data.get("pipelines", {}).items()
        }
        config = cls(
            output_root=_path(data["output_root"], base=base),
            temporary_root=_path(data["temporary_root"], base=base) if data.get("temporary_root") else None,
            default_pipeline=str(data.get("default_pipeline", "fm_steel")), pipelines=pipelines,
            options=dict(data.get("options", {})), schema_version=1,
            created_at=str(data.get("created_at", _now())), updated_at=str(data.get("updated_at", _now())),
        )
        return config.validate()


class ConfigStore:
    """Read and atomically write the global UPXO paths JSON file."""

    @staticmethod
    def default_path() -> Path:
        system = platform.system()
        if system == "Windows":
            root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
            return root / "UPXO" / "paths.json"
        if system == "Darwin":
            return Path.home() / "Library" / "Application Support" / "UPXO" / "paths.json"
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        return root / "upxo" / "paths.json"

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> UPXOPaths:
        location = _path(path or cls.default_path())
        if not location.exists():
            return UPXOPaths.defaults().validate()
        try:
            data = json.loads(location.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"Unable to read UPXO configuration: {location}") from exc
        if not isinstance(data, dict):
            raise ConfigurationError("UPXO configuration must be a JSON object")
        return UPXOPaths.from_dict(data, base=location.parent)

    @classmethod
    def save(cls, config: UPXOPaths, path: str | os.PathLike[str] | None = None) -> Path:
        location = _path(path or cls.default_path())
        config.validate()
        location.parent.mkdir(parents=True, exist_ok=True)
        config.updated_at = _now()
        payload = json.dumps(config.as_dict(), indent=2, sort_keys=False) + "\n"
        fd, temporary = tempfile.mkstemp(prefix=f".{location.name}.", suffix=".tmp", dir=location.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, location)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return location

