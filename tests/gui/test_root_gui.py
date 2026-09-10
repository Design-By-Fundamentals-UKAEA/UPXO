"""
test_root_gui.py
================
Automated unit tests for UPXO Root GUI pipeline registry, configuration persistence,
parameter delegation, and recent pipeline history tracking.
"""
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

_REPO_ROOT = Path(__file__).parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from upxo.gui.root.pipeline_registry import (
    PipelineEntry,
    PipelineRegistry,
    SidebarItem,
    SidebarSection,
)
from upxo.gui.root.pipelines import build_registry, get_registry
from upxo.gui.root.app import load_gui_config, save_gui_config


class TestPipelineRegistry(unittest.TestCase):
    def test_pipeline_entry_validation(self):
        # Valid subprocess entry with extra_args
        entry = PipelineEntry(
            key="test_sub",
            title="Test Subprocess",
            description="Test Desc",
            category="Test Cat",
            kind="subprocess",
            script_path="path/to/script.py",
            extra_args=("--flag", "val"),
        )
        self.assertEqual(entry.key, "test_sub")
        self.assertEqual(entry.extra_args, ("--flag", "val"))

        # Invalid kind raises ValueError
        with self.assertRaises(ValueError):
            PipelineEntry(
                key="invalid",
                title="Title",
                description="Desc",
                category="Cat",
                kind="invalid_kind",
            )

        # Subprocess without script_path raises ValueError
        with self.assertRaises(ValueError):
            PipelineEntry(
                key="sub_nopath",
                title="Title",
                description="Desc",
                category="Cat",
                kind="subprocess",
            )

    def test_registry_assembly(self):
        registry = build_registry()
        self.assertIsInstance(registry, PipelineRegistry)
        all_entries = registry.all()
        self.assertGreater(len(all_entries), 0)

        categories = registry.categories()
        self.assertIn("Image Operations", categories)

        by_cat = registry.by_category()
        self.assertIn("Image Operations", by_cat)

    def test_config_persistence(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_config = Path(tmp_dir) / "gui_config.json"
            with patch("upxo.gui.root.app._get_config_path", return_value=tmp_config):
                save_gui_config({"appearance_mode": "dark", "recent_pipelines": ["mc_2d"]})
                loaded = load_gui_config()
                self.assertEqual(loaded.get("appearance_mode"), "dark")
                self.assertEqual(loaded.get("recent_pipelines"), ["mc_2d"])


if __name__ == "__main__":
    unittest.main()
