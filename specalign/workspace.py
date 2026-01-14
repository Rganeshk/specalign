"""Workspace management for specalign."""

import os
from pathlib import Path
from typing import Optional


class Workspace:
    """Manages the specalign workspace structure."""

    WORKSPACE_DIR = ".specalign"
    SUBDIRS = ["specs", "prompts", "models", "data", "results", "test_cases"]

    def __init__(self, root_path: Optional[Path] = None):
        """Initialize workspace.

        Args:
            root_path: Root path of workspace. Defaults to current directory.
        """
        self.root = Path(root_path) if root_path else Path.cwd()
        self.workspace_root = self.root / self.WORKSPACE_DIR

    @property
    def specs_dir(self) -> Path:
        """Return specs directory path."""
        return self.workspace_root / "specs"

    @property
    def prompts_dir(self) -> Path:
        """Return prompts directory path."""
        return self.workspace_root / "prompts"

    @property
    def models_dir(self) -> Path:
        """Return models directory path."""
        return self.workspace_root / "models"

    @property
    def data_dir(self) -> Path:
        """Return data directory path."""
        return self.workspace_root / "data"

    @property
    def results_dir(self) -> Path:
        """Return results directory path."""
        return self.workspace_root / "results"

    @property
    def test_cases_dir(self) -> Path:
        """Return test cases directory path."""
        return self.workspace_root / "test_cases"

    def exists(self) -> bool:
        """Check if workspace is initialized."""
        # Core directories required for workspace to exist
        core_dirs = ["specs", "prompts", "models", "data", "results"]
        return all((self.workspace_root / subdir).exists() for subdir in core_dirs)

    def initialize(self) -> None:
        """Create workspace directory structure."""
        for subdir in self.SUBDIRS:
            (self.workspace_root / subdir).mkdir(parents=True, exist_ok=True)

    def get_next_prompt_number(self) -> int:
        """Get the next available prompt number."""
        if not self.prompts_dir.exists():
            return 1

        existing = [
            int(d.name) for d in self.prompts_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        ]
        return max(existing, default=0) + 1

    def create_prompt_dir(self, number: Optional[int] = None) -> Path:
        """Create a new numbered prompt directory.

        Args:
            number: Prompt number. If None, uses next available number.

        Returns:
            Path to created prompt directory.
        """
        if number is None:
            number = self.get_next_prompt_number()

        prompt_dir = self.prompts_dir / str(number)
        prompt_dir.mkdir(parents=True, exist_ok=True)
        return prompt_dir

    def get_all_spec_files(self) -> list[Path]:
        """Get all specification files in specs directory."""
        if not self.specs_dir.exists():
            return []
        return sorted(self.specs_dir.glob("*.md"))

    def get_model_config(self, model_name: str) -> Path:
        """Get path to model configuration file.

        Args:
            model_name: Name of model config (without .yaml extension).

        Returns:
            Path to model config file.
        """
        return self.models_dir / f"{model_name}.yaml"

    def get_data_config(self, data_name: str) -> Path:
        """Get path to data configuration file.

        Args:
            data_name: Name of data config.

        Returns:
            Path to data config file.
        """
        return self.data_dir / f"{data_name}.json"
