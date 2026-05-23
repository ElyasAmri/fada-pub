"""
Results Manager
Utilities for saving and loading experiment results
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ResultsManager:
    """Manager for saving and loading experiment results."""

    def __init__(self, output_dir: Union[str, Path]) -> None:
        """
        Initialize results manager.

        Args:
            output_dir: Directory for saving results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(
        self,
        data: Dict[str, Any],
        prefix: str,
        include_timestamp: bool = True
    ) -> Path:
        """
        Save data to JSON file with optional timestamp.

        Args:
            data: Dictionary to save
            prefix: Filename prefix
            include_timestamp: Whether to add timestamp to filename

        Returns:
            Path to saved file
        """
        if include_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{prefix}_{timestamp}.json"
        else:
            filename = f"{prefix}.json"

        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)

        logger.info(f"Results saved to: {output_path}")
        return output_path

    def save_progress(self, data: Dict[str, Any], prefix: str) -> Path:
        """
        Save progress file (overwrites existing).

        Args:
            data: Dictionary to save
            prefix: Filename prefix

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / f"{prefix}_progress.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)

        return output_path

    def load_json(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Load data from JSON file.

        Args:
            filename: Name of file to load

        Returns:
            Loaded dictionary or None if file doesn't exist
        """
        file_path = self.output_dir / filename

        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_latest(self, prefix: str) -> Optional[Dict[str, Any]]:
        """
        Load the most recent file matching prefix.

        Args:
            prefix: Filename prefix to match

        Returns:
            Loaded dictionary or None if no matching files
        """
        matching_files = sorted(
            self.output_dir.glob(f"{prefix}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if not matching_files:
            logger.warning(f"No files found matching prefix: {prefix}")
            return None

        return self.load_json(matching_files[0].name)

    def list_results(self, prefix: Optional[str] = None) -> list:
        """
        List all result files, optionally filtered by prefix.

        Args:
            prefix: Optional prefix to filter by

        Returns:
            List of filenames
        """
        pattern = f"{prefix}_*.json" if prefix else "*.json"
        return sorted([f.name for f in self.output_dir.glob(pattern)])
