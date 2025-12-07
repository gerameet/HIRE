"""Experiment tracking and comparison system.

This module provides infrastructure for tracking experiment runs, persisting
metadata and results, and comparing multiple experiments.
"""

import json
import time
import hashlib
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass
class ExperimentRun:
    """Complete metadata for an experiment run.

    Captures configuration, execution environment, outputs, and metrics
    for reproducibility and comparison.
    """

    # Identifiers
    experiment_id: str
    experiment_name: str
    timestamp: str

    # Configuration
    config: Dict[str, Any]
    config_hash: str

    # Environment
    device: str

    # Outputs
    output_dir: str

    # Optional fields (with defaults)
    git_commit: Optional[str] = None
    command: Optional[str] = None
    num_images: int = 0
    num_parts: int = 0

    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Timing
    duration_seconds: float = 0.0

    # Status
    status: str = "running"  # running, completed, failed
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRun":
        """Create from dictionary."""
        return cls(**data)


class ExperimentTracker:
    """Manages experiment tracking and persistence.

    Stores all experiment runs in a JSONL file and provides querying
    and comparison capabilities.
    """

    def __init__(self, experiments_dir: str = "experiments"):
        """Initialize tracker.

        Args:
            experiments_dir: Directory to store experiment data
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        self.runs_file = self.experiments_dir / "runs.jsonl"

    def generate_experiment_id(self, name: str) -> str:
        """Generate unique experiment ID.

        Args:
            name: Experiment name

        Returns:
            Unique ID in format: name_YYYYMMDD_HHMMSS_hash
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Short hash for uniqueness
        hash_str = hashlib.md5(f"{name}_{time.time()}".encode()).hexdigest()[:4]

        # Sanitize name for filesystem
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

        return f"{safe_name}_{timestamp}_{hash_str}"

    def get_git_commit(self) -> Optional[str]:
        """Get current git commit hash if available."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    def compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute deterministic hash of configuration.

        Args:
            config: Configuration dictionary

        Returns:
            MD5 hash of config
        """
        # Sort keys for deterministic hash
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def start_run(
        self,
        name: str,
        config: Dict[str, Any],
        device: str,
        command: Optional[str] = None,
    ) -> ExperimentRun:
        """Start tracking a new experiment run.

        Args:
            name: Experiment name
            config: Full configuration
            device: Device used (cuda/cpu)
            command: Command line used (if available)

        Returns:
            ExperimentRun object
        """
        experiment_id = self.generate_experiment_id(name)
        timestamp = datetime.now().isoformat()
        config_hash = self.compute_config_hash(config)
        git_commit = self.get_git_commit()

        # Create experiment directory
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = exp_dir / "config.yaml"
        try:
            import yaml

            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        except ImportError:
            # Fallback to JSON if YAML not available
            config_path = exp_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        run = ExperimentRun(
            experiment_id=experiment_id,
            experiment_name=name,
            timestamp=timestamp,
            config=config,
            config_hash=config_hash,
            device=device,
            git_commit=git_commit,
            command=command,
            output_dir=str(exp_dir),
            status="running",
        )

        # Save initial metadata
        self._save_metadata(run)

        return run

    def complete_run(
        self,
        run: ExperimentRun,
        metrics: Dict[str, Any],
        duration: float,
        num_images: int = 0,
        num_parts: int = 0,
    ) -> None:
        """Mark run as completed and save final metrics.

        Args:
            run: ExperimentRun to complete
            metrics: Final metrics
            duration: Total duration in seconds
            num_images: Number of images processed
            num_parts: Number of parts discovered
        """
        run.status = "completed"
        run.metrics = metrics
        run.duration_seconds = duration
        run.num_images = num_images
        run.num_parts = num_parts

        # Save metadata
        self._save_metadata(run)

        # Append to runs log
        self._append_to_log(run)

    def fail_run(self, run: ExperimentRun, error: str) -> None:
        """Mark run as failed.

        Args:
            run: ExperimentRun that failed
            error: Error message
        """
        run.status = "failed"
        run.error_message = error

        self._save_metadata(run)
        self._append_to_log(run)

    def _save_metadata(self, run: ExperimentRun) -> None:
        """Save metadata JSON for a run."""
        exp_dir = Path(run.output_dir)
        metadata_path = exp_dir / "metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(run.to_dict(), f, indent=2)

    def _append_to_log(self, run: ExperimentRun) -> None:
        """Append run to JSONL log."""
        with open(self.runs_file, "a") as f:
            f.write(json.dumps(run.to_dict()) + "\n")

    def load_all_runs(self) -> List[ExperimentRun]:
        """Load all tracked experiments.

        Returns:
            List of all experiment runs
        """
        if not self.runs_file.exists():
            return []

        runs = []
        with open(self.runs_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    runs.append(ExperimentRun.from_dict(data))
                except json.JSONDecodeError:
                    continue

        return runs

    def load_run(self, experiment_id: str) -> Optional[ExperimentRun]:
        """Load a specific experiment run.

        Args:
            experiment_id: Experiment ID to load

        Returns:
            ExperimentRun or None if not found
        """
        exp_dir = self.experiments_dir / experiment_id
        metadata_path = exp_dir / "metadata.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            data = json.load(f)
            return ExperimentRun.from_dict(data)

    def query_runs(
        self,
        name_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
        min_metric: Optional[Dict[str, float]] = None,
        config_filter: Optional[Dict[str, Any]] = None,
    ) -> List[ExperimentRun]:
        """Query experiments with filters.

        Args:
            name_filter: Filter by experiment name (substring match)
            status_filter: Filter by status (completed, failed, running)
            min_metric: Minimum metric values (e.g., {"precision": 0.5})
            config_filter: Filter by config values

        Returns:
            Filtered list of runs
        """
        all_runs = self.load_all_runs()
        filtered = []

        for run in all_runs:
            # Name filter
            if name_filter and name_filter not in run.experiment_name:
                continue

            # Status filter
            if status_filter and run.status != status_filter:
                continue

            # Metric filter
            if min_metric:
                skip = False
                for metric_name, min_value in min_metric.items():
                    if metric_name not in run.metrics:
                        skip = True
                        break
                    if run.metrics[metric_name] < min_value:
                        skip = True
                        break
                if skip:
                    continue

            # Config filter
            if config_filter:
                skip = False
                for key, value in config_filter.items():
                    # Support nested keys like "embedding.method"
                    config_value = run.config
                    for part in key.split("."):
                        if isinstance(config_value, dict) and part in config_value:
                            config_value = config_value[part]
                        else:
                            skip = True
                            break

                    if skip or config_value != value:
                        skip = True
                        break

                if skip:
                    continue

            filtered.append(run)

        return filtered


class ResultsComparator:
    """Compare results across multiple experiments."""

    def __init__(self, tracker: ExperimentTracker):
        """Initialize comparator.

        Args:
            tracker: ExperimentTracker instance
        """
        self.tracker = tracker

    def compare_metrics(
        self, experiment_ids: List[str], metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare metrics across experiments.

        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: Specific metrics to compare (None = all)

        Returns:
            Comparison data with metrics for each experiment
        """
        runs = [self.tracker.load_run(eid) for eid in experiment_ids]
        runs = [r for r in runs if r is not None]

        if not runs:
            return {}

        # Collect all metric names
        all_metrics = set()
        for run in runs:
            all_metrics.update(run.metrics.keys())

        # Filter to requested metrics
        if metrics:
            all_metrics = set(metrics) & all_metrics

        # Build comparison table
        comparison = {
            "experiments": [
                {
                    "experiment_id": run.experiment_id,
                    "name": run.experiment_name,
                    "timestamp": run.timestamp,
                    "status": run.status,
                    "metrics": {m: run.metrics.get(m, None) for m in all_metrics},
                }
                for run in runs
            ],
            "metric_names": sorted(all_metrics),
        }

        return comparison

    def export_comparison_csv(
        self,
        experiment_ids: List[str],
        output_path: str,
        metrics: Optional[List[str]] = None,
    ) -> None:
        """Export comparison as CSV.

        Args:
            experiment_ids: Experiments to compare
            output_path: Path to save CSV
            metrics: Metrics to include
        """
        import csv

        comparison = self.compare_metrics(experiment_ids, metrics)

        if not comparison:
            return

        metric_names = comparison["metric_names"]

        with open(output_path, "w", newline="") as f:
            fieldnames = ["experiment_id", "name", "timestamp", "status"] + metric_names
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()

            for exp in comparison["experiments"]:
                row = {
                    "experiment_id": exp["experiment_id"],
                    "name": exp["name"],
                    "timestamp": exp["timestamp"],
                    "status": exp["status"],
                }
                row.update(exp["metrics"])
                writer.writerow(row)

    def generate_comparison_plots(
        self,
        experiment_ids: List[str],
        output_dir: str,
        metrics: Optional[List[str]] = None,
    ) -> None:
        """Generate comparison plots.

        Args:
            experiment_ids: Experiments to compare
            output_dir: Directory to save plots
            metrics: Metrics to plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib/seaborn not available. Skipping plots.")
            return

        comparison = self.compare_metrics(experiment_ids, metrics)

        if not comparison:
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Plot each metric
        for metric_name in comparison["metric_names"]:
            values = []
            labels = []

            for exp in comparison["experiments"]:
                metric_value = exp["metrics"].get(metric_name)
                if metric_value is not None:
                    values.append(metric_value)
                    labels.append(exp["name"][:20])  # Truncate long names

            if not values:
                continue

            # Create bar plot
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(values)), values)
            plt.xticks(range(len(values)), labels, rotation=45, ha="right")
            plt.ylabel(metric_name)
            plt.title(f"Comparison: {metric_name}")
            plt.tight_layout()

            plot_path = output_path / f"compare_{metric_name.replace('/', '_')}.png"
            plt.savefig(plot_path)
            plt.close()
