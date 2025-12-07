"""Interactive metrics comparison dashboards.

Creates comprehensive visualizations for comparing experiments and analyzing
performance across different configurations.
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def create_metrics_dashboard(
    experiments: List[Dict[str, Any]],
    output_path: str,
    metrics_to_plot: Optional[List[str]] = None
):
    """Create comprehensive metrics comparison dashboard.
    
    Args:
        experiments: List of experiment metadata dictionaries
        output_path: Path to save HTML dashboard
        metrics_to_plot: Specific metrics to visualize (None = all)
        
    Returns:
        Plotly Figure object
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plotly required. Install with: pip install plotly")
    
    logger.info(f"Creating dashboard for {len(experiments)} experiments")
    
    # Extract metrics
    df = _experiments_to_dataframe(experiments)
    
    # Determine metrics to plot
    if metrics_to_plot is None:
        metrics_to_plot = [
            col for col in df.columns
            if col not in ['experiment_id', 'name', 'timestamp', 'duration', 'config', 'status']
            and pd.api.types.is_numeric_dtype(df[col])
        ]
    
    # Filter to available metrics
    metrics_to_plot = [m for m in metrics_to_plot if m in df.columns]
    
    if not metrics_to_plot:
        logger.warning("No numeric metrics found to plot")
        return None
    
    # Create subplots
    n_metrics = len(metrics_to_plot)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=metrics_to_plot,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Add traces for each metric
    for idx, metric in enumerate(metrics_to_plot):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        # Bar chart for each experiment
        fig.add_trace(
            go.Bar(
                x=df['name'],
                y=df[metric],
                name=metric,
                text=df[metric].round(4),
                textposition='auto',
                showlegend=False,
                marker=dict(
                    color=df[metric],
                    colorscale='Viridis',
                    showscale=False
                )
            ),
            row=row,
            col=col
        )
        
        # Update axes
        fig.update_xaxes(title_text="Experiment", row=row, col=col, tickangle=-45)
        fig.update_yaxes(title_text=metric.replace('_', ' ').title(), row=row, col=col)
    
    # Update layout
    fig.update_layout(
        title_text="Experiment Metrics Comparison",
        height=400 * n_rows,
        width=1200,
        showlegend=False,
        font=dict(size=10)
    )
    
    # Save HTML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    logger.info(f"Saved metrics dashboard to {output_path}")
    
    return fig


def generate_comparison_report(
    experiment_ids: List[str],
    output_dir: Path,
    tracker=None
) -> None:
    """Generate comprehensive comparison report between experiments.
    
    Creates:
    - metrics_comparison.csv
    - config_diff.txt
    - visualizations/dashboard.html
    - visualizations/timeline.html
    - report.html
    
    Args:
        experiment_ids: List of experiment IDs to compare
        output_dir: Directory to save report files
        tracker: ExperimentTracker instance (optional)
    """
    logger.info(f"Generating comparison report for {len(experiment_ids)} experiments")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load experiments
    if tracker is None:
        from ..tracking import ExperimentTracker
        tracker = ExperimentTracker()
    
    experiments = []
    for exp_id in experiment_ids:
        try:
            exp = tracker.get_experiment(exp_id)
            if exp:
                experiments.append(exp)
        except Exception as e:
            logger.warning(f"Could not load experiment {exp_id}: {e}")
    
    if not experiments:
        logger.error("No valid experiments found")
        return
    
    # 1. Create metrics comparison CSV
    metrics_df = _compare_metrics(experiments)
    metrics_csv = output_dir / "metrics_comparison.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    logger.info(f"Saved metrics comparison to {metrics_csv}")
    
    # 2. Generate config diff
    config_diff = _generate_config_diff(experiments)
    config_diff_path = output_dir / "config_diff.txt"
    config_diff_path.write_text(config_diff)
    logger.info(f"Saved config diff to {config_diff_path}")
    
    # 3. Create visualizations
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Main dashboard
    create_metrics_dashboard(
        [exp.__dict__ if hasattr(exp, '__dict__') else exp for exp in experiments],
        viz_dir / "dashboard.html"
    )
    
    # Timeline visualization
    create_experiment_timeline(experiments, viz_dir / "timeline.html")
    
    # 4. Generate HTML report
    html_report = _generate_html_report(
        experiments,
        metrics_df,
        config_diff,
        viz_dir
    )
    report_path = output_dir / "report.html"
    report_path.write_text(html_report)
    logger.info(f"Saved comparison report to {report_path}")
    
    logger.info(f"Comparison report complete: {output_dir}")


def create_experiment_timeline(
    experiments: List[Dict[str, Any]],
    output_path: str
):
    """Create timeline visualization of experiments.
    
    Args:
        experiments: List of experiment dictionaries
        output_path: Path to save HTML
        
    Returns:
        Plotly Figure
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        return None
    
    # Extract timeline data
    df = _experiments_to_dataframe(experiments)
    
    if 'timestamp' not in df.columns or 'duration' not in df.columns:
        logger.warning("Missing timestamp or duration data for timeline")
        return None
    
    # Create Gantt-style chart
    fig = go.Figure()
    
    for idx, row in df.iterrows():
        start_time = pd.to_datetime(row['timestamp'])
        end_time = start_time + pd.Timedelta(seconds=row.get('duration', 0))
        
        fig.add_trace(go.Scatter(
            x=[start_time, end_time],
            y=[idx, idx],
            mode='lines+markers',
            name=row['name'],
            line=dict(width=10),
            hovertemplate=f"<b>{row['name']}</b><br>"
                         f"Start: {start_time}<br>"
                         f"Duration: {row.get('duration', 0):.1f}s<br>"
                         "<extra></extra>"
        ))
    
    fig.update_layout(
        title="Experiment Timeline",
        xaxis_title="Time",
        yaxis_title="Experiment",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(df))),
            ticktext=df['name'].tolist()
        ),
        height=max(400, len(experiments) * 50),
        showlegend=False
    )
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    
    return fig


def _experiments_to_dataframe(experiments: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert experiment list to pandas DataFrame."""
    records = []
    
    for exp in experiments:
        # Handle both dict and object formats
        if hasattr(exp, '__dict__'):
            exp_dict = exp.__dict__.copy()
        else:
            exp_dict = exp.copy()
        
        # Flatten metrics
        metrics = exp_dict.pop('metrics', {})
        record = {**exp_dict, **metrics}
        
        # Convert metadata to JSON string
        if 'metadata' in record and isinstance(record['metadata'], dict):
            record['metadata'] = json.dumps(record['metadata'])
        
        records.append(record)
    
    return pd.DataFrame(records)


def _compare_metrics(experiments: List) -> pd.DataFrame:
    """Create metrics comparison dataframe."""
    df = _experiments_to_dataframe(experiments)
    
    # Select relevant columns
    id_cols = ['experiment_id', 'name', 'timestamp']
    metric_cols = [
        col for col in df.columns
        if col not in id_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    return df[id_cols + metric_cols]


def _generate_config_diff(experiments: List) -> str:
    """Generate configuration difference report."""
    if len(experiments) < 2:
        return "Need at least 2 experiments for comparison"
    
    lines = ["# Configuration Differences\n\n"]
    
    # Compare first experiment with others
    base_exp = experiments[0]
    base_config = base_exp.config if hasattr(base_exp, 'config') else {}
    
    for i, exp in enumerate(experiments[1:], 1):
        exp_config = exp.config if hasattr(exp, 'config') else {}
        
        lines.append(f"## {base_exp.name if hasattr(base_exp, 'name') else 'Exp 1'} vs "
                    f"{exp.name if hasattr(exp, 'name') else f'Exp {i+1}'}\n\n")
        
        diffs = _dict_diff(base_config, exp_config)
        if diffs:
            for key, (val1, val2) in diffs.items():
                lines.append(f"- **{key}**: {val1} → {val2}\n")
        else:
            lines.append("No differences\n")
        
        lines.append("\n")
    
    return "".join(lines)


def _dict_diff(dict1: Dict, dict2: Dict, prefix: str = "") -> Dict:
    """Recursively find differences between dictionaries."""
    diffs = {}
    
    all_keys = set(dict1.keys()) | set(dict2.keys())
    
    for key in all_keys:
        full_key = f"{prefix}.{key}" if prefix else key
        
        if key not in dict1:
            diffs[full_key] = (None, dict2[key])
        elif key not in dict2:
            diffs[full_key] = (dict1[key], None)
        elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            nested_diffs = _dict_diff(dict1[key], dict2[key], full_key)
            diffs.update(nested_diffs)
        elif dict1[key] != dict2[key]:
            diffs[full_key] = (dict1[key], dict2[key])
    
    return diffs


def _generate_html_report(
    experiments: List,
    metrics_df: pd.DataFrame,
    config_diff: str,
    viz_dir: Path
) -> str:
    """Generate HTML comparison report."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Experiment Comparison Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1, h2 {{
            color: #333;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric {{
            font-weight: bold;
            color: #2196F3;
        }}
        pre {{
            background: #f4f4f4;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }}
        iframe {{
            width: 100%;
            height: 600px;
            border: none;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>🔬 Experiment Comparison Report</h1>
    
    <div class="section">
        <h2>📊 Summary</h2>
        <p><strong>Number of Experiments:</strong> {len(experiments)}</p>
        <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>📈 Metrics Comparison</h2>
        {metrics_df.to_html(classes='table', index=False)}
    </div>
    
    <div class="section">
        <h2>⚙️ Configuration Differences</h2>
        <pre>{config_diff}</pre>
    </div>
    
    <div class="section">
        <h2>📊 Interactive Dashboard</h2>
        <iframe src="visualizations/dashboard.html"></iframe>
    </div>
    
    <div class="section">
        <h2>⏱️ Experiment Timeline</h2>
        <iframe src="visualizations/timeline.html"></iframe>
    </div>
</body>
</html>
"""
    return html
