# Experiment-1: Hierarchical Visual Pipeline

A unified research framework for hierarchical visual representations through compositional parsing, object-centric learning, and structured embeddings.

## Quick Start

```bash
# Navigate to scripts directory
cd experiment-1/scripts

# Run a quick test experiment (3 images, dummy models, <1 minute)
./experiment run --quick

# Run a full experiment with real models
./experiment run --name "my_experiment" \
  -o segmentation.model=mask2former \
  -o embedding.method=dinov2

# List all experiments
./experiment list

# Compare multiple experiments
./experiment compare exp_A exp_B exp_C --output comparison/
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements-phase1.txt
```

## Unified CLI

All experiment operations are performed through a single `experiment` command:

### Commands

- **`run`** — Execute full pipeline (segmentation → hierarchy → embeddings → evaluation)
- **`list`** — List all tracked experiments
- **`show`** — Show details of a specific experiment
- **`compare`** — Compare metrics across multiple experiments
- **`evaluate`** — Run evaluation on existing experiment results
- **`clean`** — Clean up experiment outputs
- **`models`** — Manage model downloads and cache ✨ NEW
- **`visualize`** — Generate visualizations for experiments ✨ NEW

### Run Experiments

```bash
# Full pipeline with default config
./experiment run

# Custom configuration
./experiment run --config my_config.yaml

# Quick test mode (3 images, fast models)
./experiment run --quick

# Override specific settings
./experiment run \
  --name "production_test" \
  -o data.max_images=20 \
  -o segmentation.model=yolo \
  -o embedding.method=dinov2 \
  -o embedding.model_size=base \
  -o evaluation.enabled=true

# Use specific device
./experiment run --device cuda
```

### Model Management ✨ NEW

```bash
# List all available models
./experiment models list

# Download specific model
./experiment models download sam-vit-b

# Download all required models for a config
./experiment models download all

# Verify model integrity
./experiment models verify sam-vit-b
./experiment models verify  # Verify all

# Clean model cache
./experiment models clean sam-vit-b  # Specific model
./experiment models clean --force    # All models (no confirm)
```

### Visualization ✨ NEW

```bash
# Visualize embedding space
./experiment visualize embeddings exp_ABC123 \
  --method umap \
  --color-by label \
  --output embeddings.html

# Cluster analysis
./experiment visualize clusters exp_ABC123 \
  --n-clusters 10 \
  --method tsne

# Generate comparison report
./experiment visualize compare exp_A exp_B exp_C \
  --output reports/comparison
```

### Track and Compare

```bash
# List all experiments
./experiment list

# Filter by status or name
./experiment list --status completed
./experiment list --name "baseline"

# Show experiment details
./experiment show exp_20231207_143022_a3f8

# Compare multiple experiments
./experiment compare exp_A exp_B exp_C \
  --metrics precision@5 recall@5 \
  --output comparison_report/
```

### Clean Up

```bash
# Delete specific experiment
./experiment clean --experiment-id exp_ABC123

# Delete all failed experiments
./experiment clean --status failed --force
```

## Configuration

Experiments are configured via YAML files with CLI overrides:

Default config: `hierarchical_pipeline/configs/default_experiment.yaml`

```yaml
experiment:
  name: "baseline_experiment"
  quick_mode: false
  save_intermediates: true
  seed: 42

data:
  images_path: "../images"
  max_images: null  # null = all images

segmentation:
  model: "dummy"  # dummy, sam, yolo, mask2former, etc.
  checkpoint: null
  min_confidence: 0.5
  min_area: 100

embedding:
  method: "dinov2"  # dummy, clip, dinov2, mae
  cache_dir: "cache/embeddings"
  use_cache: true

hierarchy:
  method: "bottom_up"
  spatial_threshold: 0.3
  containment_threshold: 0.7
  max_depth: 5

evaluation:
  enabled: false
  tasks: ["retrieval"]
  top_k: 5

output:
  save_parse_graphs: true
  save_embeddings: true
  save_visualizations: true

gpu:
  device: null  # auto-detect
  allow_cpu_fallback: true
```

**Override from CLI:**
```bash
./experiment run -o data.max_images=10 -o embedding.method=clip
```

## Experiment Tracking

All experiments are automatically tracked with full metadata:

```
experiments/
├── runs.jsonl                        # All experiment metadata
├── exp_baseline_20231207_143022_a3f8/
│   ├── config.yaml                   # Full configuration
│   ├── metadata.json                 # Run metadata
│   ├── graphs/                       # Parse graphs (.graph.json)
│   │   ├── image1.graph.json
│   │   └── image2.graph.json
│   ├── visualizations/               # Visualizations
│   │   ├── image1.baseline.png
│   │   ├── image1.hierarchy_overlay.png
│   │   └── image1.graph.html
│   ├── embeddings.npz                # Embeddings
│   ├── evaluation/                   # Evaluation results
│   │   └── retrieval_results.json
│   └── metrics.json                  # Aggregated metrics
└── ...
```

**Tracked metadata:**
- Timestamp, duration, device used
- Full configuration and config hash
- Git commit (if available)
- Number of images/parts processed
- All metrics and evaluation results
- Status (running, completed, failed)

## Available Models

### Segmentation Models
- `dummy` — Random masks (no dependencies, testing only)
- `sam` — Segment Anything Model (requires checkpoint)
- `yolo` — YOLOv8 instance segmentation
- `mask2former` — Mask2Former (HuggingFace)
- `segformer` — SegFormer (HuggingFace)
- `clipseg` — Text-prompted segmentation
- `detectron2` — Detectron2 models

### Embedding Methods
- `dummy` — Random embeddings (testing, no dependencies)
- `clip` — OpenAI CLIP (text-image alignment)
- `dinov2` — Meta DINO v2 (best quality)
- `mae` — Masked Autoencoder

## Examples

### Example 1: Quick Smoke Test

```bash
cd experiment-1/scripts
./experiment run --quick
```

**What it does:**
- Processes 3 images
- Uses dummy segmentation and embeddings (fast)
- Creates experiment directory with results
- Takes < 1 minute

### Example 2: Production Pipeline

```bash
./experiment run --name "production_v1" \
  -o data.images_path=../images \
  -o segmentation.model=mask2former \
  -o embedding.method=dinov2 \
  -o evaluation.enabled=true \
  -o evaluation.tasks=["retrieval","classification"]
```

**What it does:**
- Runs high-quality segmentation
- Generates DINO embeddings
- Builds hierarchies
- Runs retrieval and classification evaluation
- Saves all results with tracking

### Example 3: Systematic Comparison

```bash
# Run multiple configurations
./experiment run --name "baseline_dummy" -o embedding.method=dummy
./experiment run --name "baseline_clip" -o embedding.method=clip
./experiment run --name "baseline_dino" -o embedding.method=dinov2
./experiment run --name "baseline_mae" -o embedding.method=mae

# Compare results
./experiment list
./experiment compare \
  baseline_dummy_* baseline_clip_* baseline_dino_* baseline_mae_* \
  --output comparison/
```

**What it does:**
- Runs 4 experiments with different embeddings
- Generates comparison CSV and plots
- Easy to see which embedding works best

## Project Structure

```
experiment-1/
├── README.md                    # This file
├── requirements-phase1.txt      # Dependencies
├── images/                      # Sample images
├── experiments/                 # Tracked experiments (auto-created)
└── scripts/                     
    ├── experiment               # Main CLI (entry point)
    ├── hierarchical_pipeline/   # Core pipeline package
    │   ├── tracking/            # Experiment tracking
    │   │   ├── experiment_tracker.py
    │   │   └── experiment_runner.py
    │   ├── core/                # Data structures, interfaces
    │   ├── embedding/           # Embedding methods
    │   ├── evaluation/          # Evaluation tools
    │   │   └── evaluator.py     # Consolidated evaluator
    │   ├── adapters/            # Segmentation adapters
    │   ├── utils/               # GPU management, utilities
    │   └── configs/             # Configuration templates
    └── segmentation_pipeline/   # Segmentation models package
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `command not found: experiment` | Run `chmod +x experiment` or use `python experiment` |
| Missing checkpoint | Download model checkpoint or use `--quick` mode |
| CUDA out of memory | Use `--device cpu` or reduce batch size in config |
| Module not found | Install dependencies: `pip install -r requirements-phase1.txt` |

## Development

### Running Tests

```bash
cd experiment-1/scripts

# Test data models
python test_data_models.py

# Test hierarchical setup
python test_hierarchical_setup.py

# Run with pytest (if installed)
pytest -v
```

### Adding New Evaluation Tasks

Extend `hierarchical_pipeline/evaluation/evaluator.py`:

```python
class MyCustomEvaluator(EvaluationTask):
    def run(self, parts):
        # Your evaluation logic
        return {"my_metric": value}
```

## Next Steps

1. **Try the quick start** (3-image test)
2. **Run your first full experiment**
3. **Compare different configurations**
4. **Explore tracked experiment metadata**

## License

Research project - see main repository for license information.

## Visualization & Analysis

### Embedding Space Visualization

Explore high-dimensional embedding spaces in 2D:

```python
from hierarchical_pipeline.visualization import plot_embedding_space

# Visualize with UMAP
fig = plot_embedding_space(
    parts=all_parts,
    method="umap",  # or "tsne", "pca"
    color_by="label",
    save_path="embedding_space.html"
)
```

**Supported Dimensionality Reduction**:
- **UMAP**: Preserves both local and global structure (recommended)
- **t-SNE**: Emphasizes local neighborhoods  
- **PCA**: Fast linear reduction

**Coloring Options**:
- `label`: Semantic categories
- `image`: Source image
- `level`: Hierarchy level
- `confidence`: Detection confidence
- `area`: Part area in pixels

### Clustering Analysis

Discover semantic clusters automatically:

```python
from hierarchical_pipeline.visualization import plot_embedding_clusters

fig, labels = plot_embedding_clusters(
    parts=all_parts,
    n_clusters=10,
    method="umap",
    save_path="clusters.html"
)
```

### Comparison Reports

Generate comprehensive HTML reports comparing multiple experiments:

```python
from hierarchical_pipeline.visualization import generate_comparison_report

generate_comparison_report(
    experiment_ids=["exp_001", "exp_002", "exp_003"],
    output_dir="reports/comparison"
)
```

**Report Includes**:
- Metrics comparison table (CSV + interactive charts)
- Configuration differences
- Experiment timeline visualization
- Interactive dashboards with Plotly

