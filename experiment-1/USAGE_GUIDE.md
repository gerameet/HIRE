# Experiment-1 Usage Guide

Quick reference for common tasks and workflows.

## Getting Started

### 1. Installation

```bash
cd experiment-1/scripts
pip install -r requirements-phase1.txt
```

### 2. Download Models

```bash
# List available models
./experiment models list

# Download models you need
./experiment models download dinov2-vitb14
./experiment models download sam-vit-b

# Or download all at once (large download!)
./experiment models download all
```

### 3. Run Your First Experiment

```bash
# Quick test (dummy models, 3 images)
./experiment run --quick

# Real experiment with DINOv2
./experiment run \
  --name "my_first_run" \
  -o embedding.method=dinov2 \
  -o embedding.model_size=base \
  -o data.max_images=10
```

---

## Common Workflows

### Research Experiment

Testing different embedding methods:

```bash
# Baseline with CLIP
./experiment run --name "baseline_clip" \
  -o embedding.method=clip \
  -o data.max_images=50

# DINOv2 comparison
./experiment run --name "baseline_dinov2" \
  -o embedding.method=dinov2 \
  -o embedding.model_size=base \
  -o data.max_images=50

# Compare results
./experiment compare baseline_clip baseline_dinov2 \
  --output reports/embedding_comparison
```

### Model Size Comparison

```bash
# Test different DINOv2 sizes
for size in small base large; do
  ./experiment run \
    --name "dinov2_${size}" \
    -o embedding.method=dinov2 \
    -o embedding.model_size=$size \
    -o data.max_images=20
done

# Visualize all results
./experiment visualize compare \
  dinov2_small dinov2_base dinov2_large \
  --output reports/model_size_comparison
```

### Segmentation Model Comparison

```bash
# Compare segmentation methods
./experiment run --name "seg_sam" \
  -o segmentation.model=sam \
  -o data.max_images=20

./experiment run --name "seg_yolo" \
  -o segmentation.model=yolo \
  -o data.max_images=20

./experiment run --name "seg_mask2former" \
  -o segmentation.model=mask2former \
  -o data.max_images=20

# Compare
./experiment compare seg_sam seg_yolo seg_mask2former
```

### Semantic Hierarchy & Labeling

Enable knowledge-integrated hierarchy building and automatic part labeling.

```bash
# Run with semantic hierarchy (WordNet+CLIP) and auto-labeling
./experiment run \
  --name "semantic_run" \
  --semantic-hierarchy \
  --auto-label \
  -o embedding.method=clip \
  -o data.max_images=10
```

---

## Visualization Workflows

### Embedding Space Analysis

```bash
# Run experiment with embeddings
./experiment run --name "analysis_run" \
  -o embedding.method=dinov2 \
  -o data.max_images=100

# Get experiment ID
./experiment list | grep analysis_run

# Visualize embedding space (replace EXP_ID)
./experiment visualize embeddings EXP_ID \
  --method umap \
  --color-by label \
  --output analysis/embedding_space.html

# Try different methods
./experiment visualize embeddings EXP_ID --method tsne
./experiment visualize embeddings EXP_ID --method pca

# Cluster analysis
./experiment visualize clusters EXP_ID \
  --n-clusters 10 \
  --output analysis/clusters.html
```

### Attention Analysis

```bash
# Visualize generic attention
./experiment run \
  --visualize-attention \
  -o data.max_images=5

# Resulting heatmaps: output/hierarchical/visualizations/attention/
```

### Experiment Comparison

```bash
# Compare multiple experimental conditions
./experiment visualize compare exp1 exp2 exp3 \
  --output reports/full_comparison

# Opens HTML report at:
# reports/full_comparison/report.html
```

---

## Configuration Tips

### Quick Mode for Development

```bash
# Test pipeline changes quickly
./experiment run --quick --name "dev_test"
```

This uses:
- Dummy models (instant)
- Only 3 images
- Minimal processing

### Production Settings

```yaml
# production_config.yaml
experiment:
  name: "production_run"
  save_intermediates: true

data:
  images_path: "/data/large_dataset"
  max_images: null  # Process all

segmentation:
  model: "mask2former"
  min_confidence: 0.7
  min_area: 500

embedding:
  method: "dinov2"
  model_size: "large"
  use_cache: true

evaluation:
  enabled: true
  tasks: ["retrieval", "classification"]
  top_k: 10

gpu:
  device: "cuda"
```

```bash
./experiment run --config production_config.yaml
```

### GPU Management

```bash
# Force CPU (testing on laptop)
./experiment run --device cpu --quick

# Specify GPU
./experiment run --device cuda

# Auto-detect (default)
./experiment run
```

---

## Troubleshooting

### Models Not Found

```bash
# Check what's available
./experiment models list

# Download missing models
./experiment models download MODEL_NAME
```

### Out of Memory

1. Reduce batch size in config:
   ```yaml
   gpu:
     batch_size: 16  # Default 32
   ```

2. Use smaller model:
   ```bash
   -o embedding.model_size=small
   ```

3. Process fewer images:
   ```bash
   -o data.max_images=10
   ```

### Slow Performance

- Enable caching:
  ```bash
  -o embedding.use_cache=true
  ```

- Use pre-computed embeddings if re-running

### Verify Installation

```bash
# Quick test
./experiment run --quick

# Verify models
./experiment models verify
```

---

## Advanced Usage

### Custom Hierarchy Parameters

```bash
./experiment run \
  -o hierarchy.spatial_threshold=0.4 \
  -o hierarchy.containment_threshold=0.8 \
  -o hierarchy.max_depth=7
```

### Multiple Override Formats

```bash
# Boolean
-o experiment.quick_mode=true

# Numbers
-o data.max_images=50

# Strings
-o experiment.name=\"my experiment\"

# Lists (JSON format)
-o evaluation.tasks='[\"retrieval\", \"classification\"]'
```

### Experiment Management

```bash
# List recent experiments
./experiment list

# Show specific experiment
./experiment show EXP_ID --config

# Clean up failed runs
./experiment clean --status failed --force

# Delete specific experiment
./experiment clean --experiment-id EXP_ID
```

---

## Model Reference

### DINOv2 Models

| Size  | Params | Dim | Memory | Speed |
|-------|--------|-----|--------|-------|
| small | 22M    | 384 | ~1GB   | Fast  |
| base  | 86M    | 768 | ~2GB   | Med   |
| large | 307M   | 1024| ~4GB   | Slow  |
| giant | 1.1B   | 1536| ~8GB   | V.Slow|

### CLIP Models

| Model      | Dim | Memory | Use Case |
|------------|-----|--------|----------|
| base-32    | 512 | ~2GB   | General  |
| large-14   | 768 | ~4GB   | Quality  |

### Segmentation Models

| Model        | Speed | Quality | Memory |
|--------------|-------|---------|--------|
| dummy        | +++   | -       | Minimal|
| yolo         | ++    | +       | 2GB    |
| sam          | +     | +++     | 4GB    |
| mask2former  | +     | ++      | 3GB    |

---

## Best Practices

1. **Start Small**: Use `--quick` to test pipelines
2. **Cache Everything**: Enable embedding cache for iteration
3. **Version Control**: Keep config files in git
4. **Name Experiments**: Use descriptive names
5. **Compare Often**: Use visualize compare for insights
6. **Monitor Resources**: Watch GPU memory usage
7. **Save Intermediates**: Enable for debugging
8. **Document Results**: Use comparison reports

---

## Getting Help

```bash# View command help
./experiment --help
./experiment run --help
./experiment models --help
./experiment visualize --help
```

For more details, see:
- Main README: `experiment-1/README.md`
- API docs: `hierarchical_pipeline/README.md`
- Examples: `experiment-1/examples/` (coming soon)
