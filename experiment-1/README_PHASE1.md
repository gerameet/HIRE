# Phase 1 — Hierarchical Visual Pipeline (Quick Start)

This small README explains how to run the Phase 1 pipeline locally. Phase 1 provides a minimal, explorable pipeline that:

- Runs a segmentation model (e.g. `dummy`, `sam`, `mask2former`) on images
- Converts segmentation results into `Part` objects
- Builds a bottom-up parse graph (containment-based)
- Saves parse graph JSON and optional mask overlay images

Prerequisites
-------------

Create and activate a Python virtual environment and install the minimal dependencies listed in `experiment-1/requirements-phase1.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r experiment-1/requirements-phase1.txt
```

Note: `torch` and `torchvision` are listed as optional in the file; install appropriate CUDA or CPU builds for your system if you plan to run GPU models.

Run the pipeline
----------------

The CLI script is `experiment-1/scripts/run_phase1.py`. Examples:

- Run the dummy segmentation model over a directory of images, save overlays and graphs:

```bash
cd experiment-1/scripts
python run_phase1.py --images ../images/ --model dummy --output output/ --save-overlay --save-graph
```

- List available segmentation models registered in the pipeline:

```bash
python run_phase1.py --images ../images/ --list-models
```

- Pass model-specific parameters (JSON string or path to JSON file). For example, the `dummy` model accepts `num_masks`:

```bash
python run_phase1.py --images ../images/ --model dummy --model-params '{"num_masks": 5}' --save-overlay --save-graph
```

Outputs
-------

By default outputs are written to the `--output` directory. When `--save-graph` and/or `--save-overlay` are used the script will write files named after each image:

- `<image_stem>.graph.json` — serialized `ParseGraph` (JSON)
- `<image_stem>.overlay.png` — overlay of all discovered masks on the original image

Example output:

```
experiment-1/scripts/output/image.graph.json
experiment-1/scripts/output/image.overlay.png
```

Inspecting Results
------------------

- `*.graph.json` contains nodes and edges with node IDs referring to parts; you can load this JSON into Python and convert it back to the `ParseGraph` using `ParseGraph.from_json()` from `hierarchical_pipeline.core.data`.
- `*.overlay.png` is a visualization of discovered parts overlaid on the image.

Next steps
----------

This Phase 1 CLI is intended as a starting point. Common next improvements:

- Add embedding generation (DINO/CLIP) and store embeddings in `Part.embedding`
- Add more robust mask filtering and selection policies for SAM/Mask2Former
- Add a small notebook to visualize parse graphs interactively

If you want, I can add a README example that demonstrates loading a saved graph and plotting it using `networkx` and the `visualization.plot_parse_tree` helper. Tell me and I'll add it.
