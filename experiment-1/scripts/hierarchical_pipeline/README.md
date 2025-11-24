# Hierarchical Visual Pipeline

A modular research pipeline for exploring hierarchical visual representations and world models through compositional grammars, object-centric learning, and structured embeddings.

## Project Structure

```
hierarchical_pipeline/
├── __init__.py              # Package initialization
├── README.md                # This file
├── config.py                # Configuration system
├── configs/                 # Configuration files
│   └── default.yaml         # Default configuration
├── core/                    # Core components
│   ├── __init__.py
│   ├── interfaces.py        # Abstract interfaces
│   └── data.py              # Data structures (Part, Node, ParseGraph)
└── utils/                   # Utility modules
    ├── __init__.py
    └── gpu.py               # GPU management utilities
```

## Core Components

### Abstract Interfaces

The pipeline is built around three main abstract interfaces that enable modular experimentation:

1. **PartDiscoveryMethod**: Discovers fine-grained visual parts
   - Implementations: Slot Attention, SAM, COCA, existing segmentation models
   
2. **EmbeddingMethod**: Generates semantic embeddings for parts
   - Implementations: DINO, CLIP, MAE, MoCo
   
3. **HierarchyBuilder**: Constructs compositional parse graphs
   - Implementations: Bottom-up clustering, top-down parsing, hybrid approaches

### Data Structures

- **Part**: A discovered visual part with mask, bbox, and embedding
- **Node**: A node in the parse graph representing a part at a hierarchy level
- **Edge**: A hierarchical relationship between nodes
- **ParseGraph**: Complete hierarchical representation as a DAG

## GPU Support

The pipeline includes comprehensive GPU utilities:

- **GPUManager**: Centralized GPU device management
  - Automatic CUDA detection
  - Memory monitoring and logging
  - Graceful CPU fallback

- **handle_oom**: Decorator for automatic OOM error handling
  - Cache clearing and retry
  - CPU fallback
  - Custom fallback functions

- **AutoBatchProcessor**: Automatic batch size adjustment
  - Dynamically reduces batch size on OOM
  - Finds optimal batch size for available memory

### Example GPU Usage

```python
from hierarchical_pipeline.utils import GPUManager, get_device, handle_oom

# Simple device selection
device = get_device()  # Auto-detect best device

# Full GPU management
gpu_manager = GPUManager(device="cuda", allow_cpu_fallback=True)
print(f"Using device: {gpu_manager.device}")
gpu_manager.log_memory_usage("Before processing: ")

# OOM handling decorator
@handle_oom(retry_on_cpu=True, clear_cache=True)
def process_batch(data, device):
    return model(data.to(device))
```

## Configuration System

The pipeline uses a simple YAML-based configuration system (falls back to JSON if PyYAML not available).

### Loading Configuration

```python
from hierarchical_pipeline import load_config, create_default_config

# Load from file
config = load_config("configs/default.yaml")

# Create default config
config = create_default_config(save_path="my_config.yaml")

# Access config values
print(config.part_discovery.method)
print(config.embedding.model_name)
print(config.gpu.device)
```

### Configuration Structure

```yaml
part_discovery:
  method: "slot_attention"
  params:
    num_slots: 7

embedding:
  method: "dino"
  model_name: "facebook/dino-vitb16"
  embedding_dim: 768

hierarchy:
  method: "bottom_up"
  params:
    spatial_threshold: 0.3

gpu:
  device: null  # Auto-detect
  allow_cpu_fallback: true
  batch_size: 8
```

## Quick Start

```python
from hierarchical_pipeline import (
    PartDiscoveryMethod,
    EmbeddingMethod,
    HierarchyBuilder,
    load_config,
    GPUManager
)

# Load configuration
config = load_config("configs/default.yaml")

# Initialize GPU
gpu_manager = GPUManager(device=config.gpu.device)

# Components will be implemented in subsequent phases:
# - Part discovery methods (Phase 1)
# - Embedding methods (Phase 2)
# - Hierarchy builders (Phase 5)
```

## Requirements

### Core Dependencies
- `torch` / `torchvision`: Deep learning and GPU support
- `numpy`: Numerical computing
- `Pillow`: Image processing

### Optional Dependencies
- `PyYAML`: YAML config support (falls back to JSON)
- Model-specific dependencies will be added in later phases

## Development Philosophy

This is a research project prioritizing:
- **Modularity**: Easy component swapping for experimentation
- **GPU Optimization**: Efficient GPU usage with automatic memory management
- **Interpretability**: Clear code with extensive logging
- **Reproducibility**: Deterministic results with configuration tracking

## Next Steps

This is Phase 1 (Foundation). Upcoming phases will add:
- Phase 2: Self-supervised embeddings (DINO, CLIP)
- Phase 3: Object-centric discovery (Slot Attention, SAM)
- Phase 4: Hyperbolic embeddings and knowledge integration
- Phase 5: Advanced hierarchy construction
- Phase 6: Analysis, metrics, and documentation

## License

Research project - see main repository for license information.
