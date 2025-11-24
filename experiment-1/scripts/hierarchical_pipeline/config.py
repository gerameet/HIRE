"""Simple YAML configuration system for hierarchical pipeline.

This module provides a lightweight configuration system without heavy dependencies,
using only Python's built-in YAML support or falling back to JSON.
"""

import os
import json
import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# Try to import yaml, fall back to JSON if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logger.warning("PyYAML not installed. Using JSON for config files.")


@dataclass
class PartDiscoveryConfig:
    """Configuration for part discovery methods."""
    method: str = "slot_attention"  # "slot_attention", "sam", "coca", "yolo"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    method: str = "dino"  # "dino", "clip", "mae", "moco"
    model_name: str = "facebook/dino-vitb16"
    embedding_dim: int = 768
    use_hyperbolic: bool = False
    hyperbolic_model: str = "poincare"  # "poincare" or "lorentz"


@dataclass
class HierarchyConfig:
    """Configuration for hierarchy construction."""
    method: str = "bottom_up"  # "bottom_up", "top_down", "hybrid", "gnn"
    params: Dict[str, Any] = field(default_factory=lambda: {
        "spatial_threshold": 0.3,
        "containment_threshold": 0.7,
        "max_depth": 5,
    })


@dataclass
class KnowledgeConfig:
    """Configuration for knowledge graph integration."""
    use_knowledge_graph: bool = False
    sources: list = field(default_factory=lambda: ["wordnet"])
    alignment_method: str = "clip_similarity"
    alignment_threshold: float = 0.5


@dataclass
class OutputConfig:
    """Configuration for output and saving."""
    save_parse_graphs: bool = True
    save_embeddings: bool = True
    save_visualizations: bool = True
    output_dir: str = "output/hierarchical"


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    plot_parse_tree: bool = True
    plot_spatial_overlay: bool = True
    plot_embedding_space: bool = True
    plot_attention_maps: bool = False


@dataclass
class GPUConfig:
    """Configuration for GPU usage."""
    device: Optional[str] = None  # None for auto-detect, "cuda", "cuda:0", "cpu"
    allow_cpu_fallback: bool = True
    use_mixed_precision: bool = False
    batch_size: int = 8


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    name: str = "hierarchical_visual_pipeline"
    part_discovery: PartDiscoveryConfig = field(default_factory=PartDiscoveryConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Dictionary representation
        """
        return asdict(self)

    def save(self, path: str) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save config (supports .yaml, .yml, .json)
        """
        config_dict = self.to_dict()
        
        # Determine format from extension
        ext = os.path.splitext(path)[1].lower()
        
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        if ext in [".yaml", ".yml"] and HAS_YAML:
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved config to {path}")
        else:
            # Fall back to JSON
            if ext in [".yaml", ".yml"]:
                logger.warning(f"YAML not available, saving as JSON instead")
                path = path.replace(".yaml", ".json").replace(".yml", ".json")
            
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Saved config to {path}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """Create config from dictionary.
        
        Args:
            config_dict: Dictionary with config values
            
        Returns:
            PipelineConfig instance
        """
        # Extract nested configs
        part_discovery = PartDiscoveryConfig(**config_dict.get("part_discovery", {}))
        embedding = EmbeddingConfig(**config_dict.get("embedding", {}))
        hierarchy = HierarchyConfig(**config_dict.get("hierarchy", {}))
        knowledge = KnowledgeConfig(**config_dict.get("knowledge", {}))
        output = OutputConfig(**config_dict.get("output", {}))
        visualization = VisualizationConfig(**config_dict.get("visualization", {}))
        gpu = GPUConfig(**config_dict.get("gpu", {}))
        
        return cls(
            name=config_dict.get("name", "hierarchical_visual_pipeline"),
            part_discovery=part_discovery,
            embedding=embedding,
            hierarchy=hierarchy,
            knowledge=knowledge,
            output=output,
            visualization=visualization,
            gpu=gpu,
        )


def load_config(path: str) -> PipelineConfig:
    """Load configuration from file.
    
    Args:
        path: Path to config file (.yaml, .yml, or .json)
        
    Returns:
        PipelineConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    ext = os.path.splitext(path)[1].lower()
    
    try:
        if ext in [".yaml", ".yml"]:
            if not HAS_YAML:
                raise ValueError(
                    "PyYAML not installed. Please install it or use JSON config."
                )
            with open(path, "r") as f:
                config_dict = yaml.safe_load(f)
        elif ext == ".json":
            with open(path, "r") as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {ext}")
        
        logger.info(f"Loaded config from {path}")
        return PipelineConfig.from_dict(config_dict)
        
    except Exception as e:
        raise ValueError(f"Failed to load config from {path}: {e}")


def create_default_config(save_path: Optional[str] = None) -> PipelineConfig:
    """Create a default configuration.
    
    Args:
        save_path: Optional path to save the default config
        
    Returns:
        Default PipelineConfig instance
    """
    config = PipelineConfig()
    
    if save_path:
        config.save(save_path)
        logger.info(f"Created default config at {save_path}")
    
    return config


def validate_config(config: PipelineConfig) -> bool:
    """Validate configuration values.
    
    Args:
        config: PipelineConfig to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If config is invalid
    """
    # Validate part discovery method
    valid_discovery_methods = ["slot_attention", "sam", "coca", "yolo", "existing"]
    if config.part_discovery.method not in valid_discovery_methods:
        raise ValueError(
            f"Invalid part discovery method: {config.part_discovery.method}. "
            f"Must be one of {valid_discovery_methods}"
        )
    
    # Validate embedding method
    valid_embedding_methods = ["dino", "clip", "mae", "moco"]
    if config.embedding.method not in valid_embedding_methods:
        raise ValueError(
            f"Invalid embedding method: {config.embedding.method}. "
            f"Must be one of {valid_embedding_methods}"
        )
    
    # Validate hierarchy method
    valid_hierarchy_methods = ["bottom_up", "top_down", "hybrid", "gnn"]
    if config.hierarchy.method not in valid_hierarchy_methods:
        raise ValueError(
            f"Invalid hierarchy method: {config.hierarchy.method}. "
            f"Must be one of {valid_hierarchy_methods}"
        )
    
    # Validate hyperbolic model
    if config.embedding.use_hyperbolic:
        valid_hyperbolic = ["poincare", "lorentz"]
        if config.embedding.hyperbolic_model not in valid_hyperbolic:
            raise ValueError(
                f"Invalid hyperbolic model: {config.embedding.hyperbolic_model}. "
                f"Must be one of {valid_hyperbolic}"
            )
    
    # Validate GPU device
    if config.gpu.device is not None:
        valid_devices = ["cpu", "cuda"]
        device_prefix = config.gpu.device.split(":")[0]
        if device_prefix not in valid_devices:
            raise ValueError(
                f"Invalid device: {config.gpu.device}. "
                f"Must start with 'cpu' or 'cuda'"
            )
    
    # Validate thresholds
    if not 0 <= config.knowledge.alignment_threshold <= 1:
        raise ValueError(
            f"alignment_threshold must be between 0 and 1, "
            f"got {config.knowledge.alignment_threshold}"
        )
    
    logger.info("Configuration validated successfully")
    return True


# Example usage and default config template
DEFAULT_CONFIG_YAML = """
# Hierarchical Visual Pipeline Configuration

pipeline:
  name: "hierarchical_visual_pipeline"

part_discovery:
  method: "slot_attention"  # Options: slot_attention, sam, coca, yolo, existing
  params:
    num_slots: 7
    slot_dim: 64
    num_iterations: 3

embedding:
  method: "dino"  # Options: dino, clip, mae, moco
  model_name: "facebook/dino-vitb16"
  embedding_dim: 768
  use_hyperbolic: false
  hyperbolic_model: "poincare"  # Options: poincare, lorentz

hierarchy:
  method: "bottom_up"  # Options: bottom_up, top_down, hybrid, gnn
  params:
    spatial_threshold: 0.3
    containment_threshold: 0.7
    max_depth: 5

knowledge:
  use_knowledge_graph: false
  sources:
    - "wordnet"
  alignment_method: "clip_similarity"
  alignment_threshold: 0.5

output:
  save_parse_graphs: true
  save_embeddings: true
  save_visualizations: true
  output_dir: "output/hierarchical"

visualization:
  plot_parse_tree: true
  plot_spatial_overlay: true
  plot_embedding_space: true
  plot_attention_maps: false

gpu:
  device: null  # null for auto-detect, or "cuda", "cuda:0", "cpu"
  allow_cpu_fallback: true
  use_mixed_precision: false
  batch_size: 8
"""
