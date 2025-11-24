# Design Document

## Overview

This design outlines a modular research pipeline for exploring hierarchical visual representations and world models. The system builds on the existing segmentation infrastructure to investigate how compositional grammars, object-centric learning, and structured embeddings can create interpretable hierarchical representations of visual scenes.

The pipeline follows a multi-stage architecture:

1. **Part Discovery**: Extract fine-grained visual parts using object-centric models
2. **Embedding Generation**: Encode parts using self-supervised vision models
3. **Hierarchy Construction**: Build compositional parse graphs from discovered parts
4. **Knowledge Integration**: Align visual hierarchies with external knowledge graphs
5. **Analysis & Visualization**: Evaluate and interpret learned structures

The design prioritizes modularity and experimentation, allowing researchers to easily swap components and compare different approaches.

## Architecture

### High-Level Pipeline Flow

```
Input Image
    ↓
┌─────────────────────────────────────────────────────┐
│ Stage 1: Part Discovery                             │
│ - Existing Segmentation (YOLO/SAM/Mask2Former)     │
│ - Object-Centric Models (Slot Attention/COCA)      │
│ - Foundation Models (SAM with auto-mask-gen)       │
└─────────────────────────────────────────────────────┘
    ↓ [Parts/Segments with masks]
┌─────────────────────────────────────────────────────┐
│ Stage 2: Embedding Generation                       │
│ - Self-Supervised Encoders (DINO/MAE/MoCo)        │
│ - Vision-Language Models (CLIP)                    │
│ - Hyperbolic Embedding Projection                  │
└─────────────────────────────────────────────────────┘
    ↓ [Parts with embeddings]
┌─────────────────────────────────────────────────────┐
│ Stage 3: Hierarchy Construction                     │
│ - Spatial Relationship Extraction                  │
│ - Bottom-Up Clustering                             │
│ - Top-Down Grammar Parsing                         │
│ - Hybrid Approaches                                │
└─────────────────────────────────────────────────────┘
    ↓ [Parse Graph]
┌─────────────────────────────────────────────────────┐
│ Stage 4: Knowledge Integration                      │
│ - Load External Ontologies (WordNet/ConceptNet)   │
│ - Concept Alignment via Embedding Similarity       │
│ - Semantic Label Propagation                       │
└─────────────────────────────────────────────────────┘
    ↓ [Enriched Parse Graph]
┌─────────────────────────────────────────────────────┐
│ Stage 5: Analysis & Visualization                   │
│ - Hierarchy Metrics                                │
│ - Embedding Quality Analysis                       │
│ - Interactive Visualizations                       │
└─────────────────────────────────────────────────────┘
    ↓
Output: Hierarchical World Model
```

### Component Architecture

The system uses a plugin-based architecture where each stage has:
- **Abstract Interface**: Defines the contract for that stage
- **Multiple Implementations**: Different algorithms/models implementing the interface
- **Registry System**: Dynamic registration and selection of implementations
- **Configuration**: YAML/JSON configs specifying which implementations to use

```python
# Example structure
class PartDiscoveryMethod(ABC):
    @abstractmethod
    def discover_parts(self, image: Image) -> List[Part]:
        pass

class SlotAttentionDiscovery(PartDiscoveryMethod):
    # Implementation using Slot Attention
    pass

class SAMDiscovery(PartDiscoveryMethod):
    # Implementation using Segment Anything
    pass

# Registry
PART_DISCOVERY_METHODS = {
    "slot_attention": SlotAttentionDiscovery,
    "sam": SAMDiscovery,
    # ... more methods
}
```

## Components and Interfaces

### 1. Part Discovery Module

**Purpose**: Extract fine-grained visual parts from images without manual labels.

**Interface**:
```python
class PartDiscoveryMethod(ABC):
    def discover_parts(self, image: np.ndarray) -> List[Part]:
        """Discover parts in an image.
        
        Returns:
            List of Part objects containing:
            - mask: binary mask
            - bbox: bounding box
            - features: raw features (optional)
            - confidence: discovery confidence
        """
        pass
```

**Implementations to Explore**:

1. **Existing Segmentation Models** (Baseline)
   - Reuse YOLO, Mask2Former, SAM from existing pipeline
   - Provides labeled segments as starting point
   - Fast, well-tested, but requires predefined categories

2. **Slot Attention** (Unsupervised)
   - Iterative attention mechanism to bind slots to objects
   - Learns to segment without labels
   - Configurable number of slots
   - Research question: How many slots are optimal? Fixed vs adaptive?

3. **COCA-Style Clustering** (Multi-scale)
   - Hierarchical clustering with attention
   - Bottom-up part discovery
   - Naturally produces multi-scale representations
   - Research question: How to set clustering thresholds?

4. **SAM with Auto-Mask Generation** (Foundation Model)
   - Segment Anything with automatic mask proposals
   - Domain-agnostic, zero-shot
   - Can generate hierarchical masks at different granularities
   - Research question: How to filter/select relevant masks?

**Key Design Decisions**:
- Support both supervised (existing models) and unsupervised (Slot Attention) methods
- Allow hybrid approaches (e.g., SAM proposals + Slot Attention refinement)
- Store raw features for downstream embedding generation

### 2. Embedding Generation Module

**Purpose**: Encode visual parts as dense vectors capturing semantic content.

**Interface**:
```python
class EmbeddingMethod(ABC):
    def embed_part(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Generate embedding for a masked region.
        
        Args:
            image: Full image
            mask: Binary mask for the part
            
        Returns:
            Embedding vector (normalized)
        """
        pass
    
    def embed_batch(self, images: List[np.ndarray], masks: List[np.ndarray]) -> np.ndarray:
        """Batch embedding for efficiency."""
        pass
```

**Implementations to Explore**:

1. **DINO (Self-Supervised ViT)**
   - Pre-trained on ImageNet without labels
   - Produces semantically meaningful patch embeddings
   - Attention maps align with object boundaries
   - Research question: Which layer features work best?

2. **MAE (Masked Autoencoder)**
   - Self-supervised through reconstruction
   - Good for fine-grained features
   - Research question: Pre-trained vs fine-tuned on domain?

3. **CLIP (Vision-Language)**
   - Aligned with text embeddings
   - Zero-shot concept recognition
   - Can match parts to text descriptions
   - Research question: How to generate text prompts for parts?

4. **MoCo/SimCLR (Contrastive Learning)**
   - Strong general-purpose features
   - Good for similarity comparisons
   - Research question: Domain-specific contrastive learning?

**Hyperbolic Projection**:
After generating Euclidean embeddings, optionally project to hyperbolic space:

```python
class HyperbolicProjection:
    def __init__(self, model: str = "poincare"):
        # "poincare" or "lorentz"
        self.model = model
    
    def project(self, euclidean_emb: np.ndarray) -> np.ndarray:
        """Project Euclidean embedding to hyperbolic space."""
        pass
    
    def distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute hyperbolic distance."""
        pass
```

**Key Design Decisions**:
- Support multiple SSL encoders for comparison
- Implement both Euclidean and hyperbolic embeddings
- Cache embeddings to avoid recomputation
- Normalize embeddings for consistent similarity computation

### 3. Hierarchy Construction Module

**Purpose**: Build compositional parse graphs from discovered parts.

**Interface**:
```python
class HierarchyBuilder(ABC):
    def build_hierarchy(self, parts: List[Part]) -> ParseGraph:
        """Construct hierarchical parse graph from parts.
        
        Returns:
            ParseGraph with nodes (parts) and edges (relationships)
        """
        pass
```

**Implementations to Explore**:

1. **Bottom-Up Spatial Clustering**
   - Group parts based on spatial proximity
   - Use containment, overlap, adjacency metrics
   - Agglomerative clustering to form hierarchy
   - Research question: Which spatial features matter most?

2. **Top-Down Grammar Parsing**
   - Define compositional rules (e.g., "face = eyes + nose + mouth")
   - Parse image by matching rules
   - Requires predefined grammar
   - Research question: Can we learn grammar from data?

3. **Hybrid: Bottom-Up + Top-Down**
   - Start with bottom-up clustering
   - Refine with top-down grammar constraints
   - Best of both worlds
   - Research question: How to balance data-driven vs rule-driven?

4. **Graph Neural Network**
   - Learn hierarchy construction end-to-end
   - Message passing between parts
   - Predict parent-child edges
   - Research question: What supervision signal to use?

**Spatial Relationship Extraction**:
```python
class SpatialRelationships:
    @staticmethod
    def compute_containment(part1: Part, part2: Part) -> float:
        """Compute containment score (0-1)."""
        intersection = np.logical_and(part1.mask, part2.mask).sum()
        area1 = part1.mask.sum()
        return intersection / area1 if area1 > 0 else 0.0
    
    @staticmethod
    def compute_adjacency(part1: Part, part2: Part, threshold: int = 5) -> bool:
        """Check if parts are adjacent within threshold pixels."""
        # Dilate masks and check overlap
        pass
    
    @staticmethod
    def compute_overlap(part1: Part, part2: Part) -> float:
        """Compute IoU overlap."""
        pass
```

**Parse Graph Structure**:
```python
class ParseGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}  # node_id -> Node
        self.edges: List[Edge] = []  # parent-child relationships
        self.root: Optional[str] = None  # root node (scene level)
    
    def add_node(self, node: Node):
        """Add a node (part) to the graph."""
        pass
    
    def add_edge(self, parent_id: str, child_id: str, relation_type: str):
        """Add hierarchical edge."""
        pass
    
    def get_level(self, level: int) -> List[Node]:
        """Get all nodes at a specific hierarchy level."""
        pass
    
    def traverse_dfs(self) -> Iterator[Node]:
        """Depth-first traversal."""
        pass
```

**Key Design Decisions**:
- Support multiple hierarchy construction strategies
- Store both spatial and semantic relationships
- Allow cyclic graphs for complex scenes (not just trees)
- Implement efficient graph traversal and querying

### 4. Knowledge Integration Module

**Purpose**: Align visual hierarchies with external knowledge graphs.

**Interface**:
```python
class KnowledgeIntegrator(ABC):
    def load_knowledge_graph(self, source: str):
        """Load external ontology (WordNet, ConceptNet)."""
        pass
    
    def align_concepts(self, part: Part, candidates: List[str]) -> str:
        """Align visual part to knowledge graph concept."""
        pass
    
    def propagate_labels(self, parse_graph: ParseGraph) -> ParseGraph:
        """Propagate semantic labels through hierarchy."""
        pass
```

**Implementations to Explore**:

1. **WordNet Integration**
   - Load WordNet synsets and hypernym relationships
   - Use WordNet glosses as text descriptions
   - Align parts to synsets via CLIP similarity
   - Research question: How to handle polysemy?

2. **ConceptNet Integration**
   - Load ConceptNet common-sense relationships
   - Use "PartOf", "IsA", "UsedFor" relations
   - Guide hierarchy construction with prior knowledge
   - Research question: How to weight visual vs knowledge evidence?

3. **Learned Concept Embeddings**
   - Train concept embeddings from image-text pairs
   - Learn domain-specific concept hierarchies
   - Research question: How much data needed?

**Concept Alignment**:
```python
class ConceptAligner:
    def __init__(self, clip_model, knowledge_graph):
        self.clip = clip_model
        self.kg = knowledge_graph
        self.concept_embeddings = self._embed_concepts()
    
    def _embed_concepts(self) -> Dict[str, np.ndarray]:
        """Embed all concepts in knowledge graph using CLIP text encoder."""
        concepts = {}
        for concept in self.kg.get_all_concepts():
            text = self.kg.get_description(concept)
            concepts[concept] = self.clip.encode_text(text)
        return concepts
    
    def align(self, part_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find top-k most similar concepts."""
        similarities = {}
        for concept, emb in self.concept_embeddings.items():
            sim = cosine_similarity(part_embedding, emb)
            similarities[concept] = sim
        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
```

**Key Design Decisions**:
- Support multiple knowledge sources
- Use embedding similarity for concept alignment
- Allow manual concept mappings for disambiguation
- Propagate labels hierarchically (parent concepts constrain children)

### 5. Analysis & Visualization Module

**Purpose**: Evaluate and interpret learned hierarchical representations.

**Metrics**:
```python
class HierarchyMetrics:
    @staticmethod
    def compute_depth(parse_graph: ParseGraph) -> int:
        """Maximum depth of hierarchy."""
        pass
    
    @staticmethod
    def compute_branching_factor(parse_graph: ParseGraph) -> float:
        """Average number of children per node."""
        pass
    
    @staticmethod
    def compute_balance(parse_graph: ParseGraph) -> float:
        """How balanced is the tree (0=unbalanced, 1=perfect)."""
        pass
    
    @staticmethod
    def compute_embedding_distortion(embeddings: np.ndarray, 
                                     hierarchy: ParseGraph,
                                     metric: str = "euclidean") -> float:
        """Measure how well embeddings preserve hierarchical distances."""
        pass
```

**Visualizations**:
```python
class HierarchyVisualizer:
    def plot_parse_tree(self, parse_graph: ParseGraph, output_path: str):
        """Render hierarchy as tree diagram."""
        # Use networkx + graphviz or plotly
        pass
    
    def plot_spatial_overlay(self, image: np.ndarray, 
                            parse_graph: ParseGraph,
                            output_path: str):
        """Overlay part boundaries with hierarchical colors."""
        pass
    
    def plot_embedding_space(self, embeddings: np.ndarray,
                            labels: List[str],
                            method: str = "umap",
                            output_path: str):
        """Project embeddings to 2D and visualize."""
        # Use UMAP or t-SNE
        pass
    
    def plot_attention_maps(self, image: np.ndarray,
                           attention: np.ndarray,
                           output_path: str):
        """Visualize attention weights (for Slot Attention)."""
        pass
```

**Key Design Decisions**:
- Compute both structural and embedding quality metrics
- Generate publication-quality visualizations
- Support interactive visualizations (Plotly, Bokeh)
- Export metrics to CSV/JSON for analysis

## Data Models

### Core Data Structures

```python
@dataclass
class Part:
    """A discovered visual part."""
    id: str
    mask: np.ndarray  # Binary mask (H, W)
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    features: Optional[np.ndarray]  # Raw features from discovery
    embedding: Optional[np.ndarray]  # Semantic embedding
    confidence: float  # Discovery confidence
    metadata: Dict[str, Any]  # Additional info

@dataclass
class Node:
    """A node in the parse graph."""
    id: str
    part: Part
    level: int  # Hierarchy level (0=leaf, higher=more abstract)
    concept: Optional[str]  # Aligned concept from knowledge graph
    concept_confidence: float
    children: List[str]  # Child node IDs
    parent: Optional[str]  # Parent node ID

@dataclass
class Edge:
    """An edge in the parse graph."""
    parent_id: str
    child_id: str
    relation_type: str  # "contains", "part_of", "adjacent", etc.
    confidence: float
    spatial_features: Dict[str, float]  # Containment, overlap, etc.

@dataclass
class ParseGraph:
    """Hierarchical parse graph for an image."""
    image_path: str
    image_size: Tuple[int, int]
    nodes: Dict[str, Node]
    edges: List[Edge]
    root: Optional[str]
    metadata: Dict[str, Any]
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph for analysis."""
        pass
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        pass
    
    @classmethod
    def from_json(cls, json_str: str) -> "ParseGraph":
        """Deserialize from JSON."""
        pass

@dataclass
class HierarchicalRepresentation:
    """Complete hierarchical representation for an image."""
    image_path: str
    parts: List[Part]
    parse_graph: ParseGraph
    embeddings: Dict[str, np.ndarray]  # node_id -> embedding
    embedding_space: str  # "euclidean" or "hyperbolic"
    processing_time: float
    config: Dict[str, Any]  # Pipeline configuration used
```

### Configuration Schema

```yaml
# config.yaml
pipeline:
  name: "hierarchical_visual_pipeline"
  
part_discovery:
  method: "slot_attention"  # or "sam", "coca", "yolo"
  params:
    num_slots: 7
    slot_dim: 64
    num_iterations: 3
    
embedding:
  method: "dino"  # or "clip", "mae", "moco"
  model_name: "facebook/dino-vitb16"
  embedding_dim: 768
  use_hyperbolic: true
  hyperbolic_model: "poincare"
  
hierarchy:
  method: "bottom_up"  # or "top_down", "hybrid", "gnn"
  params:
    spatial_threshold: 0.3
    containment_threshold: 0.7
    max_depth: 5
    
knowledge:
  use_knowledge_graph: true
  sources:
    - "wordnet"
    - "conceptnet"
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
  plot_attention_maps: true
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property Reflection

Before defining properties, let's identify and eliminate redundancy:

**Redundancies Identified:**
1. Properties about "supporting multiple methods" (1.3, 3.2, 8.1-8.3) can be consolidated into interface conformance properties
2. Properties about visualization (7.1-7.5) can be combined into comprehensive visualization properties
3. Properties about configuration (1.4, 8.4, 10.3) overlap and can be unified
4. Properties about serialization (9.1, 9.2) are related and can be combined into round-trip properties
5. Properties about integration (11.1-11.4) can be consolidated into compatibility properties

**Consolidated Property Set:**
- Object-centric discovery properties (1.1, 1.2)
- Parse graph construction properties (2.1, 2.2, 2.3)
- Embedding generation properties (3.1, 3.3, 3.5)
- Hyperbolic embedding properties (4.1, 4.3, 4.4)
- Knowledge integration properties (5.1, 5.2, 5.3, 5.4)
- Foundation model properties (6.1, 6.2, 6.3, 6.4)
- Visualization properties (consolidated from 7.x)
- Hierarchy construction properties (consolidated from 8.x)
- Serialization round-trip properties (consolidated from 9.1, 9.2)
- Metrics computation properties (9.3, 9.4)
- Interface conformance properties (consolidated from 10.1, 10.2)
- Integration compatibility properties (consolidated from 11.x)
- Multi-scale properties (12.1, 12.2, 12.4)

### Correctness Properties

Property 1: Slot extraction completeness
*For any* valid image, when processed through an object-centric discovery method, the system should produce a non-empty list of slots where each slot contains a valid mask and embedding
**Validates: Requirements 1.1, 1.2**

Property 2: Interface conformance
*For any* registered part discovery method, embedding method, or hierarchy builder, the implementation should conform to its abstract interface and be callable through the registry system
**Validates: Requirements 1.3, 3.2, 10.1, 10.2**

Property 3: Spatial relationship consistency
*For any* pair of discovered parts, if part A contains part B (containment > threshold), then in the parse graph, A should be an ancestor of B in the hierarchy
**Validates: Requirements 2.2, 2.3**

Property 4: Parse graph validity
*For any* constructed parse graph, the graph should be a valid directed acyclic graph (DAG) with a single root node and all nodes reachable from the root
**Validates: Requirements 2.1, 2.4**

Property 5: Embedding dimensionality consistency
*For any* set of parts processed with the same embedding method, all generated embeddings should have the same dimensionality as specified by the encoder configuration
**Validates: Requirements 3.1, 3.3**

Property 6: Hyperbolic distance preservation
*For any* hierarchical structure with known parent-child relationships, when embeddings are projected to hyperbolic space, the hyperbolic distance between parent-child pairs should be smaller than the distance between random pairs
**Validates: Requirements 4.3, 4.4**

Property 7: Hyperbolic space validity
*For any* embedding projected to hyperbolic space (Poincaré or Lorentz), the embedding should satisfy the mathematical constraints of that space (e.g., norm < 1 for Poincaré ball)
**Validates: Requirements 4.1, 4.2**

Property 8: Knowledge graph alignment
*For any* visual part with an embedding, when aligned to a knowledge graph using similarity, the top-k retrieved concepts should have similarity scores in descending order
**Validates: Requirements 5.2, 5.5**

Property 9: Label propagation consistency
*For any* parse graph with some labeled nodes, when labels are propagated through the knowledge graph hierarchy, child nodes should receive labels consistent with their parent's label in the knowledge graph
**Validates: Requirements 5.3, 5.4**

Property 10: SAM mask generation
*For any* image processed with SAM automatic mask generation, the system should produce a set of masks where each mask is a valid binary array with the same spatial dimensions as the input image
**Validates: Requirements 6.1, 6.2**

Property 11: Zero-shot labeling
*For any* segment and a set of candidate text labels, when using CLIP for zero-shot classification, the system should assign the label with the highest visual-text similarity score
**Validates: Requirements 6.3, 6.4**

Property 12: Visualization completeness
*For any* parse graph, when visualization is requested, the system should generate all specified visualization types (tree diagram, spatial overlay, embedding projection, attention maps) without errors
**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**

Property 13: Hierarchy construction configurability
*For any* valid hierarchy construction configuration (bottom-up, top-down, or hybrid), the system should successfully build a parse graph using the specified strategy
**Validates: Requirements 8.1, 8.2, 8.3, 8.4**

Property 14: Serialization round-trip
*For any* parse graph, serializing to a format (JSON, GraphML, NetworkX) and then deserializing should produce an equivalent graph structure with the same nodes, edges, and metadata
**Validates: Requirements 9.1, 9.2**

Property 15: Metrics computation
*For any* parse graph, the system should successfully compute all structural metrics (depth, branching factor, balance) and return valid numerical values
**Validates: Requirements 9.3, 9.4**

Property 16: Configuration application
*For any* valid configuration file, when loaded, the system should instantiate the specified components and apply all configuration parameters correctly
**Validates: Requirements 10.3, 10.4**

Property 17: Experiment reproducibility
*For any* experiment run with a given configuration and random seed, running the experiment again with the same configuration and seed should produce identical results (within numerical precision)
**Validates: Requirements 10.4, 10.5**

Property 18: Backward compatibility
*For any* SegmentationResult object from the existing pipeline, the hierarchical pipeline should successfully process it and produce a valid parse graph
**Validates: Requirements 11.2, 11.3, 11.4**

Property 19: Multi-scale feature extraction
*For any* image processed with multi-scale settings, the system should extract features at all specified scales and each scale should produce valid embeddings
**Validates: Requirements 12.1, 12.4**

Property 20: Scale alignment
*For any* multi-scale parse graph, parts at coarser scales should spatially contain or overlap with parts at finer scales that are their descendants in the hierarchy
**Validates: Requirements 12.2, 12.5**

## Error Handling

### Error Categories

1. **Input Validation Errors**
   - Invalid image formats or corrupted images
   - Malformed configuration files
   - Missing required parameters
   - Invalid parameter values (e.g., negative slot counts)

2. **Model Loading Errors**
   - Missing model weights
   - Incompatible model versions
   - CUDA out of memory
   - Model initialization failures

3. **Processing Errors**
   - Segmentation failures (no parts discovered)
   - Embedding generation failures
   - Graph construction failures (cyclic dependencies)
   - Knowledge graph loading failures

4. **Serialization Errors**
   - Disk write failures
   - Invalid output paths
   - Serialization format errors

### Error Handling Strategy

```python
class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass

class InputValidationError(PipelineError):
    """Raised when input validation fails."""
    pass

class ModelLoadingError(PipelineError):
    """Raised when model loading fails."""
    pass

class ProcessingError(PipelineError):
    """Raised during processing."""
    pass

class SerializationError(PipelineError):
    """Raised during serialization/deserialization."""
    pass

# Error handling in pipeline
class HierarchicalPipeline:
    def process_image(self, image_path: str) -> HierarchicalRepresentation:
        try:
            # Validate input
            if not os.path.exists(image_path):
                raise InputValidationError(f"Image not found: {image_path}")
            
            # Load image
            image = self._load_image(image_path)
            
            # Discover parts
            try:
                parts = self.part_discovery.discover_parts(image)
                if len(parts) == 0:
                    logger.warning(f"No parts discovered in {image_path}")
                    # Return empty representation rather than failing
                    return HierarchicalRepresentation.empty(image_path)
            except Exception as e:
                raise ProcessingError(f"Part discovery failed: {e}")
            
            # Generate embeddings
            try:
                embeddings = self.embedding_method.embed_batch(
                    [image] * len(parts),
                    [p.mask for p in parts]
                )
            except Exception as e:
                raise ProcessingError(f"Embedding generation failed: {e}")
            
            # Build hierarchy
            try:
                parse_graph = self.hierarchy_builder.build_hierarchy(parts)
            except Exception as e:
                raise ProcessingError(f"Hierarchy construction failed: {e}")
            
            # Integrate knowledge (optional, don't fail if unavailable)
            if self.knowledge_integrator:
                try:
                    parse_graph = self.knowledge_integrator.align_concepts(parse_graph)
                except Exception as e:
                    logger.warning(f"Knowledge integration failed: {e}")
            
            return HierarchicalRepresentation(
                image_path=image_path,
                parts=parts,
                parse_graph=parse_graph,
                embeddings=embeddings,
                # ...
            )
            
        except PipelineError:
            raise  # Re-raise pipeline errors
        except Exception as e:
            # Wrap unexpected errors
            raise ProcessingError(f"Unexpected error processing {image_path}: {e}")
```

### Graceful Degradation

- If part discovery finds no parts, return empty representation rather than failing
- If knowledge integration fails, continue without semantic labels
- If visualization fails, log warning but don't fail the pipeline
- If some embeddings fail, process remaining parts and mark failed ones

### Logging Strategy

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('hierarchical_pipeline')

# Log levels:
# DEBUG: Detailed diagnostic information
# INFO: General pipeline progress
# WARNING: Recoverable issues (e.g., no parts found, knowledge integration failed)
# ERROR: Serious issues that prevent processing
# CRITICAL: System-level failures
```

## Testing Strategy

### Unit Testing

Unit tests verify individual components in isolation:

1. **Part Discovery Tests**
   - Test each discovery method with synthetic images
   - Verify mask validity (binary, correct shape)
   - Test edge cases (empty images, single-color images)

2. **Embedding Tests**
   - Test each encoder with known inputs
   - Verify embedding dimensionality
   - Test batch processing

3. **Hierarchy Construction Tests**
   - Test spatial relationship computation
   - Test graph construction algorithms
   - Verify DAG properties

4. **Knowledge Integration Tests**
   - Test knowledge graph loading
   - Test concept alignment
   - Test label propagation

5. **Serialization Tests**
   - Test round-trip for all formats
   - Test with edge cases (empty graphs, large graphs)

### Property-Based Testing

Property-based tests verify universal properties across many random inputs:

**Testing Framework**: Use `hypothesis` for Python property-based testing

**Key Properties to Test**:

1. **Slot Extraction Completeness** (Property 1)
   - Generate random images
   - Verify non-empty slot list with valid masks

2. **Spatial Relationship Consistency** (Property 3)
   - Generate random part configurations
   - Verify containment implies hierarchy

3. **Parse Graph Validity** (Property 4)
   - Generate random parse graphs
   - Verify DAG properties

4. **Embedding Dimensionality** (Property 5)
   - Generate random parts
   - Verify consistent dimensions

5. **Hyperbolic Distance Preservation** (Property 6)
   - Generate random hierarchies
   - Verify parent-child distances < random distances

6. **Serialization Round-Trip** (Property 14)
   - Generate random parse graphs
   - Verify serialize-deserialize equivalence

**Example Property Test**:
```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst

@given(
    image=npst.arrays(
        dtype=np.uint8,
        shape=st.tuples(
            st.integers(min_value=64, max_value=512),  # height
            st.integers(min_value=64, max_value=512),  # width
            st.just(3)  # channels
        )
    )
)
def test_slot_extraction_completeness(image):
    """Property 1: Slot extraction should always produce valid results."""
    pipeline = HierarchicalPipeline(config)
    parts = pipeline.part_discovery.discover_parts(image)
    
    # Should produce at least one part (or handle gracefully)
    assert len(parts) >= 0
    
    # All parts should have valid masks
    for part in parts:
        assert part.mask.shape[:2] == image.shape[:2]
        assert part.mask.dtype == bool
        assert part.embedding is not None
        assert len(part.embedding.shape) == 1  # 1D vector

@given(
    parse_graph=st.builds(generate_random_parse_graph)
)
def test_serialization_round_trip(parse_graph):
    """Property 14: Serialization should preserve graph structure."""
    # Serialize
    json_str = parse_graph.to_json()
    
    # Deserialize
    reconstructed = ParseGraph.from_json(json_str)
    
    # Verify equivalence
    assert len(reconstructed.nodes) == len(parse_graph.nodes)
    assert len(reconstructed.edges) == len(parse_graph.edges)
    assert reconstructed.root == parse_graph.root
    
    # Verify node properties
    for node_id in parse_graph.nodes:
        assert node_id in reconstructed.nodes
        orig_node = parse_graph.nodes[node_id]
        recon_node = reconstructed.nodes[node_id]
        assert orig_node.level == recon_node.level
        assert orig_node.concept == recon_node.concept
```

### Integration Testing

Integration tests verify components work together:

1. **End-to-End Pipeline Test**
   - Process a small test dataset
   - Verify all stages complete successfully
   - Check output files are created

2. **Multi-Method Comparison Test**
   - Run same image through different methods
   - Verify all produce valid outputs
   - Compare results

3. **Knowledge Integration Test**
   - Process images with knowledge graph
   - Verify concepts are aligned
   - Check label propagation

### Evaluation Metrics

For research evaluation (not automated tests):

1. **Part Discovery Quality**
   - Segmentation accuracy (if ground truth available)
   - Part coherence (visual similarity within parts)
   - Coverage (% of image explained by parts)

2. **Hierarchy Quality**
   - Depth distribution
   - Branching factor distribution
   - Balance metrics
   - Interpretability (human evaluation)

3. **Embedding Quality**
   - Cluster coherence (silhouette score)
   - Hierarchy preservation (distortion metrics)
   - Semantic alignment (if labels available)

4. **Knowledge Integration Quality**
   - Concept alignment accuracy
   - Label propagation accuracy
   - Disambiguation success rate

### Testing Configuration

```yaml
# test_config.yaml
testing:
  unit_tests:
    enabled: true
    coverage_threshold: 0.8
    
  property_tests:
    enabled: true
    num_examples: 100  # Number of random examples per property
    max_examples: 1000  # Maximum if searching for counterexamples
    
  integration_tests:
    enabled: true
    test_dataset: "data/test_images"
    
  evaluation:
    enabled: true
    eval_dataset: "data/eval_images"
    ground_truth: "data/eval_annotations"
    metrics:
      - "segmentation_accuracy"
      - "hierarchy_depth"
      - "embedding_coherence"
```

## Implementation Notes

### Research vs Production Code

This is a research project, so the implementation should prioritize:

1. **Modularity**: Easy to swap components and try new ideas
2. **Interpretability**: Clear code with extensive logging and visualization
3. **Reproducibility**: Deterministic results with seed control
4. **Experimentation**: Support for running multiple configurations and comparing results

NOT priorities:
- Production-level optimization
- Deployment infrastructure
- Scalability to millions of images
- Real-time performance

### Recommended Libraries

**Core**:
- `torch` / `torchvision`: Deep learning
- `numpy`: Numerical computing
- `PIL` / `opencv`: Image processing

**Models**:
- `transformers`: DINO, CLIP, MAE models
- `segment-anything`: SAM
- `timm`: Vision model zoo

**Graphs**:
- `networkx`: Graph data structures and algorithms
- `python-igraph`: Fast graph operations

**Knowledge Graphs**:
- `nltk`: WordNet access
- `conceptnet5`: ConceptNet API

**Hyperbolic Geometry**:
- `geoopt`: Riemannian optimization
- `hyptorch`: Hyperbolic neural networks

**Visualization**:
- `matplotlib`: Static plots
- `plotly`: Interactive visualizations
- `graphviz`: Graph rendering
- `umap-learn`: Dimensionality reduction

**Testing**:
- `pytest`: Test framework
- `hypothesis`: Property-based testing

**Utilities**:
- `hydra`: Configuration management
- `wandb` or `tensorboard`: Experiment tracking
- `tqdm`: Progress bars

### Development Workflow

1. **Start Simple**: Begin with existing segmentation + basic clustering
2. **Add Components Incrementally**: Add one new component at a time
3. **Visualize Early**: Create visualizations to understand what's happening
4. **Compare Methods**: Run multiple methods on same images to compare
5. **Iterate**: Refine based on visual inspection and metrics

### Experiment Tracking

Use Weights & Biases or similar for tracking experiments:

```python
import wandb

# Initialize experiment
wandb.init(
    project="hierarchical-visual-representations",
    config={
        "part_discovery": "slot_attention",
        "embedding": "dino",
        "hierarchy": "bottom_up",
        # ... all config
    }
)

# Log metrics
wandb.log({
    "hierarchy_depth": depth,
    "num_parts": len(parts),
    "processing_time": time,
})

# Log visualizations
wandb.log({"parse_tree": wandb.Image(tree_image)})
```

## Future Extensions

Potential research directions to explore:

1. **Temporal Hierarchies**: Extend to video, tracking how hierarchies evolve
2. **3D Hierarchies**: Apply to 3D scenes (point clouds, meshes)
3. **Cross-Modal Hierarchies**: Integrate text, audio, other modalities
4. **Learned Grammars**: Learn compositional rules from data
5. **Interactive Parsing**: Allow human-in-the-loop refinement
6. **Causal Hierarchies**: Model causal relationships between parts
7. **Generative Models**: Generate images from hierarchical specifications
8. **Transfer Learning**: Transfer hierarchies across domains
