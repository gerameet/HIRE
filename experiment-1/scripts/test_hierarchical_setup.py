"""Test script to verify hierarchical pipeline setup.

This script tests:
1. Module imports
2. GPU detection and management
3. Configuration loading
4. Data structure creation
"""

import sys
import numpy as np

print("=" * 60)
print("Testing Hierarchical Pipeline Setup")
print("=" * 60)

# Test 1: Module imports
print("\n[1/5] Testing module imports...")
try:
    from hierarchical_pipeline import (
        PartDiscoveryMethod,
        EmbeddingMethod,
        HierarchyBuilder,
        Part,
        Node,
        Edge,
        ParseGraph,
        GPUManager,
        get_device,
        handle_oom,
        load_config,
        create_default_config,
        PipelineConfig,
    )
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: GPU detection
print("\n[2/5] Testing GPU detection...")
try:
    import torch
    device = get_device()
    print(f"✓ Device detected: {device}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  CUDA device name: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ GPU detection failed: {e}")

# Test 3: GPU Manager
print("\n[3/5] Testing GPU Manager...")
try:
    gpu_manager = GPUManager(allow_cpu_fallback=True)
    print(f"✓ GPU Manager initialized")
    print(f"  Using device: {gpu_manager.device}")
    print(f"  GPU available: {gpu_manager.is_gpu_available()}")
    
    if gpu_manager.is_gpu_available():
        mem_info = gpu_manager.get_memory_info()
        print(f"  GPU memory: {mem_info['allocated_mb']:.1f}MB allocated, "
              f"{mem_info['free_mb']:.1f}MB free")
except Exception as e:
    print(f"✗ GPU Manager failed: {e}")

# Test 4: Configuration
print("\n[4/5] Testing configuration system...")
try:
    # Create default config
    config = create_default_config()
    print(f"✓ Default config created")
    print(f"  Part discovery method: {config.part_discovery.method}")
    print(f"  Embedding method: {config.embedding.method}")
    print(f"  Hierarchy method: {config.hierarchy.method}")
    print(f"  GPU device: {config.gpu.device or 'auto-detect'}")
    
    # Test config validation
    from hierarchical_pipeline.config import validate_config
    validate_config(config)
    print(f"✓ Config validation passed")
    
    # Test config save/load
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "test_config.json")
        config.save(config_path)
        loaded_config = load_config(config_path)
        print(f"✓ Config save/load successful")
        
except Exception as e:
    print(f"✗ Configuration test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Data structures
print("\n[5/5] Testing data structures...")
try:
    # Create a Part
    mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
    part = Part(
        id="part_0",
        mask=mask,
        bbox=(10, 10, 50, 50),
        confidence=0.95,
    )
    print(f"✓ Part created (area: {part.get_area()} pixels)")
    
    # Create a Node
    node = Node(
        id="node_0",
        part=part,
        level=0,
        concept="object",
        concept_confidence=0.8,
    )
    print(f"✓ Node created (level: {node.level})")
    
    # Create a ParseGraph
    graph = ParseGraph(
        image_path="test.jpg",
        image_size=(640, 480),
    )
    graph.add_node(node)
    print(f"✓ ParseGraph created ({len(graph.nodes)} nodes)")
    
    # Test serialization
    json_str = graph.to_json()
    reconstructed = ParseGraph.from_json(json_str)
    print(f"✓ Serialization round-trip successful")
    
    # Test summary
    summary = graph.get_summary()
    print(f"✓ Graph summary: {summary}")
    
except Exception as e:
    print(f"✗ Data structure test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Setup verification complete!")
print("=" * 60)
print("\nNext steps:")
print("  - Implement part discovery methods (Phase 1, Task 2-3)")
print("  - Add embedding generation (Phase 2)")
print("  - Build hierarchy construction (Phase 5)")
