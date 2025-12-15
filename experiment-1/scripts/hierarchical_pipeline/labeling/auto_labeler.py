"""Automatic labeling for visual parts using CLIP zero-shot."""

from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import logging

from ..embedding.methods import CLIPEmbedding
from ..core.data import Part, ParseGraph

logger = logging.getLogger(__name__)


class CLIPAutoLabeler:
    """Zero-shot labeler using CLIP."""
    
    def __init__(
        self, 
        embedder: CLIPEmbedding, 
        vocabulary: List[str],
        device: str = "cpu"
    ):
        """Initialize labeler.
        
        Args:
            embedder: Initialized CLIPEmbedding instance
            vocabulary: List of class names/concepts
            device: 'cpu' or 'cuda'
        """
        self.embedder = embedder
        self.model = embedder.model
        self.processor = embedder.processor
        self.device = device
        self.vocabulary = vocabulary
        
        self.text_features = None
        self._precompute_text_features()

    def _precompute_text_features(self):
        """Encode vocabulary into CLIP text space."""
        if not self.vocabulary:
            return
            
        logger.info(f"Encoding {len(self.vocabulary)} labels for auto-labeling...")
        
        # Batch process vocab
        batch_size = 128
        all_features = []
        
        for i in range(0, len(self.vocabulary), batch_size):
            batch = self.vocabulary[i : i + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
                
            # Normalize
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu())
            
        self.text_features = torch.cat(all_features, dim=0)

    def label_part(self, part: Part, top_k: int = 5) -> List[Tuple[str, float]]:
        """Assign labels to a single part.
        
        Args:
            part: Part with valid embedding
            top_k: Number of top labels to return
            
        Returns:
            List of (label, score) tuples
        """
        if part.embedding is None or self.text_features is None:
            return []
            
        # Part embedding (normalized)
        # Check if embedding matches CLIP dimension
        if part.embedding.shape[0] != self.text_features.shape[1]:
            # This happens if we try to label using DINO embedding with CLIP labeler
            # We strictly need CLIP embedding here.
            # Ideally the pipeline ensures this, or we re-embed?
            # For now, return empty if mismatch
            return []
            
        part_emb = torch.from_numpy(part.embedding).float().to(self.text_features.device)
        
        # Similarity
        # (vocab_size, D) @ (D,) -> (vocab_size,)
        scores = (100.0 * self.text_features @ part_emb)
        scores = scores.softmax(dim=0)
        
        # Top usage
        vals, indices = scores.topk(min(top_k, len(self.vocabulary)))
        
        results = []
        for val, idx in zip(vals, indices):
            results.append((self.vocabulary[idx.item()], val.item()))
            
        return results

    def label_graph(self, graph: ParseGraph, top_k: int = 5):
        """Label all parts in a hierarchy."""
        for node in graph.nodes.values():
            if node.part.embedding is not None:
                labels = self.label_part(node.part, top_k=top_k)
                
                # Store in part
                node.part.labels = [l for l, s in labels]
                node.part.label_scores = {l: s for l, s in labels}
                
                # Store in node too?
                # Data structure says inherited/combined on Node
                # Own labels ideally derived from part
