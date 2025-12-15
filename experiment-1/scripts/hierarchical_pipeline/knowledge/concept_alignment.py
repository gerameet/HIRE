"""Alignment between visual parts and semantic concepts.

Uses CLIP to compute similarity between part embeddings and text concepts.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
import logging

from ..core.data import Part
from .wordnet import WordNetKnowledge

logger = logging.getLogger(__name__)


class CLIPConceptAligner:
    """Align visual embeddings to text concepts via CLIP."""
    
    def __init__(
        self, 
        clip_model: Any, 
        clip_processor: Any, 
        wordnet: Optional[WordNetKnowledge] = None,
        device: str = "cpu"
    ):
        """Initialize aligner.
        
        Args:
            clip_model: Loaded CLIPModel (transformers)
            clip_processor: Loaded CLIPProcessor
            wordnet: WordNet interface (optional, for concept expansion)
            device: 'cpu' or 'cuda'
        """
        self.model = clip_model
        self.processor = clip_processor
        self.wordnet = wordnet or WordNetKnowledge()
        self.device = device
        
        self.concept_embeddings = {}  # Cache: concept -> normalization embedding
        
    def _encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode text to CLIP embedding space."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            
        # Normalize
        text_features = text_features.cpu().numpy()
        text_features = text_features / (np.linalg.norm(text_features, axis=1, keepdims=True) + 1e-8)
        
        return text_features

    def precompute_concept_embeddings(self, concepts: List[str]):
        """Precompute embeddings for a list of logical concepts."""
        # Filter cached
        to_encode = [c for c in concepts if c not in self.concept_embeddings]
        
        if not to_encode:
            return
            
        # Encode batch
        # Limit batch size if needed
        batch_size = 32
        for i in range(0, len(to_encode), batch_size):
            batch = to_encode[i : i + batch_size]
            embeddings = self._encode_text(batch)
            
            for c, emb in zip(batch, embeddings):
                self.concept_embeddings[c] = emb
                
    def align_part(
        self, 
        part: Part, 
        candidates: List[str], 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Align a single part to candidate concepts.
        
        Args:
            part: Visual part with embedding (must be CLIP embedding!)
            candidates: List of text concepts
            top_k: Number of results
            
        Returns:
            List of (concept, score)
        """
        if part.embedding is None:
            return []
            
        # Ensure we have candidate embeddings
        self.precompute_concept_embeddings(candidates)
        
        # Stack candidate embeddings
        candidate_embs = np.stack([self.concept_embeddings[c] for c in candidates])
        
        # Compute cosine similarity
        # Part embedding (D,) -> (1, D)
        # Candidates (N, D)
        scores = (candidate_embs @ part.embedding).flatten()
        
        # Top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((candidates[idx], float(scores[idx])))
            
        return results
