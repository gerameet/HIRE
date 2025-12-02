"""Hyperbolic embedding projection and distance computation.

Provides utilities for projecting Euclidean embeddings to hyperbolic space
(Poincaré ball or Lorentz model) for better hierarchical representation.

Hyperbolic space naturally represents tree-like hierarchies due to exponential
volume growth, making it ideal for hierarchical visual representations.
"""

import numpy as np
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)

# Try to import geoopt for Riemannian optimization
try:
    import torch
    import geoopt

    HAS_GEOOPT = True
except ImportError:
    HAS_GEOOPT = False
    logger.warning(
        "geoopt not available. Hyperbolic projections will use simple implementations."
    )


class HyperbolicProjection:
    """Project Euclidean embeddings to hyperbolic space.

    Supports two models:
    - Poincaré ball: {x ∈ ℝⁿ : ||x|| < 1}
    - Lorentz model: {x ∈ ℝⁿ⁺¹ : ⟨x,x⟩_L = -1, x₀ > 0}

    The projection is done via exponential map from origin.
    """

    def __init__(
        self,
        model: Literal["poincare", "lorentz"] = "poincare",
        curvature: float = 1.0,
        dim: Optional[int] = None,
    ):
        """Initialize hyperbolic projection.

        Args:
            model: Hyperbolic model ("poincare" or "lorentz")
            curvature: Curvature parameter (default 1.0)
            dim: Embedding dimension (inferred from first projection if None)
        """
        self.model = model
        self.curvature = curvature
        self.dim = dim

        logger.info(f"HyperbolicProjection initialized (model={model}, c={curvature})")

    def project(self, euclidean_emb: np.ndarray) -> np.ndarray:
        """Project Euclidean embedding to hyperbolic space.

        Args:
            euclidean_emb: Euclidean embedding vector (d,)

        Returns:
            Hyperbolic embedding
        """
        if self.model == "poincare":
            return self._project_poincare(euclidean_emb)
        elif self.model == "lorentz":
            return self._project_lorentz(euclidean_emb)
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def _project_poincare(self, emb: np.ndarray) -> np.ndarray:
        """Project to Poincaré ball via exponential map.

        Simple projection: scale embedding to have norm < 1
        For a more principled approach, use exp map from origin.

        Args:
            emb: Euclidean embedding

        Returns:
            Poin caré ball embedding (norm < 1)
        """
        # Compute current norm
        norm = np.linalg.norm(emb)

        if norm < 1e-8:
            # Zero vector stays at origin
            return emb.copy()

        # Scale to Poincaré ball (use tanh to smoothly map to unit ball)
        # This ensures output norm < 1
        scale = np.tanh(norm / np.sqrt(self.curvature))
        hyp_emb = (scale / norm) * emb

        # Double-check constraint (use 0.99 to ensure strict < 1.0 with float32)
        final_norm = np.linalg.norm(hyp_emb)
        if final_norm >= 0.99:
            # Numerical issue - clip to safely inside ball
            hyp_emb = hyp_emb * 0.98 / final_norm

        return hyp_emb.astype(np.float32)

    def _project_lorentz(self, emb: np.ndarray) -> np.ndarray:
        """Project to Lorentz model (hyperboloid).

        Lorentz model: {x ∈ ℝⁿ⁺¹ : ⟨x,x⟩_L = -1, x₀ > 0}
        where ⟨x,y⟩_L = -x₀y₀ + x₁y₁ + ... + xₙyₙ

        Args:
            emb: Euclidean embedding (d,)

        Returns:
            Lorentz embedding (d+1,)
        """
        # For Lorentz, we add a timelike dimension
        # Simple projection: x = [√(1 + ||v||²/c), v/√c]

        spatial_norm_sq = np.sum(emb**2)

        # Time component (ensures Lorentz inner product = -1)
        x0 = np.sqrt(1.0 + spatial_norm_sq / self.curvature)

        # Spatial components
        x_spatial = emb / np.sqrt(self.curvature)

        # Combine
        hyp_emb = np.concatenate([[x0], x_spatial])

        return hyp_emb.astype(np.float32)

    def project_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Project batch of embeddings.

        Args:
            embeddings: Batch of Euclidean embeddings (N, d)

        Returns:
            Batch of hyperbolic embeddings
        """
        return np.stack([self.project(emb) for emb in embeddings])


class HyperbolicDistance:
    """Compute distances in hyperbolic space.

    Provides distance metrics for Poincaré ball and Lorentz model.
    These distances respect the curvature of hyperbolic space.
    """

    def __init__(
        self, model: Literal["poincare", "lorentz"] = "poincare", curvature: float = 1.0
    ):
        """Initialize distance computer.

        Args:
            model: Hyperbolic model
            curvature: Curvature parameter
        """
        self.model = model
        self.curvature = curvature

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute hyperbolic distance between two points.

        Args:
            x, y: Hyperbolic embeddings

        Returns:
            Hyperbolic distance
        """
        if self.model == "poincare":
            return self._poincare_distance(x, y)
        elif self.model == "lorentz":
            return self._lorentz_distance(x, y)
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def _poincare_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Poincaré ball distance.

        d(x, y) = arcosh(1 + 2 * ||x - y||² / ((1 - ||x||²)(1 - ||y||²)))

        Args:
            x, y: Points in Poincaré ball

        Returns:
            Distance
        """
        # Compute norms
        norm_x_sq = np.sum(x**2)
        norm_y_sq = np.sum(y**2)

        # Ensure inside ball
        if norm_x_sq >= 1.0 or norm_y_sq >= 1.0:
            logger.warning(
                f"Point outside Poincaré ball: ||x||²={norm_x_sq}, ||y||²={norm_y_sq}"
            )
            # Clamp to valid range
            if norm_x_sq >= 1.0:
                x = x * 0.999 / np.sqrt(norm_x_sq)
                norm_x_sq = 0.999**2
            if norm_y_sq >= 1.0:
                y = y * 0.999 / np.sqrt(norm_y_sq)
                norm_y_sq = 0.999**2

        # Compute difference
        diff_norm_sq = np.sum((x - y) ** 2)

        # Poincaré distance formula
        numerator = 2 * diff_norm_sq
        denominator = (1 - norm_x_sq) * (1 - norm_y_sq)

        if denominator < 1e-8:
            # Numerical issue - points very close to boundary
            return 100.0  # Large distance

        arg = 1 + numerator / denominator

        # arcosh(x) = log(x + sqrt(x² - 1))
        # Clamp arg to valid range
        arg = max(arg, 1.0)

        dist = np.arccosh(arg) * np.sqrt(self.curvature)

        return float(dist)

    def _lorentz_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Lorentz model distance.

        d(x, y) = arcosh(-⟨x, y⟩_L / c)

        Args:
            x, y: Points in Lorentz model

        Returns:
            Distance
        """
        # Lorentz inner product: -x₀y₀ + x₁y₁ + ... + xₙyₙ
        inner_product = -x[0] * y[0] + np.sum(x[1:] * y[1:])

        # Distance formula
        arg = -inner_product / self.curvature

        # Clamp to valid range for arcosh
        arg = max(arg, 1.0)

        dist = np.arccosh(arg) * np.sqrt(self.curvature)

        return float(dist)

    def pairwise_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise distances for batch.

        Args:
            embeddings: Batch of hyperbolic embeddings (N, d)

        Returns:
            Distance matrix (N, N)
        """
        n = len(embeddings)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = self.distance(embeddings[i], embeddings[j])
                distances[i, j] = dist
                distances[j, i] = dist

        return distances


def verify_hyperbolic_constraints(
    embeddings: np.ndarray, model: Literal["poincare", "lorentz"] = "poincare"
) -> bool:
    """Verify embeddings satisfy hyperbolic space constraints.

    Args:
        embeddings: Hyperbolic embeddings (N, d)
        model: Hyperbolic model

    Returns:
        True if all constraints satisfied
    """
    if model == "poincare":
        # Check all norms < 1
        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(norms >= 1.0):
            logger.error(f"Poincaré constraint violated: max norm = {norms.max()}")
            return False
        return True

    elif model == "lorentz":
        # Check Lorentz inner product = -1 for all points
        # ⟨x,x⟩_L = -x₀² + ||x_spatial||² should be -1
        for i, x in enumerate(embeddings):
            lorentz_norm = -x[0] ** 2 + np.sum(x[1:] ** 2)
            if abs(lorentz_norm + 1.0) > 1e-3:
                logger.error(
                    f"Lorentz constraint violated at {i}: ⟨x,x⟩_L = {lorentz_norm}"
                )
                return False
        return True

    else:
        raise ValueError(f"Unknown model: {model}")
