"""Tests for hyperbolic projection and distance computation."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hierarchical_pipeline.embedding.hyperbolic import (
    HyperbolicProjection,
    HyperbolicDistance,
    verify_hyperbolic_constraints,
)


class TestPoincaréProjection:
    """Tests for Poincaré ball projection."""

    def test_initialization(self):
        """Test projection initialization."""
        proj = HyperbolicProjection(model="poincare", curvature=1.0)
        assert proj.model == "poincare"
        assert proj.curvature == 1.0

    def test_projection_constraint(self):
        """Test that projected embeddings satisfy norm < 1."""
        proj = HyperbolicProjection(model="poincare")

        # Test various Euclidean embeddings
        for _ in range(10):
            emb = np.random.randn(768)
            hyp_emb = proj.project(emb)

            # Check norm < 1
            norm = np.linalg.norm(hyp_emb)
            assert norm < 1.0, f"Norm {norm} >= 1.0"

    def test_zero_vector(self):
        """Test that zero vector stays at origin."""
        proj = HyperbolicProjection(model="poincare")

        zero = np.zeros(768)
        result = proj.project(zero)

        assert np.allclose(result, 0.0, atol=1e-6)

    def test_batch_projection(self):
        """Test batch projection."""
        proj = HyperbolicProjection(model="poincare")

        embeddings = np.random.randn(5, 768)
        hyp_embeddings = proj.project_batch(embeddings)

        # Check shape
        assert hyp_embeddings.shape == (5, 768)

        # Check all norms < 1
        norms = np.linalg.norm(hyp_embeddings, axis=1)
        assert np.all(norms < 1.0)


class TestLorentzProjection:
    """Tests for Lorentz model projection."""

    def test_projection_dimension(self):
        """Test that Lorentz adds time dimension."""
        proj = HyperbolicProjection(model="lorentz")

        emb = np.random.randn(768)
        hyp_emb = proj.project(emb)

        # Should have d+1 dimensions
        assert hyp_emb.shape == (769,)

    def test_lorentz_constraint(self):
        """Test Lorentz inner product constraint."""
        proj = HyperbolicProjection(model="lorentz", curvature=1.0)

        for _ in range(10):
            emb = np.random.randn(768)
            hyp_emb = proj.project(emb)

            # Check ⟨x,x⟩_L = -1 (use 1e-3 tolerance for float32)
            lorentz_norm = -hyp_emb[0] ** 2 + np.sum(hyp_emb[1:] ** 2)
            assert abs(lorentz_norm + 1.0) < 1e-3, f"Lorentz norm {lorentz_norm}"


class TestHyperbolicDistance:
    """Tests for hyperbolic distance computation."""

    def test_poincare_distance_symmetry(self):
        """Test that distance is symmetric."""
        proj = HyperbolicProjection(model="poincare")
        dist_calc = HyperbolicDistance(model="poincare")

        x = proj.project(np.random.randn(768))
        y = proj.project(np.random.randn(768))

        d_xy = dist_calc.distance(x, y)
        d_yx = dist_calc.distance(y, x)

        assert abs(d_xy - d_yx) < 1e-6

    def test_poincare_distance_positive(self):
        """Test that distances are non-negative."""
        proj = HyperbolicProjection(model="poincare")
        dist_calc = HyperbolicDistance(model="poincare")

        for _ in range(10):
            x = proj.project(np.random.randn(768))
            y = proj.project(np.random.randn(768))

            d = dist_calc.distance(x, y)
            assert d >= 0.0

    def test_poincare_distance_zero(self):
        """Test that distance to self is zero."""
        proj = HyperbolicProjection(model="poincare")
        dist_calc = HyperbolicDistance(model="poincare")

        x = proj.project(np.random.randn(768))
        d = dist_calc.distance(x, x)

        assert d < 1e-5

    def test_pairwise_distances_shape(self):
        """Test pairwise distance matrix shape."""
        proj = HyperbolicProjection(model="poincare")
        dist_calc = HyperbolicDistance(model="poincare")

        embeddings = np.random.randn(5, 768)
        hyp_embeddings = proj.project_batch(embeddings)

        distances = dist_calc.pairwise_distances(hyp_embeddings)

        assert distances.shape == (5, 5)
        assert np.allclose(distances, distances.T)  # Symmetric
        assert np.allclose(np.diag(distances), 0.0, atol=1e-5)  # Zero diagonal


class TestConstraintVerification:
    """Tests for constraint verification."""

    def test_valid_poincare_embeddings(self):
        """Test verification passes for valid Poincaré embeddings."""
        proj = HyperbolicProjection(model="poincare")

        embeddings = np.random.randn(10, 768)
        hyp_embeddings = proj.project_batch(embeddings)

        assert verify_hyperbolic_constraints(hyp_embeddings, model="poincare")

    def test_invalid_poincare_embeddings(self):
        """Test verification fails for invalid embeddings (norm >= 1)."""
        # Create embeddings with norm >= 1
        embeddings = np.random.randn(10, 768)
        embeddings = (
            embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) * 1.1
        )

        assert not verify_hyperbolic_constraints(embeddings, model="poincare")

    def test_valid_lorentz_embeddings(self):
        """Test verification passes for valid Lorentz embeddings."""
        proj = HyperbolicProjection(model="lorentz")

        embeddings = np.random.randn(10, 768)
        hyp_embeddings = proj.project_batch(embeddings)

        assert verify_hyperbolic_constraints(hyp_embeddings, model="lorentz")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
