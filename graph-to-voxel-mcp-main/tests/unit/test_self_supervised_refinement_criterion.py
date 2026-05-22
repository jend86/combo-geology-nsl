from __future__ import annotations

import math

import numpy as np
import pytest

from graph_to_voxel.engine import VoxelField
from graph_to_voxel.refinement import (
    RefinementCriterionConfig,
    confidence_weighted_capped_forward_kl,
    coverage_ratio,
    reject_dedup,
    score_refinement,
)
from graph_to_voxel.voxel import entropy_from_probs
from tests.fixtures.toy_graphs import two_unit_horizontal


def _field(
    unit_ids: list[str],
    unit_probs: np.ndarray,
    domain_mask: np.ndarray | None = None,
) -> VoxelField:
    probs = np.asarray(unit_probs, dtype=np.float32)
    mask = np.ones(probs.shape[1:], dtype=bool) if domain_mask is None else domain_mask
    most_likely = probs.argmax(axis=0).astype(np.int16)
    most_likely[~mask] = -1
    return VoxelField(
        most_likely_unit=most_likely,
        unit_probs=probs,
        entropy=entropy_from_probs(probs, mask),
        domain_mask=mask,
        scalar_field=np.zeros((0, *mask.shape), dtype=np.float32),
        x=np.arange(mask.shape[0], dtype=float),
        y=np.arange(mask.shape[1], dtype=float),
        z=np.arange(mask.shape[2], dtype=float),
        unit_ids=unit_ids,
        feature_names=[],
    )


def test_confidence_weighted_forward_kl_does_not_veto_diffuse_reference() -> None:
    config = RefinementCriterionConfig(epsilon=0.01, effective_sample_size=1.0)
    diffuse_reference = _field(
        ["a", "b"],
        np.array([[[[0.5]]], [[[0.5]]]], dtype=np.float32),
    )
    sharpened_candidate = _field(
        ["a", "b"],
        np.array([[[[1.0]]], [[[0.0]]]], dtype=np.float32),
    )

    loss = confidence_weighted_capped_forward_kl(
        diffuse_reference,
        sharpened_candidate,
        config=config,
    )

    assert loss.loss_bits == pytest.approx(0.0)


def test_confident_contradiction_is_capped_per_source() -> None:
    config = RefinementCriterionConfig(epsilon=0.01, effective_sample_size=1.0)
    reference = _field(["a", "b"], np.array([[[[1.0]]], [[[0.0]]]], dtype=np.float32))
    contradiction = _field(["a", "b"], np.array([[[[0.0]]], [[[1.0]]]], dtype=np.float32))

    loss = confidence_weighted_capped_forward_kl(reference, contradiction, config=config)

    assert loss.loss_bits == pytest.approx(math.log2(1.0 / config.epsilon))


def test_coverage_gate_measures_candidate_coverage_of_reference_support() -> None:
    reference = _field(
        ["a", "b"],
        np.ones((2, 2, 1, 1), dtype=np.float32) * 0.5,
        domain_mask=np.array([[[True]], [[True]]]),
    )
    candidate = _field(
        ["a", "b"],
        np.ones((2, 2, 1, 1), dtype=np.float32) * 0.5,
        domain_mask=np.array([[[True]], [[False]]]),
    )

    assert coverage_ratio(reference, candidate) == pytest.approx(0.5)


def test_score_refinement_fails_before_scoring_when_candidate_retreats_mask() -> None:
    graph = two_unit_horizontal(z_interface=5.0)
    reference = _field(
        ["above", "below"],
        np.ones((2, 2, 1, 1), dtype=np.float32) * 0.5,
        domain_mask=np.array([[[True]], [[True]]]),
    )
    candidate = _field(
        ["above", "below"],
        np.ones((2, 2, 1, 1), dtype=np.float32) * 0.5,
        domain_mask=np.array([[[True]], [[False]]]),
    )
    config = RefinementCriterionConfig(coverage_threshold=0.75)

    result = score_refinement(
        candidate_graph=graph,
        candidate_field=candidate,
        reference_a_graph=graph,
        reference_a_field=reference,
        reference_b_graph=graph,
        reference_b_field=reference,
        config=config,
    )

    assert result.score_bits == math.inf
    assert [failure.name for failure in result.gate_failures] == ["coverage_a", "coverage_b"]


def test_score_refinement_prefers_candidate_that_covers_both_references() -> None:
    graph = two_unit_horizontal(z_interface=5.0)
    reference_a = _field(
        ["above", "below"],
        np.array([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]], dtype=np.float32),
    )
    reference_b = _field(
        ["above", "below"],
        np.array([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]], dtype=np.float32),
    )
    matching_candidate = _field(
        ["above", "below"],
        np.array([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]], dtype=np.float32),
    )
    wrong_candidate = _field(
        ["above", "below"],
        np.array([[[[0.0]], [[1.0]]], [[[1.0]], [[0.0]]]], dtype=np.float32),
    )
    config = RefinementCriterionConfig(epsilon=0.01, effective_sample_size=1.0)

    matching = score_refinement(
        candidate_graph=graph,
        candidate_field=matching_candidate,
        reference_a_graph=graph,
        reference_a_field=reference_a,
        reference_b_graph=graph,
        reference_b_field=reference_b,
        config=config,
    )
    wrong = score_refinement(
        candidate_graph=graph,
        candidate_field=wrong_candidate,
        reference_a_graph=graph,
        reference_a_field=reference_a,
        reference_b_graph=graph,
        reference_b_field=reference_b,
        config=config,
    )

    assert not matching.gate_failures
    assert matching.score_bits < wrong.score_bits
    assert wrong.fit_bits > matching.fit_bits


def test_dedup_rejects_structural_near_copy() -> None:
    graph = two_unit_horizontal(z_interface=5.0)
    config = RefinementCriterionConfig(dedup_epsilon=0.01)

    assert reject_dedup(graph, [two_unit_horizontal(z_interface=5.0)], config=config)
    assert not reject_dedup(graph, [two_unit_horizontal(z_interface=8.0)], config=config)
