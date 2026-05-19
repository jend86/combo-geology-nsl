from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest


def provenance() -> dict[str, Any]:
    return {
        "source": "test-fixture",
        "reference": None,
        "confidence": 1.0,
        "timestamp": datetime(2026, 5, 6, tzinfo=UTC).isoformat(),
        "agent": "pytest",
    }


def point(value: float) -> dict[str, Any]:
    return {"kind": "Point", "value": value}


def gaussian(mean: float, std: float) -> dict[str, Any]:
    return {"kind": "Gaussian", "mean": mean, "std": std}


def interval(lo: float, hi: float) -> dict[str, Any]:
    return {"kind": "Interval", "lo": lo, "hi": hi}


@pytest.fixture
def two_unit_graph_dict() -> dict[str, Any]:
    prov = provenance()
    contacts = [
        ("c0", 0.0, 0.0, 5.0),
        ("c1", 10.0, 0.0, 5.0),
        ("c2", 0.0, 10.0, 5.0),
    ]
    return {
        "nodes": [
            {
                "kind": "StratigraphicUnit",
                "id": "unit_upper",
                "unit_id": "upper",
                "series_id": "main",
                "topology": "layer",
                "p_exists": 1.0,
                "provenance": prov,
                "bulk_volume_bounds": interval(400.0, 600.0),
                "metadata": {"connectivity": "single"},
            },
            {
                "kind": "StratigraphicUnit",
                "id": "unit_lower",
                "unit_id": "lower",
                "series_id": "main",
                "topology": "layer",
                "p_exists": 1.0,
                "provenance": prov,
                "metadata": {"connectivity": "single"},
            },
            *[
                {
                    "kind": "Contact",
                    "id": contact_id,
                    "position": [point(x), point(y), point(z)],
                    "between": ["upper", "lower"],
                    "polarity": 1,
                    "p_exists": 1.0,
                    "provenance": prov,
                }
                for contact_id, x, y, z in contacts
            ],
            {
                "kind": "Orientation",
                "id": "o0",
                "position": [point(5.0), point(5.0), point(5.0)],
                "dip": {
                    "kind": "Orientation",
                    "dip_mean": 0.0,
                    "dip_kappa": 100.0,
                    "azimuth_mean": 0.0,
                    "azimuth_kappa": 100.0,
                },
                "for_unit": "upper",
                "p_exists": 1.0,
                "provenance": prov,
            },
        ],
        "edges": [
            {
                "kind": "OVERLIES",
                "source": "unit_upper",
                "target": "unit_lower",
                "p_exists": 1.0,
                "provenance": prov,
            },
        ],
        "metadata": {"name": "two-unit-horizontal"},
    }


@pytest.fixture
def gaussian_interface_graph_dict(two_unit_graph_dict: dict[str, Any]) -> dict[str, Any]:
    graph = two_unit_graph_dict.copy()
    graph["nodes"] = [node.copy() for node in two_unit_graph_dict["nodes"]]
    for node in graph["nodes"]:
        if node["kind"] == "Contact":
            node["position"] = [node["position"][0], node["position"][1], gaussian(5.0, 0.6)]
    return graph
