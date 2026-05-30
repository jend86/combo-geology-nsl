"""Spatial search tools — geological web search + geonames lookup.

Network is always mocked: no live DuckDuckGo / OpenStreetMap calls in CI.

The module is loaded directly by file path rather than via
``import voxel_features.mcp.tools.search_tools`` so the test does not pull in
the package ``__init__`` (which imports numpy-backed scoring tools). search_tools
itself only needs ``requests`` + ``mcp.types``, so it loads even where the heavy
geo/scoring stack is unavailable.
"""

import importlib.util
from pathlib import Path

import pytest

_ST_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "voxel_features" / "mcp" / "tools" / "search_tools.py"
)
_spec = importlib.util.spec_from_file_location("search_tools_under_test", _ST_PATH)
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)


class TestWebSearchGeological:
    def test_success_with_mocked_ddgs(self, monkeypatch) -> None:
        class FakeDDGS:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def text(self, query, max_results=3, timelimit=None):
                return [{"title": "T", "href": "http://x", "body": "B"}]

        monkeypatch.setattr(st, "DDGS", FakeDDGS)
        out = st.web_search_geological("Vladimirovskoye formation")
        assert out["success"] is True
        assert out["results"][0]["url"] == "http://x"
        assert "Kazakhstan" in out["query_used"]  # geographic context appended

    def test_ddgs_unavailable_returns_failure(self, monkeypatch) -> None:
        monkeypatch.setattr(st, "DDGS", None)
        out = st.web_search_geological("anything")
        assert out["success"] is False
        assert out["results"] == []


class TestGeonamesLookup:
    def test_filters_to_region_and_ranks_in_basin_first(self, monkeypatch) -> None:
        class FakeResp:
            def raise_for_status(self):
                return None

            def json(self):
                return [
                    {"lat": "51.0", "lon": "69.0", "display_name": "in basin",
                     "importance": 0.5, "type": "x", "class": "y"},
                    {"lat": "47.0", "lon": "75.0", "display_name": "kz out of basin",
                     "importance": 0.9, "type": "x", "class": "y"},
                    {"lat": "10.0", "lon": "10.0", "display_name": "far away",
                     "importance": 0.9, "type": "x", "class": "y"},
                ]

        monkeypatch.setattr(st.requests, "get", lambda *a, **k: FakeResp())
        out = st.geonames_lookup("M42-I")
        assert out["success"] is True
        # Far-away point is outside the broad Kazakhstan bounds → filtered out.
        assert all(r["display_name"] != "far away" for r in out["results"])
        # In-basin point ranks first.
        assert out["results"][0]["in_teniz_basin"] is True

    def test_network_error_returns_failure(self, monkeypatch) -> None:
        import requests
        def boom(*a, **k):
            raise requests.RequestException("network down")

        monkeypatch.setattr(st.requests, "get", boom)
        out = st.geonames_lookup("anything")
        assert out["success"] is False
        assert out["results"] == []


def test_distance_score_zero_at_basin_center() -> None:
    bounds = {"min_lon": 66.5, "max_lon": 71.5, "min_lat": 49.5, "max_lat": 52.5}
    # Basin centre is (lat 51.0, lon 69.0).
    assert st._calculate_distance_score(51.0, 69.0, bounds) == pytest.approx(0.0, abs=1e-9)
