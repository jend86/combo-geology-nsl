"""Search tools for geological location resolution."""

from __future__ import annotations

import time
import requests
from typing import Any, Dict, List
from mcp.types import TextContent

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None


def web_search_geological(query: str) -> Dict[str, Any]:
    """
    Search for geological location information using DuckDuckGo.
    Automatically adds Kazakhstan context to improve relevance.

    Args:
        query: Search query (e.g., "Vladimirovskoye geological formation")

    Returns:
        Dict with 'success', 'results' (list of up to 3 results), 'error'
    """
    if not DDGS:
        return {
            "success": False,
            "error": "DuckDuckGo search not available. Install: pip install duckduckgo-search",
            "results": []
        }

    try:
        # Enhance query with geographic context
        enhanced_query = f"{query} Kazakhstan Teniz Basin coordinates"

        with DDGS() as ddgs:
            results = []
            search_results = ddgs.text(enhanced_query, max_results=3, timelimit='y')

            for result in search_results:
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                })

        return {
            "success": True,
            "results": results,
            "query_used": enhanced_query,
            "timestamp": time.time()
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Search failed: {str(e)}",
            "results": []
        }


def geonames_lookup(place_name: str, region: str = "Kazakhstan") -> Dict[str, Any]:
    """
    Look up geographical coordinates using OpenStreetMap Nominatim.
    Filters results to Kazakhstan/Central Asia region.

    Args:
        place_name: Name to search for (e.g., "Vladimirovskoye", "M42-I")
        region: Geographic region to constrain search (default: Kazakhstan)

    Returns:
        Dict with 'success', 'results' (list of locations), 'error'
    """
    try:
        # Nominatim API endpoint
        url = "https://nominatim.openstreetmap.org/search"

        # Search with geographic constraint
        params = {
            "q": f"{place_name} {region}",
            "format": "json",
            "limit": 5,
            "countrycodes": "kz",  # Kazakhstan country code
            "addressdetails": 1,
            "extratags": 1,
        }

        headers = {
            "User-Agent": "VoxelFeatures/1.0 (geological-research)"
        }

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Filter and format results
        filtered_results = []
        teniz_basin_bounds = {
            "min_lon": 66.5, "max_lon": 71.5,
            "min_lat": 49.5, "max_lat": 52.5
        }

        for item in data:
            try:
                lat = float(item.get("lat", 0))
                lon = float(item.get("lon", 0))

                # Check if within broader Kazakhstan bounds (allow some flexibility)
                if 60.0 <= lon <= 80.0 and 45.0 <= lat <= 55.0:
                    in_teniz_basin = (
                        teniz_basin_bounds["min_lon"] <= lon <= teniz_basin_bounds["max_lon"] and
                        teniz_basin_bounds["min_lat"] <= lat <= teniz_basin_bounds["max_lat"]
                    )

                    filtered_results.append({
                        "display_name": item.get("display_name", ""),
                        "lat": lat,
                        "lon": lon,
                        "type": item.get("type", ""),
                        "class": item.get("class", ""),
                        "importance": float(item.get("importance", 0)),
                        "in_teniz_basin": in_teniz_basin,
                        "distance_score": _calculate_distance_score(lat, lon, teniz_basin_bounds)
                    })
            except (ValueError, TypeError):
                continue

        # Sort by relevance (in basin first, then by distance score, then importance)
        filtered_results.sort(key=lambda x: (
            not x["in_teniz_basin"],  # False sorts before True, so in_basin items first
            x["distance_score"],
            -x["importance"]
        ))

        return {
            "success": True,
            "results": filtered_results[:3],  # Top 3 results
            "total_found": len(data),
            "filtered_count": len(filtered_results),
            "search_term": f"{place_name} {region}"
        }

    except requests.RequestException as e:
        return {
            "success": False,
            "error": f"Network error: {str(e)}",
            "results": []
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Lookup failed: {str(e)}",
            "results": []
        }


def _calculate_distance_score(lat: float, lon: float, basin_bounds: Dict[str, float]) -> float:
    """Calculate distance score from Teniz Basin center (lower is better)."""
    basin_center_lat = (basin_bounds["min_lat"] + basin_bounds["max_lat"]) / 2
    basin_center_lon = (basin_bounds["min_lon"] + basin_bounds["max_lon"]) / 2

    # Simple euclidean distance (good enough for scoring)
    distance = ((lat - basin_center_lat) ** 2 + (lon - basin_center_lon) ** 2) ** 0.5
    return distance
