from __future__ import annotations

import json
from pathlib import Path

from graph_to_voxel.graph.core import Graph


def save_graph(graph: Graph, path: str | Path) -> None:
    Path(path).write_text(json.dumps(graph.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_graph(path: str | Path) -> Graph:
    return Graph.from_file(path)
