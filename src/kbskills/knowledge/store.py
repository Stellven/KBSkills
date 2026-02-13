"""Storage management - knowledge base status and utilities."""

import json
from pathlib import Path

from kbskills.config import Config


def get_kb_status(config: Config) -> dict:
    """Get knowledge base status information."""
    graph_dir = Path(config.data_dir) / "graph"

    info = {
        "Data Directory": config.data_dir,
        "Graph Directory": str(graph_dir),
        "Graph Exists": str(graph_dir.exists()),
    }

    if not graph_dir.exists():
        info["Status"] = "Not initialized (run 'kbskills ingest' first)"
        return info

    # Count files in graph directory
    graph_files = list(graph_dir.glob("*"))
    info["Graph Files"] = str(len(graph_files))

    # Check for graph data
    graph_json = graph_dir / "graph_chunk_entity_relation.json"
    if graph_json.exists():
        try:
            data = json.loads(graph_json.read_text())
            entities = sum(1 for v in data.values() if v.get("type") == "entity") if isinstance(data, dict) else 0
            relations = sum(1 for v in data.values() if v.get("type") == "relation") if isinstance(data, dict) else 0
            info["Entities"] = str(entities)
            info["Relations"] = str(relations)
        except (json.JSONDecodeError, AttributeError):
            pass

    # Check kv store for document count
    kv_full = graph_dir / "kv_store_full_docs.json"
    if kv_full.exists():
        try:
            data = json.loads(kv_full.read_text())
            info["Documents"] = str(len(data))
        except json.JSONDecodeError:
            pass

    # Total storage size
    total_size = sum(f.stat().st_size for f in graph_dir.rglob("*") if f.is_file())
    if total_size > 1_000_000:
        info["Storage Size"] = f"{total_size / 1_000_000:.1f} MB"
    else:
        info["Storage Size"] = f"{total_size / 1_000:.1f} KB"

    return info
