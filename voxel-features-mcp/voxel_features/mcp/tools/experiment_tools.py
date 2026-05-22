"""Experiment recording and crossbreeding tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voxel_features.knowledge_graph import KnowledgeGraph, ExperimentRecord


def experiment_record(
    kg: KnowledgeGraph,
    hypothesis: str,
    rationale: str,
    data_spec: dict[str, Any],
    code_executed: str,
    result_summary: str,
    feature_layer_name: str | None = None,
    mdl_before: float | None = None,
    mdl_after: float | None = None,
    mdl_delta: float | None = None,
    mutual_info: dict[str, float] | None = None,
    admitted: bool = False,
    parent_experiments: list[str] | None = None,
    episode_id: str | None = None,
    variation_name: str | None = None,
) -> dict[str, Any]:
    """
    Record a completed experiment.
    
    This is called by the framework after the Evaluate phase completes.
    The rewriting agent then uses this record to generate training data.
    
    Args:
        hypothesis: The hypothesis that was tested
        rationale: Why the hypothesis was proposed
        data_spec: Files/columns/analysis used
        code_executed: The code that was run
        result_summary: What the analysis found
        feature_layer_name: Name of created layer (if any)
        mdl_before: MDL before adding layer
        mdl_after: MDL after adding layer  
        mdl_delta: Change in MDL
        mutual_info: MI with existing layers
        admitted: Whether layer was kept
        parent_experiments: IDs of parent experiments (for crossbreeding)
        episode_id: Episode identifier
        variation_name: Variation name
    
    Returns:
        experiment_id: The recorded experiment's ID
        prompt_response_pair: Generated training data
    """
    record = ExperimentRecord(
        hypothesis=hypothesis,
        rationale=rationale,
        data_spec=data_spec,
        code_executed=code_executed,
        result_summary=result_summary,
        feature_layer_name=feature_layer_name,
        mdl_before=mdl_before,
        mdl_after=mdl_after,
        mdl_delta=mdl_delta,
        mutual_info=mutual_info or {},
        admitted=admitted,
        parent_experiments=parent_experiments or [],
        episode_id=episode_id,
        variation_name=variation_name,
    )
    
    exp_id = kg.record(record)
    
    return {
        "success": True,
        "experiment_id": exp_id,
        "admitted": admitted,
        "prompt_response_pair": record.prompt_response_pair,
    }


def experiment_get(
    kg: KnowledgeGraph,
    experiment_id: str,
) -> dict[str, Any]:
    """
    Get an experiment record by ID.
    
    Args:
        experiment_id: The experiment ID
    
    Returns:
        The full experiment record
    """
    record = kg.get(experiment_id)
    if record is None:
        return {"success": False, "error": f"Experiment {experiment_id} not found"}
    
    return {
        "success": True,
        "experiment": record.to_dict(),
    }


def experiment_list_admitted(kg: KnowledgeGraph) -> dict[str, Any]:
    """
    List all admitted experiments.
    
    These are experiments where the hypothesis was supported and
    the feature layer improved compression.
    
    Returns:
        List of admitted experiment summaries
    """
    admitted = kg.list_admitted()
    
    return {
        "success": True,
        "count": len(admitted),
        "experiments": [
            {
                "id": exp.id,
                "hypothesis": exp.hypothesis,
                "feature_layer_name": exp.feature_layer_name,
                "mdl_delta": exp.mdl_delta,
                "timestamp": exp.timestamp,
            }
            for exp in admitted
        ],
    }


def experiment_get_crossbreed_pairs(
    kg: KnowledgeGraph,
    max_pairs: int = 5,
) -> dict[str, Any]:
    """
    Get pairs of admitted experiments for crossbreeding.
    
    Returns pairs that haven't been combined yet, with a
    suggested prompt for each pair.
    
    Args:
        max_pairs: Maximum number of pairs to return
    
    Returns:
        List of experiment pairs with crossbreed prompts
    """
    pairs = kg.get_crossbreed_pairs(max_pairs=max_pairs)
    
    result = []
    for exp_a, exp_b in pairs:
        prompt = kg.generate_crossbreed_prompt(exp_a, exp_b)
        result.append({
            "experiment_a": {
                "id": exp_a.id,
                "hypothesis": exp_a.hypothesis,
                "feature_layer_name": exp_a.feature_layer_name,
            },
            "experiment_b": {
                "id": exp_b.id,
                "hypothesis": exp_b.hypothesis,
                "feature_layer_name": exp_b.feature_layer_name,
            },
            "crossbreed_prompt": prompt,
            "parent_ids": [exp_a.id, exp_b.id],
        })
    
    return {
        "success": True,
        "count": len(result),
        "pairs": result,
    }


def experiment_export_training(
    kg: KnowledgeGraph,
    output_path: str,
) -> dict[str, Any]:
    """
    Export all experiments as JSONL training data.
    
    Args:
        output_path: Path to write JSONL file
    
    Returns:
        Number of records exported
    """
    count = kg.export_training_data(Path(output_path))
    stats = kg.stats()
    
    return {
        "success": True,
        "output_path": output_path,
        "records_exported": count,
        "stats": stats,
    }
