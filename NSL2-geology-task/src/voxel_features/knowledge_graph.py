"""Knowledge graph for experiment records and crossbreeding.

Stores the results of hypothesis testing experiments:
- What hypothesis was tested
- What code was run
- What the results were
- Whether the layer was admitted

Used for:
1. Training data export (prompt/response pairs)
2. Crossbreeding (combining successful experiments to generate new hypotheses)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExperimentRecord:
    """Record of a single hypothesis testing experiment."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Hypothesis details
    hypothesis: str = ""
    rationale: str = ""
    data_spec: dict[str, Any] = field(default_factory=dict)
    
    # Execution details
    code_executed: str = ""
    result_summary: str = ""
    
    # Feature layer (if created)
    feature_layer_name: str | None = None
    
    # Scoring (BIC on ridge CV with depth folds)
    bic_before: float | None = None
    bic_after: float | None = None
    bic_delta: float | None = None
    cv_mse_delta: float | None = None  # Cross-validated MSE change
    predicted_value: float | None = None  # For hypothesis agent training (-bic_delta)
    mutual_info: dict[str, float] = field(default_factory=dict)
    
    # Admission decision
    admitted: bool = False
    
    # Lineage for crossbreeding
    parent_experiments: list[str] = field(default_factory=list)
    
    # Training data export
    prompt_response_pair: dict[str, str] = field(default_factory=dict)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    episode_id: str | None = None
    variation_name: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentRecord:
        return cls(**data)
    
    def generate_training_pair(self, include_score: bool = True) -> dict[str, Any]:
        """
        Generate a prompt/response pair for training.
        
        The prompt includes the hypothesis context.
        The response includes what was discovered.
        
        If include_score=True, also returns the predicted_value for training
        the hypothesis agent to predict which hypotheses will be valuable.
        """
        # Build prompt
        prompt_parts = [
            f"Hypothesis: {self.hypothesis}",
            f"Rationale: {self.rationale}",
        ]
        if self.data_spec:
            prompt_parts.append(f"Data specification: {json.dumps(self.data_spec)}")
        prompt = "\n".join(prompt_parts)
        
        # Build response
        response_parts = [
            f"Analysis result: {self.result_summary}",
        ]
        if self.bic_delta is not None:
            direction = "improved" if self.bic_delta < 0 else "worsened"
            response_parts.append(
                f"BIC: {direction} by {abs(self.bic_delta):.2f}"
            )
        if self.cv_mse_delta is not None:
            cv_direction = "improved" if self.cv_mse_delta < 0 else "worsened"
            response_parts.append(
                f"Depth CV-MSE: {cv_direction} by {abs(self.cv_mse_delta):.4f}"
            )
        if self.admitted:
            response_parts.append(
                f"Conclusion: Hypothesis supported. Feature '{self.feature_layer_name}' "
                f"added to world model."
            )
        else:
            response_parts.append(
                "Conclusion: Hypothesis not supported or redundant. "
                "Feature not added."
            )
        response = "\n".join(response_parts)
        
        result = {"prompt": prompt, "response": response}
        
        # Include score for hypothesis-value prediction training
        if include_score and self.predicted_value is not None:
            result["score"] = self.predicted_value
        
        return result


class KnowledgeGraph:
    """
    Persistent store for experiment records.
    
    Supports:
    - Recording new experiments
    - Querying admitted experiments for crossbreeding
    - Exporting training data
    """
    
    def __init__(self, store_path: Path | str):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self._index_path = self.store_path / "experiments.json"
        self._experiments: dict[str, ExperimentRecord] = {}
        
        if self._index_path.exists():
            self._load()
    
    def _load(self) -> None:
        """Load experiments from disk."""
        with open(self._index_path) as f:
            data = json.load(f)
        self._experiments = {
            exp_id: ExperimentRecord.from_dict(exp_data)
            for exp_id, exp_data in data.items()
        }
    
    def _save(self) -> None:
        """Save experiments to disk."""
        data = {
            exp_id: exp.to_dict()
            for exp_id, exp in self._experiments.items()
        }
        with open(self._index_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def record(self, experiment: ExperimentRecord) -> str:
        """Record a new experiment. Returns the experiment ID."""
        # Generate training pair
        experiment.prompt_response_pair = experiment.generate_training_pair()
        
        self._experiments[experiment.id] = experiment
        self._save()
        return experiment.id
    
    def get(self, experiment_id: str) -> ExperimentRecord | None:
        """Get an experiment by ID."""
        return self._experiments.get(experiment_id)
    
    def list_all(self) -> list[ExperimentRecord]:
        """List all experiments."""
        return list(self._experiments.values())
    
    def list_admitted(self) -> list[ExperimentRecord]:
        """List only admitted experiments (for crossbreeding)."""
        return [exp for exp in self._experiments.values() if exp.admitted]
    
    def list_rejected(self) -> list[ExperimentRecord]:
        """List only rejected experiments."""
        return [exp for exp in self._experiments.values() if not exp.admitted]
    
    def get_crossbreed_pairs(
        self,
        max_pairs: int = 5,
        exclude_parents: list[str] | None = None,
        prefer_orthogonal: bool = True,
    ) -> list[tuple[ExperimentRecord, ExperimentRecord]]:
        """
        Get pairs of admitted experiments for crossbreeding.
        
        Ranking strategy (Solomonoff-informed):
        1. Sort by combined score (best performers first)
        2. Prefer pairs with low mutual information (orthogonal features)
        3. Skip already-crossed combinations
        
        Returns pairs that haven't been combined yet.
        """
        admitted = self.list_admitted()
        if len(admitted) < 2:
            return []
        
        exclude_parents = exclude_parents or []
        
        # Sort by predicted_value (higher = better compression + generalization)
        admitted_sorted = sorted(
            admitted,
            key=lambda e: e.predicted_value or 0,
            reverse=True,  # Best first
        )
        
        # Find pairs that haven't been crossed
        candidate_pairs = []
        seen_combinations = set()
        
        for i, exp_a in enumerate(admitted_sorted):
            for exp_b in admitted_sorted[i+1:]:
                # Skip if this combination already exists as a parent pair
                combo_key = tuple(sorted([exp_a.id, exp_b.id]))
                if combo_key in seen_combinations:
                    continue
                
                # Check if any existing experiment has this pair as parents
                already_crossed = any(
                    set([exp_a.id, exp_b.id]) <= set(exp.parent_experiments)
                    for exp in self._experiments.values()
                )
                
                if not already_crossed:
                    # Compute orthogonality score (low MI = more orthogonal)
                    mi_between = 0.0
                    if exp_a.feature_layer_name and exp_b.feature_layer_name:
                        mi_between = exp_a.mutual_info.get(exp_b.feature_layer_name, 0)
                        mi_between += exp_b.mutual_info.get(exp_a.feature_layer_name, 0)
                        mi_between /= 2
                    
                    # Combined rank: sum of values, penalize by MI if prefer_orthogonal
                    value_a = exp_a.predicted_value or 0
                    value_b = exp_b.predicted_value or 0
                    pair_score = value_a + value_b
                    if prefer_orthogonal:
                        pair_score -= mi_between  # Lower MI = better
                    
                    candidate_pairs.append((pair_score, exp_a, exp_b))
                    seen_combinations.add(combo_key)
        
        # Sort pairs by combined score and take top N
        candidate_pairs.sort(key=lambda x: x[0], reverse=True)
        pairs = [(exp_a, exp_b) for _, exp_a, exp_b in candidate_pairs[:max_pairs]]
        
        return pairs
    
    def generate_crossbreed_prompt(
        self,
        exp_a: ExperimentRecord,
        exp_b: ExperimentRecord,
    ) -> str:
        """
        Generate a prompt for crossbreeding two experiments.
        """
        return f"""These experiments both improved the world model:

Experiment 1: "{exp_a.hypothesis}"
- Result: {exp_a.result_summary}
- BIC improvement: {abs(exp_a.bic_delta or 0):.2f}
- Feature: {exp_a.feature_layer_name}

Experiment 2: "{exp_b.hypothesis}"
- Result: {exp_b.result_summary}
- BIC improvement: {abs(exp_b.bic_delta or 0):.2f}
- Feature: {exp_b.feature_layer_name}

Given that both of these patterns exist in the data, what new hypothesis 
would you propose that combines or builds on these findings?"""
    
    def export_training_data(self, output_path: Path | str) -> int:
        """
        Export all experiments as JSONL training data.
        
        Returns number of records exported.
        """
        output_path = Path(output_path)
        count = 0
        
        with open(output_path, "w") as f:
            for exp in self._experiments.values():
                record = {
                    "id": exp.id,
                    "prompt": exp.prompt_response_pair.get("prompt", ""),
                    "response": exp.prompt_response_pair.get("response", ""),
                    "admitted": exp.admitted,
                    "mdl_delta": exp.mdl_delta,
                    "timestamp": exp.timestamp,
                }
                f.write(json.dumps(record) + "\n")
                count += 1
        
        return count
    
    def stats(self) -> dict[str, Any]:
        """Get summary statistics about the knowledge graph."""
        all_exp = self.list_all()
        admitted = self.list_admitted()
        
        return {
            "total_experiments": len(all_exp),
            "admitted": len(admitted),
            "rejected": len(all_exp) - len(admitted),
            "admission_rate": len(admitted) / len(all_exp) if all_exp else 0,
            "total_mdl_improvement": sum(
                exp.mdl_delta or 0 for exp in admitted
            ),
            "unique_features": len(set(
                exp.feature_layer_name for exp in admitted if exp.feature_layer_name
            )),
        }
