from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class DerivationSpec(BaseModel):
    pipeline: Literal["kriging", "mcue", "thresholding", "manual_curation"]
    input_node_ids: list[str]
    params: dict[str, Any]
    run_id: str


class Provenance(BaseModel):
    source: str
    reference: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime
    agent: str | None = None
    derivation: DerivationSpec | None = None
