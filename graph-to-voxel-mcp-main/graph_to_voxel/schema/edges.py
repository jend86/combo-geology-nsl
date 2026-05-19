from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from graph_to_voxel.schema.provenance import Provenance


class EdgeKind(str, Enum):
    OVERLIES = ("overlies", "http://resource.geosciml.org/classifier/cgi/contacttype/conformable")
    IN_CONTACT_WITH = ("in_contact_with", "http://resource.geosciml.org/classifier/cgi/contacttype/contact")
    OFFSET_BY = ("offset_by", "http://resource.geosciml.org/classifier/cgi/contacttype/faulted_contact")
    MEMBER_OF_SERIES = ("member_of_series", "http://resource.geosciml.org/classifier/cgi/geologicunitpartrole/member")
    OBSERVED_AT = ("observed_at", "http://resource.geosciml.org/classifier/cgi/mappedfeatureobservationmethod/observed")
    WITHIN = ("within", "http://resource.geosciml.org/classifier/cgi/contacttype/spatially_within")
    AT = ("at", "http://resource.geosciml.org/classifier/cgi/samplinglocation")

    def __new__(cls, value: str, geosciml_uri: str) -> EdgeKind:
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._geosciml_uri = geosciml_uri
        return obj

    @property
    def geosciml_uri(self) -> str:
        return self._geosciml_uri


class GraphEdge(BaseModel):
    id: str | None = None
    kind: EdgeKind
    source: str
    target: str
    p_exists: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: Provenance
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind", mode="before")
    @classmethod
    def _parse_kind(cls, value: Any) -> Any:
        if isinstance(value, EdgeKind):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            if normalized in EdgeKind.__members__:
                return EdgeKind[normalized]
            lowered = normalized.lower()
            for kind in EdgeKind:
                if kind.value == lowered:
                    return kind
        return value
