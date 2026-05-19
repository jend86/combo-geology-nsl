from abc import ABC, abstractmethod
from typing import Any, List

from result import Result

from src.observability.types import InferenceResult, UsageInfo
from src.typing.message import Message


CONTEXT_OVERFLOW_PREFIX = "context_overflow:"
INFERENCE_UNAVAILABLE_PREFIX = "inference_unavailable:"


class Genner(ABC):
    client: Any = None
    collector: Any = None

    def __init__(self, identifier: str):
        self.identifier = identifier
        self.client = None
        self.collector = None

    @abstractmethod
    def plist_completion(
        self,
        messages: List[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> Result[InferenceResult, str]:
        pass

    @staticmethod
    @abstractmethod
    def get_usage_info(response: object) -> UsageInfo:
        pass
