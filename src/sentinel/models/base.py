from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass

@dataclass(frozen=True)
class GenerationResult:
    
    answer: str
    tokens: list[str]
    logprobs: list[float]
    name: str


class ModelAdapter(ABC):
    @abstractmethod
    def generate(self, prompt:str -> GenerationResult):
        ...