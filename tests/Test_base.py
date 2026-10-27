"""Tests for the model adapter contract in sentinel.models.base."""

import dataclasses

import pytest

from sentinel.models.base import GenerationResult, ModelAdapter


def make_result() -> GenerationResult:
    """A result shaped like the real llama3.2:3b response for 'Paris.'"""
    return GenerationResult(
        answer="Paris.",
        tokens=["Paris", "."],
        logprobs=[-0.0004115355259273201, -0.03675909712910652],
        name="llama3.2:3b",
    )


# --- GenerationResult -------------------------------------------------------


def test_result_stores_fields() -> None:
    result = make_result()
    assert result.answer == "Paris."
    assert result.tokens == ["Paris", "."]
    assert result.logprobs == [-0.0004115355259273201, -0.03675909712910652]
    assert result.name == "llama3.2:3b"


def test_result_is_frozen() -> None:
    result = make_result()
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.answer = "London."  # type: ignore[misc]


def test_result_requires_all_fields() -> None:
    with pytest.raises(TypeError):
        GenerationResult(answer="Paris.")  # type: ignore[call-arg]


def test_results_with_same_values_are_equal() -> None:
    assert make_result() == make_result()


# --- ModelAdapter -----------------------------------------------------------


def test_adapter_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        ModelAdapter()  # type: ignore[abstract]


def test_subclass_missing_generate_cannot_be_instantiated() -> None:
    class IncompleteAdapter(ModelAdapter):
        pass

    with pytest.raises(TypeError):
        IncompleteAdapter()  # type: ignore[abstract]


def test_complete_subclass_works() -> None:
    class FakeAdapter(ModelAdapter):
        def generate(self, prompt: str) -> GenerationResult:
            return make_result()

    adapter = FakeAdapter()
    result = adapter.generate("What is the capital of France?")
    assert isinstance(result, GenerationResult)
    assert result.answer == "Paris."