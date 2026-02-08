"""Tests for r1_interp.prompts — unique IDs, categories, JSON roundtrip."""

from __future__ import annotations

import json

from r1_interp.prompts import (
    BENCHMARK_PROMPTS,
    EXPECTED_CATEGORIES,
    BenchmarkPrompt,
    export_prompts_json,
    get_prompts_by_category,
    load_prompts_json,
)


class TestBenchmarkPrompts:
    def test_at_least_24_prompts(self):
        assert len(BENCHMARK_PROMPTS) >= 24

    def test_unique_ids(self):
        ids = [p.prompt_id for p in BENCHMARK_PROMPTS]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {[x for x in ids if ids.count(x) > 1]}"

    def test_all_expected_categories_present(self):
        actual = {p.category for p in BENCHMARK_PROMPTS}
        assert actual == EXPECTED_CATEGORIES

    def test_each_category_has_prompts(self):
        for cat in EXPECTED_CATEGORIES:
            prompts = get_prompts_by_category(cat)
            assert len(prompts) >= 2, f"Category '{cat}' has fewer than 2 prompts"

    def test_prompts_have_nonempty_text(self):
        for p in BENCHMARK_PROMPTS:
            assert p.text.strip(), f"{p.prompt_id} has empty text"

    def test_prompts_have_expected_answer(self):
        for p in BENCHMARK_PROMPTS:
            assert p.expected_answer, f"{p.prompt_id} has no expected_answer"


class TestJSONRoundtrip:
    def test_export_and_load(self, tmp_path):
        path = tmp_path / "prompts.json"
        export_prompts_json(path)
        loaded = load_prompts_json(path)
        assert len(loaded) == len(BENCHMARK_PROMPTS)
        for orig, back in zip(BENCHMARK_PROMPTS, loaded):
            assert orig.prompt_id == back.prompt_id
            assert orig.category == back.category
            assert orig.text == back.text

    def test_json_structure(self, tmp_path):
        path = tmp_path / "prompts.json"
        export_prompts_json(path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        for item in data:
            assert "prompt_id" in item
            assert "category" in item
            assert "text" in item
            assert "expected_answer" in item
            assert "requires_reasoning" in item
