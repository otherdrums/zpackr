"""Tests for prompt_gate.py — Super Dict binary training gate."""

import pytest
from zpackr.prompt_gate import should_train


class TestShouldTrain:
    def test_known_text_skipped(self, super_dict):
        text = b"The quick brown fox jumps over the lazy dog. " * 100
        assert not should_train(text, super_dict), "Known English text should be skipped"

    def test_novel_bytes_trained(self, super_dict):
        import os
        novel = os.urandom(4000)
        assert should_train(novel, super_dict), "Novel random bytes should trigger training"

    def test_empty_prompt_raises(self, super_dict):
        with pytest.raises(ValueError):
            should_train(b"", super_dict)

    def test_threshold_sensitivity(self, super_dict):
        text = b"hello world " * 50
        ratio = super_dict.compress(text)
        assert should_train(text, super_dict, threshold=ratio + 0.1), (
            "Should train when threshold just above actual ratio"
        )
        assert not should_train(text, super_dict, threshold=ratio - 0.1), (
            "Should skip when threshold just below actual ratio"
        )
