"""Tests for digit-only constrained decoding.

Validates that DigitOnlyLogitsProcessor masks non-digit tokens
and that get_digit_token_ids correctly identifies digit tokens
in the RoBERTa vocabulary.
"""

from __future__ import annotations

import pytest
import torch


class TestGetDigitTokenIds:
    """Test get_digit_token_ids with the real TrOCR tokenizer."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("microsoft/trocr-small-printed")

    @pytest.fixture(scope="class")
    def token_sets(self, tokenizer):
        from cycling_photo_ai.ocr.inference.constrained_decoding import (
            get_digit_token_ids,
        )

        return get_digit_token_ids(tokenizer)

    def test_finds_all_ten_digits(self, tokenizer, token_sets):
        """Every digit 0-9 must appear in the allowed set."""
        digit_ids, _ = token_sets
        for digit in "0123456789":
            encoded = tokenizer.encode(digit, add_special_tokens=False)
            assert any(
                tid in digit_ids for tid in encoded
            ), f"Digit '{digit}' (token ids {encoded}) not in digit_ids"

    def test_includes_special_tokens(self, tokenizer, token_sets):
        """EOS, PAD, BOS/CLS, SEP must be in the special set."""
        _, special_ids = token_sets
        for attr in ("eos_token_id", "pad_token_id", "cls_token_id", "sep_token_id"):
            token_id = getattr(tokenizer, attr, None)
            if token_id is not None:
                assert token_id in special_ids, f"{attr}={token_id} not in special_ids"

    def test_excludes_letter_tokens(self, tokenizer, token_sets):
        """Pure letter tokens like 'hello' must NOT be in digit_ids."""
        digit_ids, _ = token_sets
        letter_ids = tokenizer.encode("hello", add_special_tokens=False)
        for tid in letter_ids:
            assert tid not in digit_ids, f"Letter token id {tid} should not be in digit_ids"

    def test_no_overlap_between_digit_and_special(self, token_sets):
        """Digit ids and special ids should be disjoint sets."""
        digit_ids, special_ids = token_sets
        overlap = digit_ids & special_ids
        assert len(overlap) == 0, f"Overlap between digit and special ids: {overlap}"


class TestDigitOnlyLogitsProcessor:
    """Test DigitOnlyLogitsProcessor masks correctly."""

    @pytest.fixture()
    def processor(self):
        from cycling_photo_ai.ocr.inference.constrained_decoding import (
            DigitOnlyLogitsProcessor,
        )

        allowed = {0, 1, 5}  # pretend these are digit + special tokens
        return DigitOnlyLogitsProcessor(allowed_token_ids=allowed, vocab_size=10)

    def test_masks_non_allowed_tokens_to_neg_inf(self, processor):
        """Non-allowed token scores must become -inf."""
        input_ids = torch.tensor([[1, 2]])
        scores = torch.ones(1, 10)

        result = processor(input_ids, scores)

        for i in range(10):
            if i not in {0, 1, 5}:
                assert result[0, i] == float("-inf"), f"Token {i} should be -inf"

    def test_preserves_allowed_token_scores(self, processor):
        """Allowed token scores must remain unchanged."""
        input_ids = torch.tensor([[1, 2]])
        scores = torch.full((1, 10), 3.14)

        result = processor(input_ids, scores)

        for i in (0, 1, 5):
            assert result[0, i] == pytest.approx(3.14), f"Token {i} score changed"

    def test_works_with_batch(self, processor):
        """Processor must handle batch_size > 1."""
        input_ids = torch.tensor([[1], [2], [3]])
        scores = torch.ones(3, 10)

        result = processor(input_ids, scores)

        assert result.shape == (3, 10)
        # All rows should have same masking
        for b in range(3):
            assert result[b, 0] == 1.0
            assert result[b, 2] == float("-inf")

    def test_does_not_modify_input_tensor(self, processor):
        """Processor should not mutate the original scores tensor."""
        input_ids = torch.tensor([[1]])
        scores = torch.ones(1, 10)
        original = scores.clone()

        processor(input_ids, scores)

        # The original should be unchanged (processor works on clone)
        assert torch.equal(scores, original)
