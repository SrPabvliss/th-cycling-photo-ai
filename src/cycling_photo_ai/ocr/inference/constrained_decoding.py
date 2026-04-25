"""Constrained decoding for digit-only TrOCR output.

TrOCR uses a RoBERTa decoder with ~64K vocabulary tokens. Without
constraint, the model can hallucinate letters, words, and subword
fragments. This module restricts generation to digits 0-9 plus
special tokens (EOS, PAD, BOS/CLS, SEP).
"""

from __future__ import annotations

import torch


def get_digit_token_ids(tokenizer) -> tuple[set[int], set[int]]:
    """Scan the tokenizer vocabulary for pure-digit and special tokens.

    RoBERTa tokens may carry a ``Ġ`` (space-prefix) or ``▁`` (sentencepiece)
    marker. We strip those before checking if the remaining string is all
    digits. We also explicitly encode each digit 0-9 to catch any tokens
    that the vocabulary scan might miss.

    Args:
        tokenizer: A HuggingFace tokenizer instance.

    Returns:
        A tuple of (digit_ids, special_ids) where each is a set of token ids.
    """
    digit_ids: set[int] = set()
    special_ids: set[int] = set()

    # Collect special token ids
    for attr in ("eos_token_id", "pad_token_id", "cls_token_id", "sep_token_id", "bos_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            special_ids.add(token_id)

    # Scan full vocabulary for pure-digit tokens
    vocab: dict[str, int] = tokenizer.get_vocab()
    for token_str, token_id in vocab.items():
        stripped = token_str.lstrip("Ġ▁ ")
        if stripped and stripped.isdigit():
            digit_ids.add(token_id)

    # Also explicitly encode each single digit to be safe
    for digit_char in "0123456789":
        encoded = tokenizer.encode(digit_char, add_special_tokens=False)
        for tid in encoded:
            digit_ids.add(tid)

    # Remove any special tokens that ended up in digit_ids
    digit_ids -= special_ids

    return digit_ids, special_ids


class DigitOnlyLogitsProcessor:
    """Masks all non-digit, non-special tokens to ``-inf`` during generation.

    Compatible with ``transformers``' ``LogitsProcessor`` protocol: callable
    with signature ``(input_ids: Tensor, scores: Tensor) -> Tensor``.
    """

    def __init__(self, allowed_token_ids: set[int], vocab_size: int) -> None:
        self._mask = torch.zeros(vocab_size, dtype=torch.bool)
        for tid in allowed_token_ids:
            if tid < vocab_size:
                self._mask[tid] = True

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Mask disallowed tokens to ``-inf``, preserving allowed scores.

        Args:
            input_ids: Generated token ids so far, shape ``(batch, seq_len)``.
            scores: Logits for next token, shape ``(batch, vocab_size)``.

        Returns:
            Modified scores with disallowed positions set to ``-inf``.
        """
        mask = self._mask.to(scores.device)
        result = scores.clone()
        result[:, ~mask] = float("-inf")
        return result
