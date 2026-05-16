"""
Tests for CLIP/src/tokenizer.py — SimpleWordTokenizer.

Run
---
    cd D:/Dev/repos/paper-reforge/CLIP
    python -m pytest tests/test_tokenizer.py -v
"""

import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from tokenizer import SimpleWordTokenizer, PAD_ID, SOS_ID, EOS_ID


# ---------------------------------------------------------------------------- #
# Fixtures                                                                     #
# ---------------------------------------------------------------------------- #


@pytest.fixture
def corpus():
    return [
        "a photo of a cat",
        "a photo of a dog",
        "a blurry image of a bird",
    ]


@pytest.fixture
def tok(corpus):
    t = SimpleWordTokenizer()
    t.build_vocab(corpus)
    return t


# ---------------------------------------------------------------------------- #
# Special tokens                                                               #
# ---------------------------------------------------------------------------- #


def test_special_tokens_have_fixed_ids():
    """PAD/SOS/EOS 的 id 必须固定为 0/1/2 — 后面 causal mask 假设 EOS_ID=2 没意义,
    但下游代码会从这里 import 这些常量, 它们的相对顺序必须稳定."""
    assert PAD_ID == 0
    assert SOS_ID == 1
    assert EOS_ID == 2


def test_special_tokens_in_empty_vocab():
    """没 build_vocab 之前, vocab 只有 3 个特殊 token."""
    t = SimpleWordTokenizer()
    assert t.vocab_size == 3
    assert t.word2id["<PAD>"] == PAD_ID
    assert t.word2id["<SOS>"] == SOS_ID
    assert t.word2id["<EOS>"] == EOS_ID


# ---------------------------------------------------------------------------- #
# build_vocab                                                                  #
# ---------------------------------------------------------------------------- #


def test_build_vocab_size(tok):
    """corpus 里去重后有 9 个 unique word ({a, photo, of, cat, dog, blurry, image, bird}
    实际是 8 个 — a 重复). 加 3 个特殊 token = 11."""
    # words: a, photo, of, cat, dog, blurry, image, bird   -> 8 unique
    assert tok.vocab_size == 3 + 8


def test_build_vocab_lowercase(corpus):
    """默认 lowercase=True 时, "Cat" 和 "cat" 应该是同一个 id."""
    t = SimpleWordTokenizer(lowercase=True)
    t.build_vocab(["a Cat", "a cat"])
    # vocab: <PAD>, <SOS>, <EOS>, a, cat  → size 5
    assert t.vocab_size == 5


def test_build_vocab_idempotent(corpus, tok):
    """对同一个 corpus 再 build 一次, vocab 不应该重复增长 — 但我们的实现
    会重复添加 (build_vocab 不是 idempotent 的). 这个测试记录这个事实:
    如果想 idempotent 行为, 用户应该自己重新创建 tokenizer."""
    size_before = tok.vocab_size
    tok.build_vocab(corpus)
    # 重复 word 已经在 vocab 里, 不会被再添加
    assert tok.vocab_size == size_before


# ---------------------------------------------------------------------------- #
# encode                                                                       #
# ---------------------------------------------------------------------------- #


def test_encode_returns_long_tensor(tok):
    ids, eos_pos = tok.encode("a photo of a cat", max_len=10)
    assert isinstance(ids, torch.Tensor)
    assert ids.dtype == torch.long
    assert ids.shape == (10,)
    assert isinstance(eos_pos, int)


def test_encode_layout(tok):
    """检查 [SOS, w1, ..., wK, EOS, PAD, PAD, ...] 的 layout."""
    ids, eos_pos = tok.encode("a photo of a cat", max_len=10)
    ids_list = ids.tolist()

    # 位置 0 是 SOS
    assert ids_list[0] == SOS_ID
    # 位置 6 是 EOS (SOS + 5 words = positions 0..5, EOS at 6)
    assert eos_pos == 6
    assert ids_list[6] == EOS_ID
    # 中间是 5 个真实 word id, 都不是特殊 token
    for i in range(1, 6):
        assert ids_list[i] not in (PAD_ID, SOS_ID, EOS_ID)
    # 尾部 PAD
    for i in range(7, 10):
        assert ids_list[i] == PAD_ID


def test_encode_truncation(tok):
    """超长文本: 5 words + SOS + EOS = 7 tokens, max_len=5 时必须截断,
    但 EOS 必须保留在末尾."""
    ids, eos_pos = tok.encode("a photo of a cat", max_len=5)
    ids_list = ids.tolist()

    assert len(ids_list) == 5
    assert ids_list[0] == SOS_ID
    assert ids_list[-1] == EOS_ID
    assert eos_pos == 4  # 最后一个位置


def test_encode_short_text_padding(tok):
    """短文本 'a cat' = SOS + 2 words + EOS = 4 tokens, max_len=10 时
    后面 6 位 PAD."""
    ids, eos_pos = tok.encode("a cat", max_len=10)
    ids_list = ids.tolist()

    assert eos_pos == 3
    assert ids_list[3] == EOS_ID
    assert all(ids_list[i] == PAD_ID for i in range(4, 10))


def test_encode_eos_always_present(tok):
    """无论截断多狠, EOS 都必须存在于序列里 (downstream forward 取 EOS 时不能错位)."""
    for max_len in [3, 4, 5, 8, 16]:
        ids, eos_pos = tok.encode("a photo of a cat", max_len=max_len)
        assert ids[eos_pos].item() == EOS_ID, f"max_len={max_len} 时 EOS 丢失"


# ---------------------------------------------------------------------------- #
# decode                                                                       #
# ---------------------------------------------------------------------------- #


def test_decode_roundtrip(tok):
    """encode → decode 应该还原原文本 (lowercase 后)."""
    text = "a photo of a cat"
    ids, _ = tok.encode(text, max_len=10)
    decoded = tok.decode(ids)
    assert decoded == text


def test_decode_skips_special_tokens(tok):
    """decode 必须跳过 PAD/SOS/EOS."""
    ids, _ = tok.encode("a cat", max_len=10)
    decoded = tok.decode(ids)
    assert "<SOS>" not in decoded
    assert "<EOS>" not in decoded
    assert "<PAD>" not in decoded
    assert decoded == "a cat"
