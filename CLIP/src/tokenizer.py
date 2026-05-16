"""
M1.1 — A deliberately minimal word-level tokenizer for CLIP toy experiments.

Why not BPE / HuggingFace tokenizer?
-----------------------------------
- CLIP原版用 BPE (49152 vocab), 但训练 BPE 自身需要工程量，对学习目标无帮助。
- 我们的 toy 数据 (CIFAR-10 模板 + 后期 Flickr8k 子集) 总词表 < 5000，
  word-level 完全够用，省去 subword 一切复杂度。
- 思想优先于工程：confirm causal attention + EOS feature 这套机制走通就够了。

Special tokens
--------------
    PAD = 0    # 占位，使 batch 内文本统一长度。不参与任何计算。
    SOS = 1    # Start-of-sequence, 句首 sentinel。在 CLIP 里没特殊用途，但保留对称。
    EOS = 2    # End-of-sequence. CRITICAL — 它是我们提取 sentence feature 的位置。
               # 训练时它就是普通词表里的一个 token, 不是 nn.Parameter。

encode 的输出
-------------
    token_ids : torch.LongTensor, shape (max_len,)
    eos_pos   : int — EOS token 所在的 index, 后面 forward 时
                x[arange(B), eos_pos] 用它取 sentence feature.

Run
---
    python tokenizer.py
"""

from typing import List, Tuple

import torch


# ---------------------------------------------------------------------------- #
# Special token ids (固定, 不要动)                                              #
# ---------------------------------------------------------------------------- #
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
_SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>"]


class SimpleWordTokenizer:
    """
    极简 word-level tokenizer。

    Parameters
    ----------
    lowercase : bool, default True
        是否将所有输入文本转小写。CLIP 原版 BPE 也是 case-insensitive 的。

    Workflow
    --------
        tok = SimpleWordTokenizer()
        tok.build_vocab(["a photo of a cat", "a photo of a dog", ...])
        ids, eos_pos = tok.encode("a photo of a cat", max_len=16)
        text = tok.decode(ids)
    """

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
        # word2id / id2word 在 build_vocab 时填充, 这里先放特殊 token.
        self.word2id = {tok: i for i, tok in enumerate(_SPECIAL_TOKENS)}
        self.id2word = {i: tok for i, tok in enumerate(_SPECIAL_TOKENS)}
        self.vocab_size = len(_SPECIAL_TOKENS)

    # ------------------------------------------------------------------------ #
    # Internal helper                                                          #
    # ------------------------------------------------------------------------ #
    def _split(self, text: str) -> List[str]:
        """
        把一行文本切成 word list. 极简: lowercase 然后 split() 按空白切.
        实际数据里的标点 (",", ".") 会和单词粘在一起 ("cat,"),
        本 toy 实现忽略这点 — 真要做 retrieval 再考虑.
        """
        if self.lowercase:
            text = text.lower()
        return text.split()

    # ------------------------------------------------------------------------ #
    # Public API                                                               #
    # ------------------------------------------------------------------------ #
    def build_vocab(self, texts: List[str]) -> None:
        """
        遍历语料把所有出现过的 word 加入 vocab.
        特殊 token (PAD/SOS/EOS) 已经在 __init__ 里占位 id 0/1/2.
        """
        # ------------------------------------------------------------------ #
        # TODO 1 — 遍历 texts, 对每个 text 调 self._split, 把没见过的 word    #
        # 加进 self.word2id 和 self.id2word, 同时更新 self.vocab_size.        #
        #                                                                     #
        # for text in texts:                                                  #
        #     for w in self._split(text):                                     #
        #         if w not in self.word2id:                                   #
        #             self.word2id[w] = self.vocab_size                       #
        #             self.id2word[self.vocab_size] = w                       #
        #             self.vocab_size += 1                                    #
        #                                                                     #
        # NOTE: 不需要 OOV / <UNK> token, 因为 toy 数据 vocab 是封闭的.        #
        # 真实场景 (Flickr8k 测试集出现训练集没见过的词) 时再加 <UNK>.         #
        # ------------------------------------------------------------------ #
        for text in texts:
            for w in self._split(text):
                if w not in self.word2id:
                    self.word2id[w] = self.vocab_size
                    self.id2word[self.vocab_size] = w
                    self.vocab_size += 1

    def encode(self, text: str, max_len: int) -> Tuple[torch.LongTensor, int]:
        """
        把一行文本编成 (token_ids, eos_pos).

        Layout:
            [SOS, w1, w2, ..., wK, EOS, PAD, PAD, ...]
                                    ^
                                    eos_pos

        Truncation rule:
            如果 K + 2 > max_len, 截掉末尾 words, 保证 EOS 一定在序列里.
            (EOS 必须存在, 否则 forward 时取不出 sentence feature.)
        """
        # ------------------------------------------------------------------ #
        # TODO 2 — 实现 encode.                                                #
        #                                                                     #
        # Step 1: words = self._split(text)                                   #
        # Step 2: word_ids = [self.word2id[w] for w in words]                 #
        #         (toy 假设 vocab 封闭, 真实代码要 fallback 到 <UNK>)         #
        # Step 3: 拼成 [SOS_ID] + word_ids + [EOS_ID]                         #
        # Step 4: 如果长度 > max_len, 截到 max_len - 1 个 word + EOS         #
        #         (具体: ids = [SOS] + word_ids[:max_len - 2] + [EOS])       #
        # Step 5: eos_pos = len(ids) - 1                                      #
        # Step 6: PAD 到 max_len: ids += [PAD_ID] * (max_len - len(ids))     #
        # Step 7: return torch.tensor(ids, dtype=torch.long), eos_pos        #
        #                                                                     #
        # CRITICAL: eos_pos 是 EOS 的 index, 不是 SOS 的位置 + len(words).    #
        # 截断后这两个不一样.                                                  #
        # ------------------------------------------------------------------ #
        words = self._split(text)
        word_ids = [self.word2id[w] for w in words]
        ids = [SOS_ID] + word_ids + [EOS_ID]

        if len(ids) > max_len:
            ids = ids[:max_len - 1] + [EOS_ID]

        eos_pos = len(ids) - 1
        ids += [PAD_ID] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), eos_pos
        # 提前设置了token容许长度，遍历原文本并填入到token列表中，最后返回token列表和eos位置，不够长度的部分用PAD_ID填充，超出容许的长度则截断并在末尾添加EOS_ID。

    def decode(self, ids) -> str:
        """
        反向: token_ids -> 文本. 跳过 PAD/SOS/EOS.
        ids 可以是 list / 1D tensor / numpy array.
        """
        # ------------------------------------------------------------------ #
        # TODO 3 — 把 ids 转成 Python list (兼容 tensor / list), 然后:          #
        #                                                                     #
        # words = []                                                          #
        # for i in ids:                                                       #
        #     i = int(i)                                                      #
        #     if i in (PAD_ID, SOS_ID, EOS_ID):                              #
        #         continue                                                    #
        #     words.append(self.id2word[i])                                   #
        # return " ".join(words)                                              #
        # ------------------------------------------------------------------ #
        words = []
        for i in ids:
            i = int(i)
            if i in (PAD_ID, SOS_ID, EOS_ID):
                continue
            words.append(self.id2word[i])
        return " ".join(words)


# ---------------------------------------------------------------------------- #
# Driver                                                                       #
# ---------------------------------------------------------------------------- #


def main():
    texts = [
        "a photo of a cat",
        "a photo of a dog",
        "a blurry image of a bird",
        "a painting of an airplane",
    ]
    tok = SimpleWordTokenizer()
    tok.build_vocab(texts)
    print(f"vocab_size = {tok.vocab_size}    (expected 3 special + 12 unique words = 15)")
    print(f"word2id    = {tok.word2id}")

    # 正常 encode
    ids, eos_pos = tok.encode("a photo of a cat", max_len=10)
    print()
    print(f"text       = 'a photo of a cat'")
    print(f"ids        = {ids.tolist()}")
    print(f"eos_pos    = {eos_pos}    (expected 6 — SOS + 5 words = positions 0..5, EOS at 6)")
    print(f"decode     = '{tok.decode(ids)}'")

    # 截断: max_len=5 时 'a photo of a cat' (5 words + SOS + EOS = 7 tokens) 必须裁
    ids2, eos_pos2 = tok.encode("a photo of a cat", max_len=5)
    print()
    print(f"max_len=5 truncation:")
    print(f"ids        = {ids2.tolist()}    (expected [SOS, w, w, w, EOS], len=5)")
    print(f"eos_pos    = {eos_pos2}    (expected 4 — EOS at last position)")


if __name__ == "__main__":
    main()
