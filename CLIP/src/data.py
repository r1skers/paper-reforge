"""
M4.1 — CIFAR-10 dataset adapted for CLIP-style (image, caption) pair training.

为什么这一步用 CIFAR-10 + 模板而不是真实 caption 数据集
----------------------------------------------------
M4 是 pipeline sanity 检查, 不是真正的 retrieval. 目的:
    - 验证 forward 两塔 + InfoNCE loss 真的能反传
    - 验证 loss 真的会下降
    - 验证 similarity matrix 对角线真的会变亮

为达成这三件事, 我们不需要复杂数据 — 只需要每张图配一个语义对得上的 caption.
"a photo of a {class}" 这种模板正好够 toy-scale.

真正的 image-text retrieval 留给 M5 (Flickr8k 子集).

Templates 选择
-------------
CLIP 论文用了 80 个模板做 ImageNet zero-shot ensemble (论文 Appendix A.4).
我们 toy-scale 用 7 个 — 既能见到多样性, 又不至于训练时词表爆炸.

每张图在 __getitem__ 里 RANDOMLY 选一个模板, 所以训练 epoch 数变成
"image × template" 的隐式 augmentation. 同一张 dog 图, 可能这个 epoch 配
"a photo of a dog", 下一个 epoch 配 "a blurry photo of a dog" — 这给模型
看到了类名 "dog" 在不同 caption 上下文里的稳定性, 是 CLIP 学到判别性的关键.

Run
---
    python data.py
"""

import sys
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

# 让 tokenizer 可 import.
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from tokenizer import SimpleWordTokenizer


# ---------------------------------------------------------------------------- #
# Constants                                                                    #
# ---------------------------------------------------------------------------- #

# CIFAR-10 的 10 个 class name, 顺序 = torchvision label index 顺序.
# 来源: torchvision.datasets.CIFAR10.classes (英文 lowercase)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# 7 个 caption 模板 — 故意 mix 了 "一张普通照片" / "近照" / "黑白" / "低质量" 等
# 多种 caption 风格. CLIP 原版 80 个的精简子集.
# 每个模板用 "{}" 占位 class name.
TEMPLATES = [
    "a photo of a {}",
    "a blurry photo of a {}",
    "a black and white photo of a {}",
    "a low quality photo of a {}",
    "a close up of a {}",
    "a photo of one {}",
    "a small photo of a {}",
]

# CIFAR-10 standard normalization stats (跟 ViT 阶段保持一致).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# 数据缓存路径. 默认 CLIP/data/, 让 CLIP 完全 self-contained.
# 想复用 ViT 阶段已下载的 CIFAR-10? 给 CIFAR10ClipDataset(root=...) 显式传路径,
# 例如 root = Path("D:/Dev/repos/paper-reforge/ViT/data").
_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------- #
# Tokenizer factory                                                            #
# ---------------------------------------------------------------------------- #


def build_cifar_tokenizer() -> SimpleWordTokenizer:
    """
    构造一个 vocab 涵盖所有 (template × class) 组合的 tokenizer.

    用所有模板 × 所有 class 的实际句子做 build_vocab, 保证训练时不会出现 OOV.
    """
    # ------------------------------------------------------------------ #
    # TODO 1 — 枚举所有 (template × class) caption, build vocab.           #
    #                                                                     #
    #   tok = SimpleWordTokenizer()                                       #
    #   captions = [t.format(c) for t in TEMPLATES for c in CIFAR10_CLASSES] #
    #   tok.build_vocab(captions)                                          #
    #   return tok                                                         #
    #                                                                     #
    # 7 templates × 10 classes = 70 captions, 词表去重后约 20 个 unique word. #
    # ------------------------------------------------------------------ #
    tok = SimpleWordTokenizer()
    captions = [t.format(c) for t in TEMPLATES for c in CIFAR10_CLASSES]
    tok.build_vocab(captions)
    return tok

# ---------------------------------------------------------------------------- #
# Dataset                                                                      #
# ---------------------------------------------------------------------------- #


class CIFAR10ClipDataset(Dataset):
    """
    包装 torchvision CIFAR-10, 给每张图随机配一个模板 caption.

    Parameters
    ----------
    root        : 数据缓存路径
    train       : True = 训练集 (50k 张), False = 测试集 (10k 张)
    tokenizer   : 已 build_vocab 的 SimpleWordTokenizer
    max_len     : caption tokenize 的固定长度 (与 TextTransformer.max_len 一致)
    augment     : True 时训练集加 RandomCrop + HorizontalFlip
    seed        : random template 采样的可复现性. None 表示用 PyTorch 全局随机.

    __getitem__ 返回
    ---------------
    image     : (3, 32, 32) float tensor
    token_ids : (max_len,) long tensor
    eos_pos   : int — EOS 在 token_ids 里的位置

    DataLoader 默认 collate 会自动堆叠:
        images    : (B, 3, 32, 32)
        token_ids : (B, max_len)
        eos_pos   : (B,)
    所以本类不需要自定义 collate_fn.
    """

    def __init__(self, root: Path = _DATA_ROOT, train: bool = True,
                 tokenizer: SimpleWordTokenizer = None, max_len: int = 16,
                 augment: bool = True, seed: int = None):
        if tokenizer is None:
            raise ValueError("CIFAR10ClipDataset 必须传入 tokenizer (用 build_cifar_tokenizer())")

        self.tokenizer = tokenizer
        self.max_len = max_len

        # 图像 transform — 和 ViT 阶段一致.
        norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        if train and augment:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                norm,
            ])

        # 内部 wrap 一个 torchvision CIFAR10. 我们覆盖 __getitem__ 的返回结构.
        self.cifar = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True, transform=self.transform,
        )

        # 用一个独立 Generator, 保证 template 采样可复现 (训练时不同 epoch
        # 不同 seed, 但 eval 时固定 seed 让 visualize 能稳定).
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # ------------------------------------------------------------------ #
        # TODO 2 — 取出图和 label, 随机选一个模板, 拼成 caption, tokenize.       #
        #                                                                     #
        # Step 1: image, label = self.cifar[idx]                              #
        #         image: (3, 32, 32), label: int 0..9                          #
        #                                                                     #
        # Step 2: 随机选模板. 用 self._rng 保证可复现:                          #
        #   tpl_idx = int(torch.randint(0, len(TEMPLATES), (1,),                #
        #                                generator=self._rng).item())          #
        #   template = TEMPLATES[tpl_idx]                                       #
        #                                                                     #
        # Step 3: caption = template.format(CIFAR10_CLASSES[label])           #
        #                                                                     #
        # Step 4: token_ids, eos_pos = self.tokenizer.encode(caption, max_len) #
        #                                                                     #
        # Step 5: return image, token_ids, eos_pos                             #
        # ------------------------------------------------------------------ #
        image, label = self.cifar[idx]
        tpl_idx = int(torch.randint(0, len(TEMPLATES), (1,), generator=self._rng).item())
        template = TEMPLATES[tpl_idx]
        caption = template.format(CIFAR10_CLASSES[label])
        token_ids, eos_pos = self.tokenizer.encode(caption, self.max_len)
        return image, token_ids, eos_pos    


# ---------------------------------------------------------------------------- #
# DataLoader 工厂                                                              #
# ---------------------------------------------------------------------------- #


def get_cifar_clip_loaders(batch_size: int = 128, num_workers: int = 0,
                            max_len: int = 16, augment_train: bool = True):
    """
    一次性 build tokenizer + train/test loader. 这是 train_cifar.py 的 entry point.

    Returns
    -------
    tokenizer    : SimpleWordTokenizer
    train_loader : DataLoader, 每个 batch (images, token_ids, eos_pos)
    test_loader  : DataLoader, 同上
    """
    tokenizer = build_cifar_tokenizer()

    train_ds = CIFAR10ClipDataset(
        train=True, tokenizer=tokenizer, max_len=max_len,
        augment=augment_train,
    )
    test_ds = CIFAR10ClipDataset(
        train=False, tokenizer=tokenizer, max_len=max_len,
        augment=False, seed=0,  # eval 时固定 template 选择
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    return tokenizer, train_loader, test_loader


# ---------------------------------------------------------------------------- #
# Driver                                                                       #
# ---------------------------------------------------------------------------- #


def main():
    print("Building tokenizer from templates × classes ...")
    tokenizer = build_cifar_tokenizer()
    print(f"vocab_size = {tokenizer.vocab_size}    (3 special + ~20 unique words)")
    print(f"sample words: {list(tokenizer.word2id.keys())[:15]}")
    print()

    print("Loading CIFAR-10 (may download on first run) ...")
    train_ds = CIFAR10ClipDataset(train=True, tokenizer=tokenizer, max_len=16)
    print(f"len(train_ds) = {len(train_ds)}    (expected 50000)")
    print()

    # 看看几个 sample
    print("Sample batch contents:")
    for i in [0, 1, 2]:
        image, token_ids, eos_pos = train_ds[i]
        decoded = tokenizer.decode(token_ids)
        cifar_label = train_ds.cifar[i][1]
        print(f"  [{i}] class={CIFAR10_CLASSES[cifar_label]:12s} "
              f"caption='{decoded}'    eos_pos={eos_pos}    "
              f"image={tuple(image.shape)}")
    print()

    # 验证 DataLoader 出来的 batch shape
    print("DataLoader batch shape check:")
    _, loader, _ = get_cifar_clip_loaders(batch_size=4, max_len=16)
    images, token_ids, eos_pos = next(iter(loader))
    print(f"  images.shape    = {tuple(images.shape)}    (expected (4, 3, 32, 32))")
    print(f"  token_ids.shape = {tuple(token_ids.shape)}    (expected (4, 16))")
    print(f"  eos_pos.shape   = {tuple(eos_pos.shape)}     (expected (4,))")
    print(f"  eos_pos values  = {eos_pos.tolist()}")


if __name__ == "__main__":
    main()
