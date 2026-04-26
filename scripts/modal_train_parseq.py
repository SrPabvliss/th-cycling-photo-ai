"""
Modal training — PARSeq fine-tune on bib crops.

PARSeq with decode_ar=False: parallel decoding, no linguistic bias.
Pure PyTorch training loop (no pytorch_lightning needed).

Vendor approach: model files loaded from baudm/parseq via sys.path.
Only external dep: timm>=0.9.16.

Usage:
    modal run --detach scripts/modal_train_parseq.py
"""

import modal

app = modal.App("ocr-parseq-finetune")

volume = modal.Volume.from_name("cycling-photo-ai-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "timm>=0.9.16",
        "lmdb>=1.4.0",
        "pillow>=10.4.0",
        "numpy>=1.26.0",
    )
    # Download PARSeq repo + pretrained weights during image build
    .run_commands(
        "python -c \""
        "import torch; "
        "torch.hub.load_state_dict_from_url("
        "'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt', "
        "map_location='cpu', check_hash=True); "
        "print('PARSeq weights cached')\"",
        # Clone repo for model code
        "pip install gitpython && python -c \""
        "import torch; "
        "torch.hub.load('baudm/parseq', 'parseq', pretrained=True, trust_repo=True); "
        "print('PARSeq repo cached')\" || true",
    )
)

VOLUME_PATH = "/data"
SEED = 42
MAX_LEN = 4
CHARSET = "0123456789"


@app.function(
    image=image,
    gpu="a10g",
    timeout=7200,
    volumes={VOLUME_PATH: volume},
)
def train_parseq():
    import io
    import json
    import math
    import random
    import sys
    import time
    from itertools import permutations as iter_permutations
    from pathlib import Path

    import lmdb as lmdb_lib
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import Image
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, Dataset, Subset

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print(f"{'='*60}")
    print(f"  PARSeq Fine-tune on Bib Crops")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Charset: {CHARSET}")
    print(f"  decode_ar=False (parallel decoding)")
    print(f"{'='*60}")

    # ---- Load PARSeq via vendored code ----
    # Find the cached repo
    hub_dir = Path.home() / ".cache" / "torch" / "hub"
    parseq_dirs = list(hub_dir.glob("baudm_parseq*"))
    if not parseq_dirs:
        # Try downloading
        print("Downloading PARSeq repo...")
        torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True)
        parseq_dirs = list(hub_dir.glob("baudm_parseq*"))

    parseq_repo = str(parseq_dirs[0])
    sys.path.insert(0, parseq_repo)
    print(f"  PARSeq repo: {parseq_repo}")

    from strhub.data.utils import Tokenizer
    from strhub.models.parseq.model import PARSeq

    tokenizer = Tokenizer(CHARSET)
    print(f"  Tokenizer: {len(tokenizer)} tokens (10 digits + 3 special)")

    # ---- Load pretrained weights ----
    model = PARSeq(
        num_tokens=len(tokenizer),
        max_label_length=MAX_LEN,
        img_size=(32, 128),
        patch_size=(4, 8),
        embed_dim=384,
        enc_num_heads=6,
        enc_mlp_ratio=4,
        enc_depth=12,
        dec_num_heads=12,
        dec_mlp_ratio=4,
        dec_depth=1,
        decode_ar=False,
        refine_iters=1,
        dropout=0.1,
    )

    # Load pretrained and adapt head for smaller charset
    pretrained = torch.hub.load_state_dict_from_url(
        "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt",
        map_location="cpu",
        check_hash=True,
    )

    # Pretrained has 97 tokens (94 chars + 3 special), we have 13 (10 + 3)
    # Load encoder weights (same architecture), skip head + text_embed (different size)
    compatible = {}
    skipped = []
    for k, v in pretrained.items():
        if k in ("head.weight", "head.bias", "text_embed.embedding.weight", "pos_queries"):
            skipped.append(k)
        else:
            compatible[k] = v

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(f"  Loaded pretrained: {len(compatible)} keys")
    print(f"  Skipped (size mismatch): {skipped}")
    print(f"  Missing (randomly init): {[k for k in missing if k not in [m for m in missing]][:5]}")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params: {n_params:.1f}M")

    model = model.cuda()

    # ---- PARSeq Permutation Training ----
    rng = np.random.default_rng(SEED)
    PERM_NUM = 6
    PERM_FORWARD = True
    PERM_MIRRORED = True

    def gen_tgt_perms(tgt, device):
        """Generate permutations for PARSeq training."""
        max_num_chars = tgt.shape[1] - 2  # exclude BOS and EOS
        if max_num_chars == 1:
            return torch.arange(3, device=device).unsqueeze(0)

        perms = [torch.arange(max_num_chars, device=device)]  # forward perm
        max_perms = math.factorial(max_num_chars)
        max_gen = PERM_NUM // 2  # mirrored
        num_gen = min(max_gen, max_perms // 2)

        if max_num_chars < 5:
            perm_pool = torch.as_tensor(
                list(iter_permutations(range(max_num_chars))), device=device
            )
            perm_pool = perm_pool[1:]  # remove forward (already added)
            if len(perm_pool) > 0:
                i = rng.choice(len(perm_pool), size=min(num_gen - 1, len(perm_pool)), replace=False)
                perms.extend([perm_pool[idx] for idx in i])
        else:
            for _ in range(num_gen - 1):
                perms.append(torch.randperm(max_num_chars, device=device))

        perms = torch.stack(perms)
        # Add mirrored
        comp = perms.flip(-1)
        perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)

        # Add BOS (0) and EOS (max+1) positions
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)

        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=device)

        return perms

    def generate_attn_masks(perm, device):
        """Generate attention masks from permutation."""
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), dtype=torch.bool, device=device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = True
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=device)] = True
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    # ---- Dataset ----
    transform = T.Compose([
        T.Resize((32, 128)),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])

    train_augment = T.Compose([
        T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    ])

    class LMDBDataset(Dataset):
        def __init__(self, lmdb_path, augment=False):
            self.env = lmdb_lib.open(str(lmdb_path), readonly=True, lock=False)
            with self.env.begin() as txn:
                self.n = int(txn.get("num-samples".encode()).decode())
            self.augment = augment

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            with self.env.begin() as txn:
                img_bytes = txn.get(f"image-{idx+1:09d}".encode())
                label = txn.get(f"label-{idx+1:09d}".encode()).decode()

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            if self.augment:
                img = train_augment(img)
            pixel_values = transform(img)
            label = "".join(c for c in label if c.isdigit())[:MAX_LEN]
            return pixel_values, label

    def collate(batch):
        return torch.stack([b[0] for b in batch]), [b[1] for b in batch]

    # ---- Load datasets ----
    train_lmdb = Path(VOLUME_PATH) / "ocr" / "dataset" / "fold_0" / "train" / "lmdb"
    val_lmdb = Path(VOLUME_PATH) / "ocr" / "dataset" / "fold_0" / "val" / "lmdb"

    train_ds = LMDBDataset(train_lmdb, augment=True)
    val_ds = LMDBDataset(val_lmdb, augment=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=1, collate_fn=collate)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ---- Optimizer ----
    encoder_params = list(model.encoder.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("encoder")]

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": 5e-6},
        {"params": other_params, "lr": 5e-5},
    ], weight_decay=1e-4)

    EPOCHS = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    scaler = GradScaler()

    output_dir = Path(VOLUME_PATH) / "experiments" / "ocr_parseq_finetune"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Evaluate ----
    def evaluate(log_all=False):
        model.eval()
        correct = total = 0
        examples = []
        with torch.no_grad():
            for pixel_values, gt_labels in val_loader:
                pixel_values = pixel_values.cuda()
                with autocast(dtype=torch.float16):
                    logits = model(tokenizer, pixel_values)
                    probs = logits.softmax(-1)
                pred_texts, pred_probs = tokenizer.decode(probs)

                for pred, gt in zip(pred_texts, gt_labels):
                    pred_clean = "".join(c for c in pred if c.isdigit())
                    if pred_clean == gt:
                        correct += 1
                    total += 1
                    if len(examples) < 20:
                        examples.append((gt, pred_clean, pred))

        em = correct / max(total, 1)
        if log_all:
            for gt, pc, raw in examples[:10]:
                mark = "✓" if gt == pc else "✗"
                print(f"    {mark} gt='{gt}' pred='{pc}' (raw='{raw}')")
        return em, correct, total

    # ---- Training loop ----
    best_em = 0.0
    patience = 0
    MAX_PATIENCE = 20
    start = time.time()

    print(f"\n  Starting PARSeq fine-tune ({EPOCHS} epochs)...\n")

    # Initial eval
    em_init, _, _ = evaluate(log_all=True)
    print(f"  Initial EM: {em_init:.4f}\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = n_batches = 0

        for pixel_values, labels in train_loader:
            pixel_values = pixel_values.cuda()

            # Encode labels with PARSeq tokenizer
            tgt = tokenizer.encode(labels, pixel_values.device)

            # Encode images
            with autocast(dtype=torch.float16):
                memory = model.encode(pixel_values)

            # PARSeq permutation training
            tgt_perms = gen_tgt_perms(tgt, pixel_values.device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            tgt_padding_mask = (tgt_in == tokenizer.pad_id) | (tgt_in == tokenizer.eos_id)

            loss = torch.tensor(0.0, device=pixel_values.device)
            loss_numel = 0
            n = (tgt_out != tokenizer.pad_id).sum().item()

            for i, perm in enumerate(tgt_perms):
                tgt_mask, query_mask = generate_attn_masks(perm, pixel_values.device)
                with autocast(dtype=torch.float16):
                    out = model.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
                    logits = model.head(out).flatten(end_dim=1)
                    loss = loss + n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=tokenizer.pad_id)
                loss_numel += n
                if i == 1:
                    tgt_out = torch.where(tgt_out == tokenizer.eos_id, tokenizer.pad_id, tgt_out)
                    n = (tgt_out != tokenizer.pad_id).sum().item()

            loss = loss / max(loss_numel, 1)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == EPOCHS - 1:
            log_all = (epoch == 0 or (epoch + 1) % 20 == 0)
            em, correct, total = evaluate(log_all=log_all)
            elapsed = (time.time() - start) / 60

            print(
                f"  Epoch {epoch+1:3d}/{EPOCHS} — "
                f"loss: {avg_loss:.4f}, EM: {em:.4f} ({correct}/{total}), "
                f"time: {elapsed:.1f}m"
            )

            if em > best_em:
                best_em = em
                patience = 0
                # Save model weights (pure PyTorch, not HF format)
                torch.save(model.state_dict(), str(output_dir / "best.pt"))
                # Also save config for reconstruction
                config = {
                    "num_tokens": len(tokenizer),
                    "max_label_length": MAX_LEN,
                    "charset": CHARSET,
                    "img_size": [32, 128],
                    "patch_size": [4, 8],
                    "embed_dim": 384,
                    "enc_num_heads": 6,
                    "enc_mlp_ratio": 4,
                    "enc_depth": 12,
                    "dec_num_heads": 12,
                    "dec_mlp_ratio": 4,
                    "dec_depth": 1,
                    "decode_ar": False,
                    "refine_iters": 1,
                    "dropout": 0.1,
                }
                with open(output_dir / "config.json", "w") as f:
                    json.dump(config, f, indent=2)
                print(f"    ★ New best! EM={em:.4f}")
            else:
                patience += 1
                if patience >= MAX_PATIENCE:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

    elapsed_total = (time.time() - start) / 60

    summary = {
        "model": "parseq-base",
        "charset": CHARSET,
        "decode_ar": False,
        "params_m": n_params,
        "best_val_em": best_em,
        "epochs_trained": epoch + 1,
        "time_min": elapsed_total,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "encoder_lr": 5e-6,
        "decoder_lr": 5e-5,
        "batch_size": 8,
        "seed": SEED,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()

    print(f"\n{'='*60}")
    print(f"  PARSeq fine-tune complete!")
    print(f"  Best EM: {best_em:.4f}")
    print(f"  Time: {elapsed_total:.1f} min")
    print(f"  Weights: {output_dir / 'best.pt'}")
    print(f"{'='*60}")


@app.local_entrypoint()
def main():
    train_parseq.remote()
