"""
Modal training — PARSeq 4-phase pretraining pipeline.

Phase 1: 50K synthetic (sport fonts + fabric backgrounds)
Phase 2: 50K SVHN real digits
Phase 4: 444 custom bib crops (discriminative LR)

Same pipeline as TrOCR 4-phase for fair head-to-head comparison.

Usage:
    modal run --detach scripts/modal_train_parseq_4phase.py --phase 1
    modal run --detach scripts/modal_train_parseq_4phase.py --phase 2
    modal run --detach scripts/modal_train_parseq_4phase.py --phase 4
"""

import modal

app = modal.App("ocr-parseq-4phase")

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
    .run_commands(
        "python -c \""
        "import torch; "
        "torch.hub.load_state_dict_from_url("
        "'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt', "
        "map_location='cpu', check_hash=True); "
        "print('PARSeq weights cached')\"",
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

PHASE_CONFIGS = {
    1: {
        "name": "phase1_synthetic",
        "base_weights": "__pretrained__",
        "train_lmdb": "ocr/synthetic/lmdb",
        "val_lmdb": "ocr/synthetic/lmdb",
        "encoder_lr": 5e-6,
        "decoder_lr": 5e-5,
        "epochs": 15,
        "batch_size": 64,
        "patience": 8,
        "augment": False,
        "max_train_samples": 45000,
        "max_val_samples": 5000,
    },
    2: {
        "name": "phase2_svhn",
        "base_weights": "__phase1__",
        "train_lmdb": "ocr/svhn/lmdb",
        "val_lmdb": "ocr/svhn/lmdb",
        "encoder_lr": 5e-6,
        "decoder_lr": 5e-5,
        "epochs": 10,
        "batch_size": 64,
        "patience": 6,
        "augment": False,
        "max_train_samples": 45000,
        "max_val_samples": 5000,
    },
    4: {
        "name": "phase4_finetune",
        "base_weights": "__phase2__",
        "train_lmdb": "ocr/dataset/fold_0/train/lmdb",
        "val_lmdb": "ocr/dataset/fold_0/val/lmdb",
        "encoder_lr": 5e-6,
        "decoder_lr": 5e-5,
        "epochs": 100,
        "batch_size": 8,
        "patience": 20,
        "augment": True,
        "max_train_samples": None,
        "max_val_samples": None,
    },
}


@app.function(
    image=image,
    gpu="a10g",
    timeout=7200,
    volumes={VOLUME_PATH: volume},
)
def train_phase(phase: int):
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

    cfg = PHASE_CONFIGS[phase]
    print(f"{'='*60}")
    print(f"  PHASE {phase}: {cfg['name']} (PARSeq)")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}")

    # ---- Load PARSeq ----
    hub_dir = Path.home() / ".cache" / "torch" / "hub"
    parseq_dirs = list(hub_dir.glob("baudm_parseq*"))
    if not parseq_dirs:
        torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True)
        parseq_dirs = list(hub_dir.glob("baudm_parseq*"))
    sys.path.insert(0, str(parseq_dirs[0]))

    from strhub.data.utils import Tokenizer
    from strhub.models.parseq.model import PARSeq

    tokenizer = Tokenizer(CHARSET)

    base_dir = Path(VOLUME_PATH) / "experiments" / "ocr_parseq_4phase"

    model = PARSeq(
        num_tokens=len(tokenizer), max_label_length=MAX_LEN,
        img_size=(32, 128), patch_size=(4, 8), embed_dim=384,
        enc_num_heads=6, enc_mlp_ratio=4, enc_depth=12,
        dec_num_heads=12, dec_mlp_ratio=4, dec_depth=1,
        decode_ar=False, refine_iters=1, dropout=0.1,
    )

    # ---- Load weights ----
    if cfg["base_weights"] == "__pretrained__":
        pretrained = torch.hub.load_state_dict_from_url(
            "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt",
            map_location="cpu", check_hash=True,
        )
        compatible = {k: v for k, v in pretrained.items()
                      if k not in ("head.weight", "head.bias", "text_embed.embedding.weight", "pos_queries")}
        model.load_state_dict(compatible, strict=False)
        print(f"  Loaded pretrained encoder ({len(compatible)} keys)")
    elif cfg["base_weights"] == "__phase1__":
        p = base_dir / "phase1" / "best.pt"
        if not p.exists():
            print(f"ERROR: Phase 1 weights not found at {p}")
            return
        model.load_state_dict(torch.load(str(p), map_location="cpu"))
        print(f"  Loaded Phase 1 weights")
    elif cfg["base_weights"] == "__phase2__":
        p = base_dir / "phase2" / "best.pt"
        if not p.exists():
            print(f"ERROR: Phase 2 weights not found at {p}")
            return
        model.load_state_dict(torch.load(str(p), map_location="cpu"))
        print(f"  Loaded Phase 2 weights")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params: {n_params:.1f}M")
    model = model.cuda()

    # ---- PARSeq permutation helpers ----
    rng = np.random.default_rng(SEED)

    def gen_tgt_perms(tgt, device):
        max_num_chars = tgt.shape[1] - 2
        if max_num_chars == 1:
            return torch.arange(3, device=device).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=device)]
        max_perms = math.factorial(max_num_chars)
        num_gen = min(3, max_perms // 2)
        if max_num_chars < 5:
            perm_pool = torch.as_tensor(list(iter_permutations(range(max_num_chars))), device=device)[1:]
            if len(perm_pool) > 0:
                i = rng.choice(len(perm_pool), size=min(num_gen - 1, len(perm_pool)), replace=False)
                perms.extend([perm_pool[idx] for idx in i])
        else:
            for _ in range(num_gen - 1):
                perms.append(torch.randperm(max_num_chars, device=device))
        perms = torch.stack(perms)
        comp = perms.flip(-1)
        perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=device)
        return perms

    def generate_attn_masks(perm, device):
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
    transform = T.Compose([T.Resize((32, 128)), T.ToTensor(), T.Normalize(0.5, 0.5)])
    train_augment = T.Compose([
        T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    ]) if cfg["augment"] else None

    class LMDBDataset(Dataset):
        def __init__(self, lmdb_path, augment_fn=None):
            self.env = lmdb_lib.open(str(lmdb_path), readonly=True, lock=False)
            with self.env.begin() as txn:
                self.n = int(txn.get("num-samples".encode()).decode())
            self.augment_fn = augment_fn

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            with self.env.begin() as txn:
                img_bytes = txn.get(f"image-{idx+1:09d}".encode())
                label = txn.get(f"label-{idx+1:09d}".encode()).decode()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            if self.augment_fn:
                img = self.augment_fn(img)
            pixel_values = transform(img)
            label = "".join(c for c in label if c.isdigit())[:MAX_LEN]
            return pixel_values, label

    def collate(batch):
        return torch.stack([b[0] for b in batch]), [b[1] for b in batch]

    train_lmdb = Path(VOLUME_PATH) / cfg["train_lmdb"]
    val_lmdb = Path(VOLUME_PATH) / cfg["val_lmdb"]
    if not train_lmdb.exists():
        print(f"ERROR: Train LMDB not found at {train_lmdb}")
        return

    full_train_ds = LMDBDataset(train_lmdb, augment_fn=train_augment)
    same_lmdb = cfg["train_lmdb"] == cfg["val_lmdb"]
    max_train = cfg.get("max_train_samples")
    max_val = cfg.get("max_val_samples")

    if same_lmdb and max_train and max_val:
        all_indices = list(range(len(full_train_ds)))
        random.shuffle(all_indices)
        train_ds = Subset(full_train_ds, all_indices[:max_train])
        val_ds = Subset(full_train_ds, all_indices[max_train:max_train + max_val])
        print(f"  Same LMDB split: {len(train_ds)} train / {len(val_ds)} val")
    else:
        val_ds = LMDBDataset(val_lmdb)
        if max_train and len(full_train_ds) > max_train:
            indices = list(range(len(full_train_ds)))
            random.shuffle(indices)
            train_ds = Subset(full_train_ds, indices[:max_train])
        else:
            train_ds = full_train_ds

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4,
                              collate_fn=collate, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"  Batches/epoch: {len(train_loader)}")

    # ---- Optimizer ----
    encoder_params = list(model.encoder.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": cfg["encoder_lr"]},
        {"params": other_params, "lr": cfg["decoder_lr"]},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-7)
    scaler = GradScaler()

    output_dir = base_dir / f"phase{phase}"
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
                pred_texts, _ = tokenizer.decode(probs)
                for pred, gt in zip(pred_texts, gt_labels):
                    pred_clean = "".join(c for c in pred if c.isdigit())
                    if pred_clean == gt: correct += 1
                    total += 1
                    if len(examples) < 20: examples.append((gt, pred_clean, pred))
        em = correct / max(total, 1)
        if log_all:
            for gt, pc, raw in examples[:10]:
                mark = "✓" if gt == pc else "✗"
                print(f"    {mark} gt='{gt}' pred='{pc}' (raw='{raw}')")
        return em, correct, total

    # ---- Training ----
    best_em = 0.0
    patience_counter = 0
    start = time.time()
    print(f"\n  Starting phase {phase} ({cfg['epochs']} epochs)...\n")

    em_init, _, _ = evaluate(log_all=True)
    print(f"  Initial EM: {em_init:.4f}\n")

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = n_batches = 0

        for pixel_values, labels in train_loader:
            pixel_values = pixel_values.cuda()
            tgt = tokenizer.encode(labels, pixel_values.device)

            with autocast(dtype=torch.float16):
                memory = model.encode(pixel_values)

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

            if n_batches % 200 == 0:
                elapsed = (time.time() - start) / 60
                print(f"    batch {n_batches}/{len(train_loader)} — loss: {total_loss/n_batches:.4f}, time: {elapsed:.1f}m")

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        eval_freq = 2 if phase in (1, 2) else 5
        if (epoch + 1) % eval_freq == 0 or epoch == 0 or epoch == cfg["epochs"] - 1:
            log_all = (epoch == 0 or (epoch + 1) % 10 == 0)
            em, correct, total = evaluate(log_all=log_all)
            elapsed = (time.time() - start) / 60
            print(f"  Epoch {epoch+1:3d}/{cfg['epochs']} — loss: {avg_loss:.4f}, EM: {em:.4f} ({correct}/{total}), time: {elapsed:.1f}m")

            if em > best_em:
                best_em = em
                patience_counter = 0
                torch.save(model.state_dict(), str(output_dir / "best.pt"))
                config = {
                    "num_tokens": len(tokenizer), "max_label_length": MAX_LEN,
                    "charset": CHARSET, "img_size": [32, 128], "patch_size": [4, 8],
                    "embed_dim": 384, "enc_num_heads": 6, "enc_mlp_ratio": 4, "enc_depth": 12,
                    "dec_num_heads": 12, "dec_mlp_ratio": 4, "dec_depth": 1,
                    "decode_ar": False, "refine_iters": 1, "dropout": 0.1,
                }
                with open(output_dir / "config.json", "w") as f:
                    json.dump(config, f, indent=2)
                print(f"    ★ New best! EM={em:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

    elapsed_total = (time.time() - start) / 60
    summary = {
        "phase": phase, "name": cfg["name"], "model": "parseq-base",
        "charset": CHARSET, "decode_ar": False, "params_m": n_params,
        "best_val_em": best_em, "epochs_trained": epoch + 1,
        "time_min": elapsed_total, "train_samples": len(train_ds),
        "val_samples": len(val_ds), "seed": SEED,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    volume.commit()

    print(f"\n{'='*60}")
    print(f"  Phase {phase} complete! Best EM: {best_em:.4f}, Time: {elapsed_total:.1f}m")
    print(f"{'='*60}")


@app.local_entrypoint()
def main(phase: int = 1):
    train_phase.remote(phase)
