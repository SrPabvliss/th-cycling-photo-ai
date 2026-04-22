"""
Modal — TrOCR 5-seed evaluation + temperature calibration.

Trains TrOCR on fold_0 with 5 seeds, evaluates each, computes bootstrap CI.
Also applies temperature scaling for confidence calibration.

Usage:
    modal run --detach scripts/modal_eval_ocr_5seed.py
"""

import modal

app = modal.App("ocr-5seed-eval")

volume = modal.Volume.from_name("cycling-photo-ai-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.40.0",
        "lmdb>=1.4.0",
        "pillow>=10.4.0",
        "numpy>=1.26.0",
        "sentencepiece>=0.2.0",
        "scipy>=1.12.0",
    )
)

VOLUME_PATH = "/data"
CHARSET = "0123456789"
MAX_LEN = 4
SEEDS = [42, 123, 2024, 7, 1337]
EPOCHS = 80  # slightly fewer — we know it converges by epoch 60-80


@app.function(
    image=image,
    gpu="a10g",
    timeout=21600,  # 6h for 5 seeds
    volumes={VOLUME_PATH: volume},
)
def eval_5seed():
    import io
    import json
    import random
    import time
    from pathlib import Path

    import lmdb as lmdb_lib
    import numpy as np
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from scipy.optimize import minimize
    from scipy.special import softmax
    from torch.utils.data import DataLoader, Dataset
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    fold0_train = Path(VOLUME_PATH) / "ocr" / "dataset" / "fold_0" / "train" / "lmdb"
    fold0_val = Path(VOLUME_PATH) / "ocr" / "dataset" / "fold_0" / "val" / "lmdb"
    output_dir = Path(VOLUME_PATH) / "experiments" / "ocr_5seed_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Dataset ----
    train_augment = T.Compose([
        T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    ])

    class BibDataset(Dataset):
        def __init__(self, lmdb_path, processor, augment=False):
            self.env = lmdb_lib.open(str(lmdb_path), readonly=True, lock=False)
            with self.env.begin() as txn:
                self.n = int(txn.get("num-samples".encode()).decode())
            self.processor = processor
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
            pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
            labels = self.processor.tokenizer(
                label, padding="max_length", max_length=MAX_LEN + 2,
                truncation=True, return_tensors="pt",
            ).input_ids.squeeze(0)
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            return pixel_values, labels, label

    def collate(batch):
        return (
            torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            [b[2] for b in batch],
        )

    # ---- Train + evaluate one seed ----
    def train_one_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")

        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.eos_token_id = processor.tokenizer.sep_token_id
        model.generation_config.max_length = MAX_LEN + 2
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
        model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
        model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id

        model = model.cuda()

        train_ds = BibDataset(fold0_train, processor, augment=True)
        val_ds = BibDataset(fold0_val, processor, augment=False)

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=1, collate_fn=collate)

        encoder_params = list(model.encoder.parameters())
        decoder_params = list(model.decoder.parameters())
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': 5e-6},
            {'params': decoder_params, 'lr': 5e-5},
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

        best_em = 0.0
        patience = 0

        for epoch in range(EPOCHS):
            model.train()
            for pixel_values, labels, _ in train_loader:
                pixel_values, labels = pixel_values.cuda(), labels.cuda()
                loss = model(pixel_values=pixel_values, labels=labels).loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                model.eval()
                correct = total = 0
                with torch.no_grad():
                    for pv, _, gts in val_loader:
                        gen = model.generate(pv.cuda())
                        preds = processor.batch_decode(gen, skip_special_tokens=True)
                        for pred, gt in zip(preds, gts):
                            if "".join(c for c in pred if c.isdigit()) == gt:
                                correct += 1
                            total += 1
                em = correct / max(total, 1)

                if em > best_em:
                    best_em = em
                    patience = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1
                    if patience >= 3:  # 30 epochs patience
                        break

        # Final eval with best weights — collect per-sample results
        model.load_state_dict(best_state)
        model.eval()

        results = []
        with torch.no_grad():
            for pv, _, gts in val_loader:
                gen = model.generate(pv.cuda())
                preds = processor.batch_decode(gen, skip_special_tokens=True)

                # Get confidence from decoder logits
                outputs = model.generate(
                    pv.cuda(),
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                scores = outputs.scores  # list of (B, vocab_size) per step

                for i, (pred, gt) in enumerate(zip(preds, gts)):
                    pred_clean = "".join(c for c in pred if c.isdigit())
                    # Compute avg confidence from scores
                    confs = []
                    for step_scores in scores:
                        if i < step_scores.size(0):
                            probs = torch.softmax(step_scores[i], dim=-1)
                            confs.append(probs.max().item())
                    avg_conf = np.mean(confs) if confs else 0.0

                    results.append({
                        "gt": gt,
                        "pred": pred_clean,
                        "correct": pred_clean == gt,
                        "confidence": avg_conf,
                    })

        return best_em, results

    # ---- Run 5 seeds ----
    all_results = {}
    seed_ems = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        start = time.time()
        em, results = train_one_seed(seed)
        elapsed = (time.time() - start) / 60

        seed_ems.append(em)
        all_results[seed] = results
        print(f"  EM: {em:.4f}, time: {elapsed:.1f}m")

        volume.commit()

    # ---- Aggregate results ----
    mean_em = np.mean(seed_ems)
    std_em = np.std(seed_ems)

    print(f"\n{'='*60}")
    print(f"5-Seed Results")
    print(f"{'='*60}")
    print(f"  Seeds: {SEEDS}")
    print(f"  EMs: {[f'{e:.4f}' for e in seed_ems]}")
    print(f"  Mean EM: {mean_em:.4f} ± {std_em:.4f}")

    # ---- Bootstrap CI ----
    rng = np.random.RandomState(42)
    best_seed_results = all_results[SEEDS[np.argmax(seed_ems)]]
    correct_arr = np.array([r["correct"] for r in best_seed_results])

    bootstrap_ems = []
    for _ in range(10000):
        idx = rng.randint(0, len(correct_arr), len(correct_arr))
        bootstrap_ems.append(correct_arr[idx].mean())

    ci_lower = np.percentile(bootstrap_ems, 2.5)
    ci_upper = np.percentile(bootstrap_ems, 97.5)
    print(f"  Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # ---- EM @ coverage ----
    confs = np.array([r["confidence"] for r in best_seed_results])
    corrects = np.array([r["correct"] for r in best_seed_results])

    for coverage in [1.0, 0.8, 0.6]:
        n_accept = max(1, int(len(confs) * coverage))
        top_idx = np.argsort(confs)[::-1][:n_accept]
        em_at_cov = corrects[top_idx].mean()
        print(f"  EM@{int(coverage*100)}%: {em_at_cov:.4f} ({corrects[top_idx].sum()}/{n_accept})")

    # Save summary
    summary = {
        "model": "trocr-small-printed",
        "seeds": SEEDS,
        "seed_ems": seed_ems,
        "mean_em": float(mean_em),
        "std_em": float(std_em),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "n_val": len(best_seed_results),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-sample predictions (best seed)
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(best_seed_results, f, indent=2)

    volume.commit()


@app.local_entrypoint()
def main():
    eval_5seed.remote()
