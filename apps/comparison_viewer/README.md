# Comparison Viewer — Banco de pruebas visual

Internal Streamlit app to compare 4 detectors × 10 OCR systems × 2 color systems on 60 exploratory cycling photos. Outputs cached `CallRecord` JSONs + judgment JSONLs feeding an offline academic notebook.

## Setup

1. Install deps: `uv sync` (extra deps required: `streamlit`, `pyarrow`, `pytest-asyncio`).
2. Copy `.env.example` to `.env`, fill in API keys + snapshots.
3. Place 60 images in `data/exploratorio/images/`, optionally write `data/exploratorio/groups.yaml`.
4. Build manifest:
   ```
   python apps/comparison_viewer/scripts/build_manifest.py \
     --images-dir data/exploratorio/images \
     --groups-yaml data/exploratorio/groups.yaml \
     --output data/exploratorio/manifest.json
   ```

## Smoke test (mandatory before real session)

```
python apps/comparison_viewer/scripts/smoke_test_all.py \
  --image data/exploratorio/images/IMG_4520.jpg
```

Verifies all 16 systems work end-to-end on 1 image. Total cost: ~$0.05 USD.

## Run the app

```
streamlit run apps/comparison_viewer/streamlit_app.py
```

## Export consolidated artifacts (post-session)

```
python apps/comparison_viewer/scripts/export_for_analysis.py \
  --experiments-dir experiments/exploratorio \
  --judgments-dir judgments \
  --manifest-path data/exploratorio/manifest.json \
  --output-dir experiments/exploratorio/consolidated
```

## Decisiones de implementación

### Streamlit + asyncio integration

**Pattern:** `threading.Thread` daemon running its own `asyncio.new_event_loop()`, communicating via `queue.Queue` polled by `st.empty()` placeholder.

- **Why:** `streamlit-asyncio` library not battle-tested as of 2026-05. Thread-based approach (`run_async_in_thread()` + `drain_queue_events()` in `components/live_progress.py`) proven in local testing.
- **Reuse:** Same async-in-thread helpers invoked by all view tabs (Detection, OCR, Color) and smoke test runner.
- **Event types:** Four kinds streamed from `pipeline_runner.run_stage()`:
  - `"started"` → system execution begins (no CallRecord yet)
  - `"cached"` → result from disk cache (CallRecord populated)
  - `"done"` → execution succeeded (CallRecord with tokens/cost)
  - `"error"` → execution or timeout failed (CallRecord with error_category)

### Canonical prompts & versioning

Each VLM/color reader accepts `prompt_override` kwarg (wired at call function level, commits 0115c1f–a960ec7):

- **OCR canonical:** `prompts/ocr_canonical_v1.py` → `SEMANTIC_CONTENT` (semantic intent) + language-specific templates for Anthropic (XML), Gemini (JSON), OpenAI (system+user split).
  - Semantic SHA256: persisted in `CallRecord.prompt_sha256` for academic reproducibility.
  - Cite as: "Prompt OCR canónico v1 (sha256: {SEMANTIC_SHA256[:12]}...)"

- **Color canonical:** `prompts/color_canonical_v1.py` → similar semantic + per-provider templates.
  - Applies to `gemini_2_5_flash_color` only (local K-Means has no prompt).

- **Prompt registry:** `prompts/prompts_registry.json` auto-generated with all versioned prompts + checksums for audit trail.

### Prompt caching

Enabled platform-by-platform:

- **Anthropic:** `cache_control=ephemeral` on system block (Anthropic docs). ClaudeVlmReader sets automatically when `enable_prompt_caching=True`.
- **Gemini:** `enable_prompt_caching=True` → system_instruction implicit caching on Gemini 2.5+.
- **OpenAI:** Relies on server-side automatic cache (no explicit control in this SDK version).
- **Local models:** N/A (CPU inference, no network cache).

### Token surfacing & usage metadata

All VLM/cloud readers expose `_last_usage` dict (keys: `input_tokens`, `output_tokens`, `cached_input_tokens`, `thinking_tokens`). Call functions (`adapters/calls/*.py`) read via `_normalized_usage(reader)` and forward to `CallRecord`:

- **Claude N=3:** Multi-sample tokens summed internally by `ClaudeVlmReader` (production-tuned per ADR-011). Single value in CallRecord.
- **Gemini:** Per-call tokens populated (thinking tokens billed at output rate for Gemini 3 Pro).
- **OpenAI:** Per-call tokens + cache hits tracked.
- **Google Vision / AWS Rekognition:** Token fields omitted (per-image unit pricing, no token-level granularity).
- **Local models (YOLO, RF-DETR, PARSeq, TrOCR, K-Means):** Token fields omitted (CPU, no API metering).

Request ID (`CallRecord.request_id`):
- Read from `reader._last_request_id` when exposed by SDK (e.g., AWS Rekognition ✓, Gemini sometimes ✓, Anthropic ✓).
- Google Vision: unavailable from SDK (field stays None).

### Pricing snapshot & caching

**Snapshot date:** 2026-05-03 (hardcoded in `config/pricing.yaml`).

Unit: tokens (VLMs), per-call (Roboflow), per-1k-images (Google Vision, AWS Rekognition).

**Rates verified vs. official sources:**
- ✓ Claude Opus 4.7: corrected from PLAN (old claude-opus-4-1 $15/$75 → actual $5/$25).
- ✓ Gemini 2.5 Flash: corrected from PLAN ($0.075 input → actual $0.30).
- ✓ Gemini 2.5 Pro: output corrected $5.00 → $10.00 (≤200k context tier).
- ⚠️ Gemini 3 Pro: model name unconfirmed on Google pricing page; rates from PLAN template — verify before use.
- ⚠️ GPT-5 / GPT-4o-mini: OpenAI pricing page returned 403; rates from PLAN — verify at https://openai.com/api/pricing.
- ⚠️ Roboflow per-inference: exact rate not published (credits system); $0.001 estimate only.

Cost calculation (TimedWrapper.run):
- Token-based: `calculate_cost_tokens(pricing, input_tokens, output_tokens, cached_input_tokens, thinking_tokens)`.
- Per-call: `calculate_cost_per_call(pricing)`.

### Cache strategy

Cache keys derived from content SHA256s:

- **Detection:** key = `image_sha256` (full input image). Local models (YOLO, RF-DETR), Roboflow, Gemini Detection.
- **OCR:** key = `crop_sha256` (PNG-encoded bib region, stable across formats). Parent crop passed to call function; CallRecord tracks `parent_crop_sha256`.
- **Color:** key = `crop_sha256 + region` (e.g., `{crop_sha256}_helmet`). Per-region analysis within single crop.

Cache storage: `experiments/exploratorio/{domain}/{system_id}/raw/{cache_key}.json` containing full `CallRecord`.

### Concurrency & timeout

- **Sequential (default):** Systems run one-at-a-time, safe for Streamlit state mutation.
- **Parallel mode (opt-in UI toggle):** `asyncio.Semaphore(8)` limits global concurrency. Roboflow capped at `Semaphore(2)` (rate-limit protection per registry).
- **Per-call timeout:** 30 seconds (configurable via `settings.per_call_timeout_s`).
- **Retries:** 3 attempts max on retryable error categories only (network, timeout, rate limit). Exponential backoff base=2s: wait 2s after attempt 1, 4s after attempt 2, no wait before attempt 3.

### Local models (M4 Pro macOS)

**YOLO11m, RF-DETR-M, PARSeq, TrOCR, K-Means:** forced CPU per `RUN_CONDITIONS.md`.

- YoloDetector & RfdetrDetector do NOT accept `device=` kwarg (ultralytics constructor limit). Both auto-select CPU on macOS (no CUDA). This matches the requirement.
- PARSeq & TrOCR: device selection deferred to transformer library defaults (CPU on M4 Pro).
- K-Means: scikit-learn CPU-only.

**Latency lower bound:** M4 Pro measurements are 3–5× faster than VPS production (Hetzner CPX31 4vCPU).

### Judgment persistence

Judgments stored as JSONL append-only (each image = 1 JSON object per line):

```json
{
  "image_id": "IMG_4520.jpg",
  "region": "cyclist_clothes",  # null for detection
  "submitted_at": "2026-05-03T14:23:01Z",
  "editor": "pablo",
  "system_ids": ["parseq_base", "claude_opus_4_7"],
  "judgment": "CORRECT",  # CORRECT | INCORRECT | AMBIGUOUS
  "notes": "Bib was clearly visible"
}
```

Last-write-wins when same `(image_id, region, system_ids)` tuple re-judged. Consensus tab calculates agreement %; outlier detection via per-system mismatch rate.

## Concerns / known gaps

### Google Vision request_id

Request IDs are unavailable from the Google Vision Python SDK (as tested 2026-05-03). `CallRecord.request_id` stays `None` for Google Vision calls. Workaround: use AWS Rekognition (request IDs populated) or Gemini (request IDs exposed).

### Detection Gemini usage_metadata

`GeminiDetector` now exposes `_last_usage` (commit dc3079c) with token counts. However, `request_id` may remain `None` depending on SDK version exposure. Verify via `_last_request_id` at runtime.

### OpenAI prompt_override limitation

The `prompt_override` kwarg in OpenAI readers replaces only the **user prompt** (semantic content). The underlying reader's **system prompt** (hardcoded instruction template) remains in place. Semantic content of overridden user portion matches canonical intent, so reproducibility is preserved for the variable portion. If exact system prompt reproducibility is required, coordinate upstream (cycling_photo_ai) to expose full system template override.

## Limitations

- No formal ground truth on the 60 images (pseudo-GT via consensus + test-retest reliability score).
- Latencies measured on M4 Pro CPU are lower bounds for VPS prod (~3–5× slower on Hetzner CPX31).
- Single evaluator judgment (only Pablo); discrete taxonomy (CORRECT/INCORRECT/AMBIGUOUS) mitigates subjectivity.
- Streamlit UI has no automated tests; functional smoke test (`smoke_test_all.py`) verifies all 16 systems locally, but integration testing with actual images happens in live session.
- Prompt caching verification: OpenAI does not expose cache hit counts via SDK; Anthropic + Gemini caching assumed working per platform docs.

## Citation

If citing in thesis, reference:
- Prompts: `prompts/PROMPTS.md` + `prompts/{ocr,color}_canonical_v1.py` (include semantic SHA256 in appendix).
- Run conditions: `RUN_CONDITIONS.md`.
- Pricing snapshot: `config/pricing.yaml` (dated 2026-05-03, see comments for verification status).
- Judgment artifacts: `judgments/` JSONL (append-only, last-write-wins per image+region+systems tuple).
- Consolidated export: `experiments/exploratorio/consolidated/` Parquet files (post-session analysis).
