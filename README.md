# C-JEPA Setup Scaffold (Encoder Stage)

This repository implements C-JEPA on CLEVRER. Phase 1 (current) covers the encoder pipeline:

1. CLEVRER indexing (`jsonl`)
2. video clip loading with temporal sampling
3. frozen pretrained object-centric encoder inference
4. saving per-video slots

Masking, predictor, loss, and training are added in subsequent phases.

## Install

```bash
pip install -r requirements.txt
```

## Build index

Set `data.video_root`, `data.video_glob`, and `data.index_file` in `configs/base.yaml`.

Expected layout is configurable. Common examples:

- Flat:
  - `/data/clevrer/train/*.mp4`
- Nested:
  - `/data/clevrer/train/**/*.mp4`

Use `data.video_glob` to match either structure.

```bash
python scripts/prepare_clevrer_index.py --config configs/base.yaml --with-frame-count
```

## Verify data loading (no encoder)

After the index exists and points at real `.mp4` paths:

```bash
python scripts/test_dataloader.py --config configs/base.yaml
```

## Run slot extraction

Update `configs/base.yaml` first (especially `data.index_file` and `encoder.checkpoint_path`), then run:

```bash
python scripts/run_slot_extract.py --config configs/base.yaml
```

Outputs are stored as `.pt` files with:
- `video_id`
- `slots` (`[T, N, D]` per video)
- `frame_indices` (`[T]`)
- `metadata`

## Notes

- SAVi is first-class in this scaffold.
- VideoSAUR support is intentionally deferred as a future extension.
- Encoder validation outputs on CLEVRER val are in `figures/` (see `figures/README.md`).
