<<<<<<< HEAD
# C-JEPA Setup Scaffold (Encoder Stage)

This repository stage is intentionally limited to:

1. CLEVRER indexing (`jsonl`)
2. video clip loading with temporal sampling
3. frozen pretrained object-centric encoder inference
4. saving per-video slots

No masking/prediction heads/world-model logic is included.

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
- `src/encoders/savi_encoder.py` contains explicit TODO hooks for teammate-provided checkpoint loading details.
- VideoSAUR support is intentionally deferred as a future extension.
=======
# C-JEPA-Reproduction
>>>>>>> cde34997c8afd5e683b0734b51e557a7a0ade903
