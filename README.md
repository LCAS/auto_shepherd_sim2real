# AutoShepherd Sim2Real

AutoShepherd Sim2Real is a Docker-based image-to-image translation pipeline
for converting synthetic agricultural robotics images into more photorealistic
images. Stable Diffusion is used for inference, with an optional LoRA adapter
for the visual characteristics of the target environment.

## Project layout

- [`src/app.py`](src/app.py) — Batch image-to-image inference entry point.
- [`src/download_data.py`](src/download_data.py) — Temporary ZIP dataset
  downloader for the separate simulation and real-world datasets.
- [`scripts/download_data.sh`](scripts/download_data.sh) — Compose wrapper for
  downloading configured datasets.
- [`src/train.py`](src/train.py) — Placeholder for the future LoRA training
  entry point.
- [`data/sim/`](data/sim/) — Simulation images and downloaded simulation data.
- [`data/real/`](data/real/) — Real-world images and downloaded real-world data.
- [`example/input/`](example/input/) — Small example input set.
- [`example/results/`](example/results/) — Generated example outputs.
- [`models/`](models/) — Local LoRA adapter weights.

Dataset contents, model weights, and generated outputs are ignored by Git.
Small curated files can be added explicitly with `git add -f <path>`.

## Running inference with Docker

The Compose service expects an NVIDIA GPU and the NVIDIA Container Toolkit.
Build the image and process every supported image in `example/input/`:

```bash
docker compose up --build
```

Each input is written as
`example/results/<input-name>_realistic.png`. The pipeline supports PNG, JPEG,
BMP, TIFF, and WebP inputs. The base Stable Diffusion model and LoRA weights
are cached or loaded from `models/sim_to_real_lora` when that directory
contains an adapter.

To use different directories or a different adapter, override the
`INPUT_DIR`, `OUTPUT_DIR`, and `LORA_PATH` environment variables in
`docker-compose.yml` or when launching the container.

## Downloading datasets

Add direct ZIP URLs to the `DATASET_URLS` mapping in
[`src/download_data.py`](src/download_data.py). The simulation and real-world
lists are independent, so either dataset can be downloaded separately:

```bash
./scripts/download_data.sh --dataset sim
./scripts/download_data.sh --dataset real
```

By default, archives are extracted into `data/sim/` or `data/real/`.
`--dataset all` downloads both sets. The wrapper runs the downloader in the
Compose `data-downloader` service, which does not request a GPU. Add direct ZIP
URLs to `DATASET_URLS` before running it; the lists are currently empty. The
downloader is intentionally a temporary utility until the final dataset
hosting and versioning workflow is decided.

## Training

The standard training command will be:

```bash
docker compose run --build --rm sim2real python src/train.py
```

This runs in the GPU-enabled Compose service with `data/sim/`, `data/real/`,
and `models/` mounted. Training is not implemented yet, so the command
currently raises an explicit placeholder error. The future workflow will use
the datasets to fine-tune and save a LoRA adapter under `models/`.
