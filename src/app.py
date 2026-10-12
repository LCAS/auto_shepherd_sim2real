import os
from pathlib import Path

import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = Path(
    os.getenv("INPUT_DIR", str(PROJECT_ROOT / "example" / "input"))
)
OUTPUT_DIR = Path(
    os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "example" / "results"))
)
LORA_PATH = Path(
    os.getenv("LORA_PATH", str(PROJECT_ROOT / "models" / "sim_to_real_lora"))
)
SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

PROMPT = (
    "Photorealistic outdoor agricultural environment, natural light and "
    "realistic textures"
)
NEGATIVE_PROMPT = "CGI, cartoon, artificial textures, 3D render"


def find_input_images(input_dir: Path) -> list[Path]:
    """Return supported images in deterministic order."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    images = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(
            f"No supported images found in input directory: {input_dir}"
        )
    return images


def load_pipeline():
    """Load the base image-to-image pipeline and an optional LoRA adapter."""
    configured_device = os.getenv("DEVICE")
    device = configured_device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    pipe = AutoPipelineForImage2Image.from_pretrained(
        BASE_MODEL, torch_dtype=torch_dtype
    ).to(device)

    if LORA_PATH.is_dir() and any(LORA_PATH.iterdir()):
        pipe.load_lora_weights(str(LORA_PATH))
        pipe.set_adapters(["default_0"], adapter_weights=[0.8])
    else:
        print(f"LoRA not found at {LORA_PATH}; running base model only.")

    return pipe


def process_image(pipe, input_path: Path, output_path: Path) -> None:
    """Generate and save one photorealistic image."""
    with Image.open(input_path) as source_image:
        image = source_image.convert("RGB")

    image.thumbnail((768, 768))
    image = image.resize(
        (max(8, image.width // 8 * 8), max(8, image.height // 8 * 8))
    )

    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=image,
        strength=0.3,
        guidance_scale=6.0,
        num_inference_steps=30,
    ).images[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)


def main() -> None:
    input_images = find_input_images(INPUT_DIR)
    pipe = load_pipeline()

    print(f"Processing {len(input_images)} image(s) from {INPUT_DIR}")
    for input_path in input_images:
        output_path = OUTPUT_DIR / f"{input_path.stem}_realistic.png"
        process_image(pipe, input_path, output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
