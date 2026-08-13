from __future__ import annotations

import argparse
from pathlib import Path
import logging
from utils import set_seed

logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] - (%(name)s): %(message)s")
logger = logging.getLogger(__name__)

def run_hf(args: argparse.Namespace) -> None:
    """HuggingFace LlavaForConditionalGeneration backend."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model_path)
    model_kwargs: dict = {"device_map": args.device_map, "low_cpu_mem_usage": True}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    model = LlavaForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
    model.eval()

    image = Image.open(Path(args.image)).convert("RGB")
    input_device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(text=args.prompt, images=image, return_tensors="pt").to(input_device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    print(processor.decode(output_ids[0], skip_special_tokens=True))


def run_native(args: argparse.Namespace) -> None:
    """LLaVA native (llava.eval.run_llava) backend."""
    from llava.eval.run_llava import eval_model

    # Map args to what run_llava expects
    eval_model(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLaVA inference on a single image.")
    parser.add_argument("--image", default="images/1.jpg", help="Path to input image")
    parser.add_argument("--prompt", default="What is in this image?", help="Text prompt")

    # Backend selection
    parser.add_argument(
        "--backend",
        choices=("hf", "native"),
        default="native",
        help="Inference backend: hf (HuggingFace) or native (LLaVA library)",
    )

    # HF backend options
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")

    # Native backend options
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--conv-mode", default=None)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument(
        "--output-attention",
        action="store_true",
        help="Export all native LLaVA generation attention tensors.",
    )
    parser.add_argument(
        "--attention-output-dir",
        default="outputs/attention",
        help="Directory for the raw generation_attentions.pt artifact.",
    )
    parser.add_argument("--keep-ratio", type=float, default=1.0)

    # Common
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-2d-pe", action="store_true", help="Use 2D positional embeddings for images.")
    parser.add_argument("--pe-scale", type=float, default=1.0, help="Scale for 2D positional embeddings.")

    # Model path: resolve backend-specific default after parsing
    args, unknown = parser.parse_known_args()

    # Set backend-specific defaults
    if args.backend == "hf":
        logger.info("Using HuggingFace backend!")
        parser.set_defaults(model_path="models/llava-1.5-7b-hf")
        # HF needs a different prompt format
        if args.prompt == parser.get_default("prompt"):
            parser.set_defaults(prompt="USER: <image>\nWhat is in this image?\nASSISTANT:")
    else:
        logger.info("Using native backend!")
        parser.set_defaults(model_path="models/llava-v1.5-7b")
        # parser.set_defaults(load_in_4bit=True)

    parser.add_argument("--model-path", default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    # Resolve model_path default
    if args.model_path is None:
        args.model_path = "models/llava-v1.5-7b" if args.backend == "native" else "models/llava-1.5-7b-hf"

    if not Path(args.model_path).is_dir():
        parser.error(f"model path is not a directory: {args.model_path}")
    if not Path(args.image).is_file():
        parser.error(f"image file does not exist: {args.image}")

    # Native backend expects image_file, query
    if args.backend == "native":
        args.image_file = args.image
        args.query = args.prompt
        args.load_4bit = args.load_in_4bit
        run_native(args)
    else:
        run_hf(args)


if __name__ == "__main__":
    main()
