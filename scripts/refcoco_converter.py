"""Convert RefCOCO question JSON to the JSONL format expected by model_vqa_loader.py.

RefCOCO data schema (input):
    [{"ref_id": 0, "image": "COCO_train2014_000000580957_4.jpg",
      "sentence": "bowl behind the others can only see part",
      "bbox": [x1, y1, x2, y2] (normalised 0–1)}]

Output JSONL schema:
    {"question_id": 0, "image": "...", "text": "prompt asking model to localise",
     "bbox": [x1, y1, x2, y2], "sentence": "..."}
"""

import argparse
import json
import os
from pathlib import Path


PROMPT_TEMPLATE = (
    "Look at the image. Find the object described by: \"{sentence}\". "
    "Output ONLY the normalised bounding box in the format [x1, y1, x2, y2] "
    "with values between 0 and 1."
)


def convert(split_file: str, output_dir: str) -> Path:
    """Convert a single RefCOCO split file to JSONL."""
    with open(split_file) as f:
        questions = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    stem = Path(split_file).stem  # e.g. refcoco_val_questions
    out_path = Path(output_dir) / f"{stem}.jsonl"

    with open(out_path, "w") as out:
        for item in questions:
            line = {
                "question_id": item["ref_id"],
                "image": item["image"],
                "text": PROMPT_TEMPLATE.format(sentence=item["sentence"]),
                "bbox": item["bbox"],
                "sentence": item["sentence"],
            }
            out.write(json.dumps(line) + "\n")

    print(f"  {out_path}  ({len(questions)} samples)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RefCOCO data to JSONL for model_vqa_loader.")
    parser.add_argument(
        "--splits-dir",
        default="data/benchmarks/refcoco",
        help="Directory containing RefCOCO split JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/benchmarks/refcoco/converted",
        help="Output directory for JSONL files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["refcoco_val_questions", "refcoco_test_questions", "refcoco_testB_questions"],
        help="Split names (without .json extension).",
    )
    args = parser.parse_args()

    for split_name in args.splits:
        split_path = os.path.join(args.splits_dir, f"{split_name}.json")
        if not os.path.isfile(split_path):
            print(f"  SKIP: {split_path} not found")
            continue
        convert(split_path, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
