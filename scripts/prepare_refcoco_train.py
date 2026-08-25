"""Build the RefCOCO train split in the local JSON schema from Kangheng/refcoco parquets.

Source: https://hf-mirror.com/datasets/Kangheng/refcoco (official UNC train split,
120,624 samples, images embedded as PNG bytes, one row per referring expression).

Outputs:
  data/refcoco/refcoco_train_questions.json — one entry per sample:
      {"ref_id": i, "image": "train_img_<md5>.png",
       "sentence": "...", "bbox": [x1, y1, x2, y2]}   (bbox normalised 0-1)
  data/refcoco/train_images/train_img_<md5>.png — extracted images, deduped by
      content hash (an image is referenced by ~7 expressions on average)

The JSON schema matches refcoco_val_questions.json, so refcoco_converter.py can
turn it into the prompt-wrapped JSONL. Image file names are hash-based because
Kangheng's rows do not carry the official COCO image id; training only needs
stable, unique names.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", default="data/refcoco/train2014_dl/kangheng",
                        help="Directory with train-XXXXX-of-00125.parquet shards.")
    parser.add_argument("--output-json", default="data/refcoco/refcoco_train_questions.json")
    parser.add_argument("--image-dir", default="data/refcoco/train_images")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap on samples (for quick tests).")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    parquet_dir = project_root / args.parquet_dir
    shards = sorted(parquet_dir.glob("train-*.parquet"))
    if not shards:
        print(f"ERROR: no parquet shards found in {parquet_dir}", file=sys.stderr)
        sys.exit(1)

    out_json = project_root / args.output_json
    image_dir = project_root / args.image_dir
    image_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    total = 0
    written = 0
    for shard in shards:
        try:
            df = pd.read_parquet(shard, columns=["question", "bbox", "image", "image_size"])
        except Exception as e:
            print(f"WARNING: skipping corrupt shard {shard.name}: {e}", file=sys.stderr)
            continue
        for _, row in df.iterrows():
            total += 1
            if args.max_samples and total > args.max_samples:
                break
            img_bytes = row["image"]["bytes"]
            digest = hashlib.md5(img_bytes).hexdigest()
            file_name = f"train_img_{digest[:12]}.png"
            w, h = int(row["image_size"][0]), int(row["image_size"][1])
            bbox = json.loads(row["bbox"])  # absolute pixels [x1, y1, x2, y2]
            entries.append({
                "ref_id": total - 1,
                "image": file_name,
                "sentence": row["question"],
                "bbox": [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h],
            })
            if not (image_dir / file_name).exists():
                (image_dir / file_name).write_bytes(img_bytes)
                written += 1
        if args.max_samples and total > args.max_samples:
            break

    with open(out_json, "w") as f:
        json.dump(entries, f)
    print(f"scanned {total} rows | wrote {len(entries)} entries -> {out_json}")
    print(f"extracted {written} unique images -> {image_dir}")


if __name__ == "__main__":
    main()
