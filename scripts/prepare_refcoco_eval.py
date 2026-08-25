"""Build RefCOCO+/RefCOCOg eval splits in the local JSON schema from Kangheng parquets.

Source: https://hf-mirror.com/datasets/Kangheng/refcocop (UNC+ splits: val 10,758,
testA 5,726, testB 4,889) and https://hf-mirror.com/datasets/Kangheng/refcocog
(UMD splits: val 4,896, test 9,602). Same schema as Kangheng/refcoco: rows carry
the referring sentence, bbox as a JSON string of absolute pixels, the image as
embedded PNG bytes, and image_size (w, h).

One run per split (the parquet dirs contain val/testA/testB/test shards):
  data/refcoco/<split>.json — one entry per sample:
      {"ref_id": i, "image": "img_<md5>.png", "sentence": "...", "bbox": [...]}  (normalised 0-1, xyxy)
  data/refcoco/images/img_<md5>.png — deduped by content hash; images already
      extracted by an earlier run are reused.

The JSON schema matches refcoco_val_questions.json, so refcoco_converter.py can
turn it into the prompt-wrapped JSONL without changes.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


def detect_bbox_format(df, n=500) -> str:
    """Heuristic: xywh boxes satisfy x1 + w <= W and y1 + h <= H (the box is
    inside the image); xyxy boxes violate it whenever the right edge is past
    the middle of the image. Count the fraction of valid rows over a sample."""
    ok = 0
    total = 0
    for _, row in df.head(n).iterrows():
        try:
            bbox = json.loads(row["bbox"])
        except (TypeError, ValueError):
            continue
        if len(bbox) != 4:
            continue
        w, h = int(row["image_size"][0]), int(row["image_size"][1])
        total += 1
        if bbox[0] + bbox[2] <= w * 1.02 and bbox[1] + bbox[3] <= h * 1.02:
            ok += 1
    frac = ok / total if total else 0.0
    fmt = "xywh" if frac > 0.98 else "xyxy"
    print(f"bbox format detection: {ok}/{total} inside-image ({frac:.1%}) -> {fmt}")
    return fmt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", required=True,
                        help="Directory with parquet shards for ONE split.")
    parser.add_argument("--glob", default="*.parquet",
                        help="Glob selecting the shards (e.g. val-*.parquet).")
    parser.add_argument("--output-json", required=True,
                        help="Output JSON path, e.g. data/refcoco/refcocog_val_questions.json.")
    parser.add_argument("--image-dir", default="data/refcoco/images",
                        help="Where extracted images are written (shared across splits).")
    parser.add_argument("--bbox-format", default="auto", choices=["auto", "xyxy", "xywh"],
                        help="Source bbox format; xywh is converted to xyxy.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap on samples (for quick tests).")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    parquet_dir = project_root / args.parquet_dir
    shards = sorted(parquet_dir.glob(args.glob))
    if not shards:
        print(f"ERROR: no parquet shards matching {parquet_dir}/{args.glob}", file=sys.stderr)
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
        if args.bbox_format == "auto" and shard is shards[0]:
            bbox_format = detect_bbox_format(df)
        else:
            bbox_format = args.bbox_format
        for _, row in df.iterrows():
            total += 1
            if args.max_samples and total > args.max_samples:
                break
            img_bytes = row["image"]["bytes"]
            digest = hashlib.md5(img_bytes).hexdigest()
            file_name = f"img_{digest[:12]}.png"
            w, h = int(row["image_size"][0]), int(row["image_size"][1])
            bbox = json.loads(row["bbox"])  # absolute pixels
            if bbox_format == "xywh":
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
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
    print(f"extracted {written} new images -> {image_dir} ({len(entries) - written} already present)")


if __name__ == "__main__":
    main()
