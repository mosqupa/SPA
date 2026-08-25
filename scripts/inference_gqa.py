"""GQA testdev_balanced inference + official-protocol scoring.

Local-data entrypoint for the GQA visual-reasoning benchmark, mirroring the
RefCOCO pipeline (scripts/refcoco_inference.py). Unlike the upstream LLaVA
entrypoint (llava.eval.model_vqa_loader), this script:

  - reads questions/images from data/gqa (no playground/ layout)
  - supports visual token pruning (--keep-ratio) and the learned PositionAdapter
    (--use-pos-adapter / --adapter-path) — the exact knobs we ablate on
  - scores predictions with the official GQA protocol inline (no 822MB eval.zip)

Data layout:
    data/gqa/questions/testdev_balanced_questions.json   (dict: questionId -> meta)
    data/gqa/images/testdev_balanced/<imageId>.jpg

Usage:
    /opt/conda/envs/vlm/bin/python scripts/gqa_inference.py --keep-ratio 0.5
    /opt/conda/envs/vlm/bin/python scripts/gqa_inference.py --keep-ratio 0.5 \
        --use-pos-adapter --adapter-path outputs/pos_adapter.pt

Outputs (per keep_ratio x adapter combo):
    data/gqa/answers/<model>/random_<keep_ratio>[_adapter]/merge.jsonl
    data/gqa/answers/<model>/random_<keep_ratio>[_adapter]/metrics.txt
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

GQA_PROMPT_SUFFIX = "Answer the question using a single word or phrase."


def build_prompt(question: str, conv_mode: str) -> str:
    qs = DEFAULT_IMAGE_TOKEN + '\n' + question + '\n' + GQA_PROMPT_SUFFIX
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def preprocess_answer(s: str) -> str:
    """GQA official eval.py answer normalisation.

    Applied to both the prediction and the gold answer: lowercase and strip a
    fixed set of punctuation (comma, period, question mark, parens, quotes),
    preserving spaces. Mirrors the `preprocess` helper in the official
    stanfordnlp/gqa eval.py, which the LLaVA pipeline reaches via
    convert_gqa_for_eval.py + eval/eval.py.
    """
    return (s.lower()
            .replace(",", "")
            .replace(".", "")
            .replace("?", "")
            .replace("(", "")
            .replace(")", "")
            .replace('"', "")
            .replace("'", ""))


def evaluate(predictions: list[dict], questions: dict) -> dict:
    """Score per the official protocol: overall + binary (yes/no) + open."""
    total = correct = 0
    bin_c = bin_t = 0
    open_c = open_t = 0
    missing = 0
    for p in predictions:
        qid = str(p["question_id"])
        if qid not in questions:
            continue
        gold = preprocess_answer(questions[qid]["answer"])
        pred = preprocess_answer(p["text"])
        hit = pred == gold
        total += 1
        correct += int(hit)
        if questions[qid]["answer"] in ("yes", "no"):
            bin_t += 1
            bin_c += int(hit)
        else:
            open_t += 1
            open_c += int(hit)
    if missing:
        print(f"WARNING: {missing} predictions have no matching questionId")
    return {
        "total": total, "correct": correct, "accuracy": 100.0 * correct / max(total, 1),
        "binary": 100.0 * bin_c / max(bin_t, 1), "binary_n": bin_t,
        "open": 100.0 * open_c / max(open_t, 1), "open_n": open_t,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/llava-v1.5-7b")
    parser.add_argument("--model-name", default="llava-v1.5-7b")
    parser.add_argument("--split", default="testdev_balanced",
                        help="GQA tier (testdev_balanced is the only local split).")
    parser.add_argument("--data-dir", default="data/gqa")
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--pruning-method", default="random", choices=["random", "uniform"])
    parser.add_argument("--keep-ratio", type=float, default=1.0)
    parser.add_argument("--use-pos-adapter", action="store_true",
                        help="Add the learned PositionAdapter delta to visual features.")
    parser.add_argument("--adapter-path", default=None,
                        help="Trained PositionAdapter state_dict (default: random-init adapter).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap on questions processed (smoke tests).")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / args.data_dir
    question_file = data_dir / "questions" / f"{args.split}_questions.json"
    image_dir = data_dir / "images" / args.split

    tag = f"random_{args.keep_ratio}"
    if args.use_pos_adapter:
        tag += "_adapter"
    answer_dir = data_dir / "answers" / args.model_name / tag
    answer_dir.mkdir(parents=True, exist_ok=True)
    answer_file = answer_dir / "merge.jsonl"
    metrics_file = answer_dir / "metrics.txt"

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)

    with open(question_file) as f:
        questions = json.load(f)
    if args.max_samples:
        questions = dict(list(questions.items())[: args.max_samples])
    print(f"Split:       {args.split}")
    print(f"Questions:   {len(questions)}")
    print(f"Answer dir:  {answer_dir}")
    print()

    disable_torch_init()
    model_path = str(project_root / args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name)

    if args.use_pos_adapter:
        adapter = model.get_model().ensure_position_adapter()
        if args.adapter_path:
            adapter.load_state_dict(torch.load(str(project_root / args.adapter_path), map_location="cpu"))
            print(f"Loaded PositionAdapter from: {args.adapter_path}")
        else:
            print("WARNING: no --adapter-path given; using randomly-initialised adapter.")

    results = []
    start = time.time()
    with open(answer_file, "w") as ans_f:
        for qid, q in tqdm(questions.items(), desc="Questions", unit="q"):
            image_path = image_dir / f"{q['imageId']}.jpg"
            if not image_path.exists():
                print(f"  WARNING: missing {image_path}, skipping qid={qid}")
                continue
            image = Image.open(image_path).convert('RGB')
            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = image_tensor.to(dtype=torch.float16, device='cuda')

            prompt = build_prompt(q["question"], args.conv_mode)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors='pt').unsqueeze(0).cuda()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image.size],
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    keep_ratio=args.keep_ratio,
                    use_2d_pe=False,
                    pe_scale=1.0,
                    shuffle_pe=False,
                    use_noise=False,
                    use_pos_adapter=args.use_pos_adapter,
                )
            text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            result = {
                "question_id": qid,
                "prompt": q["question"],
                "text": text,
                "model_id": args.model_name,
            }
            results.append(result)
            ans_f.write(json.dumps(result) + "\n")

    elapsed = time.time() - start
    print(f"\nInference done in {elapsed/60:.1f} min ({elapsed/max(len(results),1):.3f}s per question)")
    print(f"Output: {answer_file}")

    stats = evaluate(results, questions)
    report = (f"Total predictions:  {stats['total']}\n"
              f"Correct:           {stats['correct']}\n\n"
              f"Overall accuracy:  {stats['accuracy']:.2f}%\n"
              f"Binary (yes/no):   {stats['binary']:.2f}%  (n={stats['binary_n']})\n"
              f"Open:              {stats['open']:.2f}%  (n={stats['open_n']})\n")
    print("\n" + report)
    metrics_file.write_text(report)
    print(f"Results saved to: {metrics_file}")


if __name__ == "__main__":
    main()
