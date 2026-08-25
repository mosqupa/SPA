"""TextVQA val inference + scoring — local data, adapter-aware.

Same pipeline shape as scripts/gqa_inference.py: one question at a time,
visual token pruning + optional learned PositionAdapter, output per combo in
data/textvqa/answers/, scored with the official TextVQA protocol (prediction
matches any of the 10 gold answers after normalisation).

Data layout:
    data/textvqa/textvqa_val_questions.json   (list: {question_id, image_id, question, answers[10]})
    data/textvqa/images/<image_id>.jpg

Usage:
    /opt/conda/envs/vlm/bin/python scripts/textvqa_inference.py --keep-ratio 0.5
    /opt/conda/envs/vlm/bin/python scripts/textvqa_inference.py --keep-ratio 0.5 \
        --use-pos-adapter --adapter-path outputs/pos_adapter.pt

Outputs:
    data/textvqa/answers/<model>/random_<keep_ratio>[_adapter]/merge.jsonl
    data/textvqa/answers/<model>/random_<keep_ratio>[_adapter]/metrics.txt
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


def build_prompt(question: str, conv_mode: str) -> str:
    qs = DEFAULT_IMAGE_TOKEN + '\n' + question + '\nAnswer the question using a single word or phrase.'
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def normalize_answer(s: str) -> str:
    """LLaVA convert_textvqa_for_submission convention: lower, strip, drop period."""
    return s.lower().strip().replace(".", "")


def evaluate(predictions: list[dict], questions: list[dict]) -> dict:
    """Prediction counts as correct if it matches ANY of the 10 gold answers."""
    total = correct = 0
    for p, q in zip(predictions, questions):
        total += 1
        pred = normalize_answer(p["text"])
        if any(pred == normalize_answer(a) for a in q["answers"]):
            correct += 1
    return {"total": total, "correct": correct, "accuracy": 100.0 * correct / max(total, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/llava-v1.5-7b")
    parser.add_argument("--model-name", default="llava-v1.5-7b")
    parser.add_argument("--data-dir", default="data/textvqa")
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
    question_file = data_dir / "textvqa_val_questions.json"
    image_dir = data_dir / "images"

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
        questions = questions[: args.max_samples]
    print(f"Split:       textvqa_val")
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
        for q in tqdm(questions, desc="Questions", unit="q"):
            image_path = image_dir / f"{q['image_id']}.jpg"
            if not image_path.exists():
                print(f"  WARNING: missing {image_path}, skipping qid={q['question_id']}")
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

            result = {"question_id": q["question_id"], "prompt": q["question"],
                      "text": text, "model_id": args.model_name}
            results.append(result)
            ans_f.write(json.dumps(result) + "\n")

    elapsed = time.time() - start
    print(f"\nInference done in {elapsed/60:.1f} min ({elapsed/max(len(results),1):.3f}s per question)")
    print(f"Output: {answer_file}")

    stats = evaluate(results, questions)
    report = (f"Total questions:  {stats['total']}\n"
              f"Correct:          {stats['correct']}\n\n"
              f"Accuracy:         {stats['accuracy']:.2f}%\n")
    print("\n" + report)
    metrics_file.write_text(report)
    print(f"Results saved to: {metrics_file}")


if __name__ == "__main__":
    main()
