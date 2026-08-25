"""Train the PositionAdapter on RefCOCO train with a fully frozen backbone.

Only the adapter's MLP + position_gate are trainable; the CLIP vision tower,
mm_projector and LLM are frozen. The task is next-token prediction on the
referring-expression bbox answer, teacher-forced (same prompt format as
scripts/refcoco_inference.py, so the eval-time distribution matches training).

Data: data/refcoco/converted/refcoco_train_questions.jsonl (built via
scripts/refcoco_converter.py from scripts/prepare_refcoco_train.py output).

Usage:
    bash -c 'cd ... && /opt/conda/envs/vlm/bin/python scripts/train_pos_adapter.py \
        --max-samples 30000 --save-path outputs/pos_adapter.pt'
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.utils import disable_torch_init


def build_pair(question_text: str, bbox: list[float], conv_mode: str) -> tuple[str, str]:
    """Return (full_prompt, question_only_prompt) for teacher forcing."""
    qs = DEFAULT_IMAGE_TOKEN + '\n' + question_text
    answer = f"[{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], answer)
    full = conv.get_prompt()

    conv_q = conv_templates[conv_mode].copy()
    conv_q.append_message(conv_q.roles[0], qs)
    conv_q.append_message(conv_q.roles[1], None)
    question_only = conv_q.get_prompt()
    return full, question_only


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/llava-v1.5-7b")
    parser.add_argument("--model-name", default="llava-v1.5-7b")
    parser.add_argument("--data-dir", default="data/refcoco")
    parser.add_argument("--split", default="refcoco_train_questions")
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--max-samples", type=int, default=30000,
                        help="Cap on training samples per epoch (RefCOCO train has 120,624).")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--save-path", default="outputs/pos_adapter.pt")
    parser.add_argument("--adapter-type", default="fourier", choices=["fourier", "raw"],
                        help="PositionAdapter variant: fourier (default) or raw (x, y) coords.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / args.data_dir
    image_dir = data_dir / "train_images"
    question_file = data_dir / "converted" / f"{args.split}.jsonl"
    if not question_file.exists():
        raise SystemExit(f"missing {question_file} — run prepare + convert first")
    save_path = project_root / args.save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    with open(question_file) as f:
        questions = [json.loads(line) for line in f][: args.max_samples]
    print(f"Split: {args.split} | samples: {len(questions)}")

    # Load model (fp16) and freeze everything except the position adapter
    disable_torch_init()
    model_path = str(project_root / args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, get_model_name_from_path(model_path))
    for p in model.parameters():
        p.requires_grad_(False)
    # ensure_position_adapter() reads config.pos_adapter_type to pick the variant
    model.config.pos_adapter_type = args.adapter_type
    adapter = model.get_model().ensure_position_adapter()
    for p in adapter.parameters():
        p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable:,} (PositionAdapter only)")

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.0)
    total_steps = args.epochs * len(questions)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / args.warmup_steps)
        * (0.5 * (1 + math.cos(math.pi * min(step / max(total_steps - 1, 1), 1.0)))),
    )

    model.train()
    t0 = time.time()
    step = 0
    running_loss = 0.0
    for epoch in range(args.epochs):
        order = list(range(len(questions)))
        random.shuffle(order)
        for idx in order:
            q = questions[idx]
            image_path = image_dir / q["image"]
            if not image_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")
            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = image_tensor.to(dtype=torch.float16, device="cuda")

            full_prompt, question_only = build_pair(q["text"], q["bbox"], args.conv_mode)
            input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors="pt").unsqueeze(0).cuda()
            q_ids = tokenizer_image_token(question_only, tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).cuda()
            labels = input_ids.clone()
            labels[:, : q_ids.shape[1]] = -100  # mask the question part

            out = model(input_ids=input_ids, images=image_tensor, image_sizes=[image.size],
                        labels=labels, use_pos_adapter=True)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            running_loss += loss.item()
            if step % args.log_interval == 0:
                avg = running_loss / args.log_interval
                running_loss = 0.0
                # Frobenius norm of the adapter delta (set by encode_images on
                # the last forward) — the observable signal that the adapter
                # is producing a non-trivial positional correction.
                delta_norm = getattr(model, "_last_pos_embed_norm", 0.0)
                print(f"step {step}/{total_steps} | loss {avg:.4f} | lr {scheduler.get_last_lr()[0]:.2e} "
                      f"| delta_norm {delta_norm:.1f} | {time.time()-t0:.0f}s", flush=True)
            if step % args.save_interval == 0:
                torch.save(adapter.state_dict(), save_path)
                print(f"  checkpoint -> {save_path}", flush=True)

    torch.save(adapter.state_dict(), save_path)
    elapsed = time.time() - t0
    print(f"done in {elapsed/60:.1f} min ({elapsed/step:.3f}s/step) | saved -> {save_path}")


if __name__ == "__main__":
    main()
