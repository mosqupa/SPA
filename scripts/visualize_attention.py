"""Render prefill attention heatmaps from an exported LLaVA attention file."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vlm")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _tick_locations(position_ids: np.ndarray, max_ticks: int = 9):
    count = min(max_ticks, len(position_ids))
    locations = np.unique(np.linspace(0, len(position_ids) - 1, count, dtype=int))
    return locations, [str(int(position_ids[index])) for index in locations]


def _mean_attention(
    layer_attention: torch.Tensor,
    batch_index: int,
    valid_indices: torch.Tensor,
) -> torch.Tensor:
    # [batch, heads, query, key] -> [valid_query, valid_key]
    attention = layer_attention[batch_index].detach().float().mean(dim=0)
    indices = valid_indices.to(attention.device)
    return attention.index_select(0, indices).index_select(1, indices)


def visualize_prefill_attention(
    attention_file: str | Path,
    output_dir: str | Path,
    *,
    batch_index: int = 0,
    top_k: int = 10,
) -> Path:
    """Load a raw export and visualize the square prefill attention matrices."""
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}.")

    attention_file = Path(attention_file)
    if not attention_file.is_file():
        raise FileNotFoundError(f"Attention export does not exist: {attention_file}")

    # Only load files produced by this project; torch.load uses Python pickle.
    payload = torch.load(attention_file, map_location="cpu")
    if payload.get("format_version") != 1:
        raise ValueError(
            f"Unsupported attention format version: {payload.get('format_version')}"
        )
    attentions = payload.get("attentions")
    if not attentions or not attentions[0]:
        raise ValueError("The export contains no attention tensors.")

    prefill_attentions = attentions[0]
    first = prefill_attentions[0]
    if first.ndim != 4 or first.shape[-2] != first.shape[-1]:
        raise ValueError(
            "Expected square prefill tensors shaped [batch, heads, query, key], "
            f"got {tuple(first.shape)}."
        )
    if not 0 <= batch_index < first.shape[0]:
        raise IndexError(
            f"batch_index {batch_index} is outside [0, {first.shape[0] - 1}]."
        )

    prompt_len = first.shape[-1]
    prompt_attention_mask = payload.get("prompt_attention_mask")
    if (
        prompt_attention_mask is not None
        and prompt_attention_mask.shape[-1] == prompt_len
    ):
        mask = prompt_attention_mask[batch_index].detach().cpu().bool()
        valid_indices = torch.nonzero(mask, as_tuple=False).flatten()
    else:
        valid_indices = torch.arange(prompt_len)
    if valid_indices.numel() == 0:
        raise ValueError("The selected prompt contains no valid tokens.")

    prompt_position_ids = payload.get("prompt_position_ids")
    if prompt_position_ids is not None and prompt_position_ids.shape[-1] == prompt_len:
        position_ids = (
            prompt_position_ids[batch_index].detach().cpu()[valid_indices].numpy()
        )
    else:
        position_ids = valid_indices.numpy()

    output_dir = Path(output_dir)
    layers_dir = output_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    # Compute a shared scale so the colors remain comparable across layers.
    samples = []
    overview_maps = []
    overview_size = min(128, valid_indices.numel())
    for layer_attention in prefill_attentions:
        mean_attention = _mean_attention(layer_attention, batch_index, valid_indices)
        scale_source = mean_attention[1:] if mean_attention.shape[0] > 1 else mean_attention
        flat = scale_source.flatten()
        stride = max(1, flat.numel() // 20_000)
        sample = flat[::stride]
        sample = sample[sample > 0]
        if sample.numel():
            samples.append(sample)
        overview_maps.append(
            F.interpolate(
                mean_attention[None, None],
                size=(overview_size, overview_size),
                mode="area",
            )[0, 0].numpy()
        )

    color_max = (
        float(torch.quantile(torch.cat(samples), 0.995).item())
        if samples else 1.0
    )
    color_max = max(color_max, np.finfo(np.float32).eps)
    tick_locations, tick_labels = _tick_locations(position_ids)
    top_rows = []

    for layer_idx, layer_attention in enumerate(prefill_attentions):
        attention_map = _mean_attention(
            layer_attention, batch_index, valid_indices
        ).numpy()
        last_query = attention_map[-1]
        count = min(top_k, last_query.shape[0])
        top_indices = np.argsort(last_query)[-count:][::-1]
        for rank, key_index in enumerate(top_indices, start=1):
            top_rows.append(
                {
                    "layer": layer_idx,
                    "rank": rank,
                    "query_index": int(valid_indices[-1]),
                    "query_position_id": int(position_ids[-1]),
                    "key_index": int(valid_indices[key_index]),
                    "key_position_id": int(position_ids[key_index]),
                    "attention": float(last_query[key_index]),
                }
            )

        fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")
        image = ax.imshow(
            attention_map,
            cmap="Blues",
            vmin=0,
            vmax=color_max,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
        )
        ax.set_title(f"Layer {layer_idx:02d} — Prefill mean attention")
        ax.set_xlabel("Key position ID")
        ax.set_ylabel("Query position ID")
        ax.set_xticks(tick_locations, tick_labels, rotation=45, ha="right")
        ax.set_yticks(tick_locations, tick_labels)
        ax.grid(False)
        fig.colorbar(image, ax=ax, label="Attention weight (mean across heads)")
        fig.text(
            0.5,
            0.01,
            f"Batch {batch_index}; {layer_attention.shape[1]} heads; "
            "shared scale clipped at sampled p99.5",
            ha="center",
            fontsize=9,
            color="#4b5563",
        )
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        fig.savefig(layers_dir / f"layer_{layer_idx:02d}.png", dpi=170)
        plt.close(fig)

    column_count = 4
    row_count = int(np.ceil(len(overview_maps) / column_count))
    overview_height = 3.25 * row_count + 0.8
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(14, overview_height),
        facecolor="white",
        squeeze=False,
    )
    overview_image = None
    for layer_idx, (ax, attention_map) in enumerate(zip(axes.flat, overview_maps)):
        overview_image = ax.imshow(
            attention_map,
            cmap="Blues",
            vmin=0,
            vmax=color_max,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
        )
        ax.set_title(f"Layer {layer_idx:02d}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(overview_maps):]:
        ax.axis("off")

    fig.suptitle("LLaVA prefill attention — mean across heads", fontsize=16, y=0.995)
    if overview_image is not None:
        colorbar_axis = fig.add_axes((0.925, 0.25, 0.012, 0.5))
        fig.colorbar(overview_image, cax=colorbar_axis, label="Attention weight")
    overview_top = 1.0 - 0.72 / overview_height
    fig.subplots_adjust(
        left=0.04,
        right=0.90,
        bottom=0.03,
        top=overview_top,
        wspace=0.12,
        hspace=0.22,
    )
    fig.savefig(output_dir / "all_layers_overview.png", dpi=170)
    plt.close(fig)

    with (output_dir / "last_query_topk.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=top_rows[0].keys())
        writer.writeheader()
        writer.writerows(top_rows)

    metadata = {
        "source_file": str(attention_file.resolve()),
        "batch_index": batch_index,
        "num_generation_steps_available": len(attentions),
        "num_layers": len(prefill_attentions),
        "num_heads": int(first.shape[1]),
        "prompt_tensor_length": int(prompt_len),
        "valid_prompt_length": int(valid_indices.numel()),
        "aggregation": "mean_across_attention_heads",
        "color_scale": {"min": 0.0, "max_sampled_p99_5": color_max},
        "axis_labels": "original_position_id_when_available_otherwise_sequence_index",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize prefill attention from a raw LLaVA attention export."
    )
    parser.add_argument(
        "--attention-file",
        default="outputs/attention/generation_attentions.pt",
        help="Raw file produced by run_llava.py --output-attention.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Visualization directory (default: <attention-file-dir>/visualizations).",
    )
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    attention_file = Path(args.attention_file)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else attention_file.parent / "visualizations"
    )
    result = visualize_prefill_attention(
        attention_file,
        output_dir,
        batch_index=args.batch_index,
        top_k=args.top_k,
    )
    print(f"Attention visualizations saved to: {result.resolve()}")


if __name__ == "__main__":
    main()
