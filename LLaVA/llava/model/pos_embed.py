"""2D sinusoidal positional embeddings for visual patch features."""

from functools import lru_cache

import torch


@lru_cache(maxsize=8)
def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Build additive 2D sinusoidal positional embeddings.

    Raster-ordered patch tokens: token t in a grid_size x grid_size image
    has coordinates (row, col) = (t // grid_size, t % grid_size).

    The first half of `embed_dim` encodes the row coordinate, the second
    half encodes the column coordinate, each with the standard sinusoidal
    scheme: pe(pos, 2k) = sin(pos / 10000^(2k/d)), pe(pos, 2k+1) = cos(...).

    Returns:
        Tensor of shape [grid_size**2, embed_dim] in float32, values in [-1, 1].
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    if grid_size < 1:
        raise ValueError(f"grid_size must be >= 1, got {grid_size}")

    half_dim = embed_dim // 2

    def _1d(positions: torch.Tensor) -> torch.Tensor:
        # positions: [L] -> [L, half_dim]
        positions = positions.unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, half_dim, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / half_dim)
        )
        pe = torch.zeros(positions.shape[0], half_dim)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe

    grid = torch.arange(grid_size)
    row_embed = _1d(grid.repeat_interleave(grid_size))  # raster row coords
    col_embed = _1d(grid.repeat(grid_size))             # raster col coords
    return torch.cat([row_embed, col_embed], dim=1)
