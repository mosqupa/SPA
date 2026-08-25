"""Learnable positional adapter for visual patch features.

Adds a content-free, position-only delta to the projected visual features:

    image_features = image_features + delta
    delta = MLP(fourier(coords))

The adapter input is pure geometry (normalised patch coordinates), never
image content, so it can only learn positional priors — it cannot memorise
training images.

Fourier features are used instead of raw (x, y) coordinates because small
MLPs have a spectral bias towards low frequencies; the log-spaced sine/cosine
bank lets the network represent fine spatial structure.

Note on the removed gate: an earlier design multiplied the delta by a
zero-initialised learnable scalar gate (position_gate) so training started
exactly at the no-PE baseline. The first gradient step has zero MLP gradient
(backprop through the gate) and, once the gate opens to ~1e-3, MLP gradients
are suppressed by three orders of magnitude — slow, noisy convergence. For
the first ablation ("does a learned positional representation help at all")
the gate is dropped so gradients flow to the MLP directly.
"""

import math

import torch
import torch.nn as nn


class PositionAdapter(nn.Module):
    def __init__(self, embed_dim: int = 4096, num_freqs: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.num_freqs = num_freqs
        # Log-spaced frequencies 2^0 .. 2^10. Upper bound keeps sin/cos of
        # large arguments well-conditioned in float32 on a 24x24 grid.
        freqs = 2.0 ** torch.linspace(0.0, 10.0, num_freqs)
        self.register_buffer("freqs", freqs)

        # Fourier features: 2 coords x (num_freqs * (sin, cos)) = 4 * num_freqs
        self.mlp = nn.Sequential(
            nn.Linear(4 * num_freqs, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        # LLaVA's disable_torch_init() turns nn.Linear.reset_parameters into a
        # no-op, so layers created after it would keep uninitialized memory.
        # Initialize explicitly (nn.init.* is not monkeypatched).
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: [L, 2] normalised to [-1, 1]. Returns delta of shape [L, embed_dim]."""
        f = self.freqs  # [F]
        x = coords.unsqueeze(-1) * f       # [L, 2, F]
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # [L, 2, 2F]
        x = x.reshape(coords.shape[0], -1) # [L, 4F]
        return self.mlp(x)                 # [L, embed_dim]

    @staticmethod
    def make_coords(grid_size: int, device, dtype=torch.float32) -> torch.Tensor:
        """Raster-ordered patch coordinates in [-1, 1] for a grid_size x grid_size image.

        Token t at grid position (row, col) = (t // grid_size, t % grid_size)
        matches the ordering used by get_2d_sincos_pos_embed. Coords are fp32 to
        match the adapter's fp32 master weights.
        """
        grid = torch.arange(grid_size, device=device, dtype=dtype)
        norm = (grid / (grid_size - 1)) * 2 - 1 if grid_size > 1 else torch.zeros_like(grid)
        row = norm.repeat_interleave(grid_size)   # raster row coords
        col = norm.repeat(grid_size)              # raster col coords
        return torch.stack([row, col], dim=1)     # [grid_size**2, 2]


class PositionAdapterRawCoords(PositionAdapter):
    """Variant using raw (x, y) instead of Fourier features.

    Ablation baseline: answers whether the Fourier feature mapping is
    necessary. Parameter count ~1.05M vs ~1.11M for the Fourier version
    (the first layer is Linear(2, hidden) instead of Linear(256, hidden)).
    """

    def __init__(self, embed_dim: int = 4096, hidden_dim: int = 256):
        # Parent builds the Fourier MLP (Linear(4*num_freqs, hidden)) and the
        # freqs buffer; both are replaced/ignored here. Overriding self.mlp
        # means _init_weights must run again: disable_torch_init makes
        # nn.Linear.reset_parameters a no-op, so the new first layer would
        # otherwise keep uninitialized memory.
        super().__init__(embed_dim=embed_dim, hidden_dim=hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self._init_weights()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: [L, 2] normalised to [-1, 1]. Returns delta of shape [L, embed_dim]."""
        return self.mlp(coords)