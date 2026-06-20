"""
ag_resunet.py
=============
A reference Attention-Gated Residual U-Net (AG-ResU-Net) for TO layout
prediction, written so the attention-gate coefficient maps alpha are
*first-class, inspectable tensors*.

The architecture mirrors the family of networks used to predict converged SIMP
layouts from early-iteration fields (the IoU 93-99% model in the study). The
exact channel counts are not the point -- the point is that each skip
connection passes through an Oktay-2018 additive Attention Gate that produces a
single-channel coefficient map alpha in [0,1] at the skip's spatial resolution.
That alpha is exactly the quantity we interrogate against the mechanics field.

If you have your OWN trained AG-ResU-Net, you do NOT need this class. You only
need each attention gate to (a) be an nn.Module and (b) store its last alpha
map. `AttentionExtractor` (see attention.py) can hook arbitrary modules; this
file is the drop-in reference + the contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Two 3x3 convs with a projected identity shortcut (pre-activation-ish)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(cout)
        self.proj = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        idn = self.proj(x)
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(h + idn)


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018).

    gating signal g (decoder, coarser) is upsampled to the skip resolution,
    then an additive attention produces a single-channel coefficient map
    alpha in [0,1] AT THE SKIP RESOLUTION. The gated skip is alpha * x.

    The last alpha computed in forward() is cached in `self.last_alpha`
    (detached, on CPU) for downstream mechanical interpretation.
    """

    def __init__(self, c_skip: int, c_gate: int, c_int: int):
        super().__init__()
        self.W_x = nn.Conv2d(c_skip, c_int, 1)
        self.W_g = nn.Conv2d(c_gate, c_int, 1)
        self.psi = nn.Conv2d(c_int, 1, 1)
        self.act = nn.ReLU(inplace=True)
        self.last_alpha: torch.Tensor | None = None  # (B,1,Hskip,Wskip)

    def forward(self, x_skip: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        g_up = F.interpolate(g, size=x_skip.shape[-2:], mode="bilinear",
                             align_corners=False)
        q = self.act(self.W_x(x_skip) + self.W_g(g_up))
        alpha = torch.sigmoid(self.psi(q))                 # (B,1,H,W) in [0,1]
        self.last_alpha = alpha.detach().to("cpu")
        return x_skip * alpha


class AGResUNet(nn.Module):
    """Compact AG-ResU-Net. depth=3 -> 3 attention gates (one per skip)."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 16,
                 depth: int = 3):
        super().__init__()
        self.depth = depth
        chs = [base * (2 ** i) for i in range(depth + 1)]   # e.g. 16,32,64,128

        # encoder
        self.enc = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        cin = in_ch
        for d in range(depth):
            self.enc.append(ResBlock(cin, chs[d]))
            cin = chs[d]
        self.bottleneck = ResBlock(chs[depth - 1], chs[depth])

        # decoder + attention gates (named so hooks are easy to find)
        self.up = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.dec = nn.ModuleList()
        for d in reversed(range(depth)):
            self.up.append(nn.ConvTranspose2d(chs[d + 1], chs[d], 2, stride=2))
            self.gates.append(AttentionGate(c_skip=chs[d], c_gate=chs[d + 1],
                                            c_int=max(chs[d] // 2, 8)))
            self.dec.append(ResBlock(chs[d] * 2, chs[d]))

        self.head = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x
        for d in range(self.depth):
            h = self.enc[d](h)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h)

        for i, d in enumerate(reversed(range(self.depth))):
            g = h                       # gating signal BEFORE upsampling
            h = self.up[i](h)
            skip = skips[d]
            gated = self.gates[i](skip, g)         # attention applied here
            h = torch.cat([h, gated], dim=1)
            h = self.dec[i](h)
        return torch.sigmoid(self.head(h))

    def gate_resolutions(self):
        """Human-readable (stage_index, c_skip) for reporting."""
        return [(i, g.W_x.in_channels) for i, g in enumerate(self.gates)]
