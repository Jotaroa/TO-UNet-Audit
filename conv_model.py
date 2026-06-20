"""
conv_model.py
=============
AG-ResU-Net (NN4TopOptUNet) for topology-optimization prediction.

Kien truc:
  - Encoder/Decoder U-Net 3 level (base_filters = 32 -> 64 -> 128)
  - ResConvBlock: Conv-BN-(SiLU) -> Conv-BN -> SEBlock (channel attention) -> +residual -> SiLU
  - AttentionGate: spatial attention tren skip connection
Input : [B, 2, 40, 40]   (channel 0 = mat do vong lap dau, channel 1 = gradient SIMP)
Output: [B, 1, 40, 40]   logits (chua qua sigmoid)
"""
import torch
import torch.nn as nn


# =====================================================================
# 1. SE BLOCK (Channel Attention)
# =====================================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation: hoc kenh nao quan trong, tang global context."""
    def __init__(self, in_ch, r=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, max(in_ch // r, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_ch // r, 4), in_ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# =====================================================================
# 2. RESIDUAL CONV BLOCK (Encoder/Decoder building block)
# =====================================================================
class ResConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN -> SE -> +residual -> SiLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.identity = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.se = SEBlock(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = self.identity(x)
        out = self.conv(x)
        out = self.se(out)
        out += residual
        return self.act(out)


# =====================================================================
# 3. ATTENTION GATE (Spatial Attention tren Skip Connection)
# =====================================================================
class AttentionGate(nn.Module):
    """Gating tu decoder de loc encoder feature tren skip connection."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# =====================================================================
# 4. AG-ResU-Net (Full model)
# =====================================================================
class NN4TopOptUNet(nn.Module):
    """U-Net + Residual + SE-Block + Attention Gates. Input 2 kenh, output logits."""
    def __init__(self, in_channels=2, out_channels=1, base_filters=32):
        super().__init__()
        # --- ENCODER ---
        self.enc1 = ResConvBlock(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResConvBlock(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResConvBlock(base_filters * 2, base_filters * 4)   # bottleneck

        # --- DECODER ---
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att2 = AttentionGate(F_g=base_filters * 4, F_l=base_filters * 2, F_int=base_filters * 2)
        self.dec2 = ResConvBlock(base_filters * 4 + base_filters * 2, base_filters * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att1 = AttentionGate(F_g=base_filters * 2, F_l=base_filters, F_int=base_filters)
        self.dec1 = ResConvBlock(base_filters * 2 + base_filters, base_filters)

        self.out_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        p1 = self.pool1(x1)
        x2 = self.enc2(p1)
        p2 = self.pool2(x2)
        x3 = self.enc3(p2)
        # Decoder block 2
        u2 = self.up2(x3)
        x2_gated = self.att2(g=u2, x=x2)
        u2 = torch.cat([u2, x2_gated], dim=1)
        x_dec2 = self.dec2(u2)
        # Decoder block 1
        u1 = self.up1(x_dec2)
        x1_gated = self.att1(g=u1, x=x1)
        u1 = torch.cat([u1, x1_gated], dim=1)
        x_dec1 = self.dec1(u1)
        # Output logits (no sigmoid)
        return self.out_conv(x_dec1)


if __name__ == "__main__":
    model = NN4TopOptUNet(in_channels=2, out_channels=1, base_filters=32)
    out = model(torch.randn(2, 2, 40, 40))
    print("Output shape:", out.shape)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
