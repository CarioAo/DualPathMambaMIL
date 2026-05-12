import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba.mamba_ssm import Mamba


# -----------------------------
# Utils
# -----------------------------
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)


def build_knn_graph(coords, k=8):
    """
    coords: [B, N, 2]
    return normalized adjacency matrix A: [B, N, N]
    """
    B, N, _ = coords.shape
    device = coords.device

    # pairwise distance
    dist = torch.cdist(coords, coords)  # [B, N, N]

    knn_idx = dist.topk(k=k + 1, largest=False)[1][:, :, 1:]
    A = torch.zeros(B, N, N, device=device)

    for b in range(B):
        for i in range(N):
            A[b, i, knn_idx[b, i]] = 1.0

    # symmetric
    A = torch.maximum(A, A.transpose(1, 2))

    # normalize
    D = A.sum(dim=-1, keepdim=True) + 1e-6
    A = A / D
    return A


# -----------------------------
# Dual-Path MambaMIL
# -----------------------------
class DualPathMambaMIL(nn.Module):
    def __init__(
        self,
        in_dim,
        n_classes,
        dropout=0.25,
        act="relu",
        mamba_layers=2,
        k=8
    ):
        super().__init__()

        self.k = k

        # -------- Patch embedding --------
        self.embed = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU() if act.lower() == "relu" else nn.GELU(),
            nn.Dropout(dropout)
        )

        # -------- Sequence Path --------
        self.seq_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(512),
                Mamba(
                    d_model=512,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                    use_fast_path=False 
                )
            )
            for _ in range(mamba_layers)
        ])

        # -------- Spatial Path --------
        self.spa_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(512),
                Mamba(
                    d_model=512,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                    use_fast_path=False
                )
            )
            for _ in range(mamba_layers)
        ])

        # -------- Gated Fusion --------
        self.gate = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(512)

        # -------- MIL Attention --------
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.classifier = nn.Linear(512, n_classes)

        self.apply(initialize_weights)

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(self, x, coords=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        B, N, _ = x.shape

        if coords is None:
            coords = torch.zeros(
                (B, N, 2),
                device=x.device,
                dtype=x.dtype
            )
        elif coords.dim() == 2:
            coords = coords.unsqueeze(0)

        # -------- Embedding --------
        h = self.embed(x)  # [B, N, 512]

        # =========================
        # Sequence Path
        # =========================
        h_seq = h
        for layer in self.seq_layers:
            res = h_seq
            h_seq = layer[0](h_seq)
            h_seq = layer[1](h_seq)
            h_seq = h_seq + res

        # =========================
        # Spatial Path
        # =========================
        A = build_knn_graph(coords, k=self.k)
        h_spa = torch.bmm(A, h)

        for layer in self.spa_layers:
            res = h_spa
            h_spa = layer[0](h_spa)
            h_spa = layer[1](h_spa)
            h_spa = h_spa + res

        # =========================
        # Gated Fusion
        # =========================
        gate = self.gate(torch.cat([h_seq, h_spa], dim=-1))
        h_fuse = gate * h_seq + (1 - gate) * h_spa
        h_fuse = self.norm(h_fuse)

        # =========================
        # MIL Pooling
        # =========================
        A_att = self.attention(h_fuse)  # [B, N, 1]
        A_att = torch.softmax(A_att.transpose(1, 2), dim=-1)
        z = torch.bmm(A_att, h_fuse).squeeze(1)  # [B, 512]

        # =========================
        # Prediction
        # =========================
        logits = self.classifier(z)
        Y_prob = torch.softmax(logits, dim=1)
        Y_hat = torch.argmax(logits, dim=1)

        return logits, Y_prob, Y_hat, A_att, None

    def relocate(self):
        """
        Required by core_utils.py
        Move model to correct device
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)


if __name__ == "__main__":
        # ===============================
        # Sanity Check for DualPathMambaMIL
        # ===============================

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -------- Dummy input --------
        B = 1  # batch size (MIL 通常为 1)
        N = 256  # number of patches
        C = 1024  # patch feature dim
        num_classes = 2

        x = torch.randn(B, N, C).to(device)  # patch features
        coords = torch.randn(B, N, 2).to(device)  # spatial coordinates

        # -------- Model --------
        model = DualPathMambaMIL(
            in_dim=C,
            n_classes=num_classes,
            dropout=0.25,
            act="relu",
            mamba_layers=2,
            k=8
        ).to(device)

        model.eval()

        # -------- Forward --------
        with torch.no_grad():
            logits, probs, preds, attn, _ = model(x, coords)

        # -------- Print results --------
        print("=== DualPathMambaMIL Sanity Check ===")
        print(f"Input features shape   : {x.shape}")
        print(f"Coords shape           : {coords.shape}")
        print(f"Logits shape           : {logits.shape}")
        print(f"Probabilities shape    : {probs.shape}")
        print(f"Predictions shape      : {preds.shape}")
        print(f"Attention map shape    : {attn.shape}")
        print("Forward pass successful ✅")
