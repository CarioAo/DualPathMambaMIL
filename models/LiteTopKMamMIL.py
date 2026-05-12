
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Mamba =====
from mamba.mamba_ssm import SRMamba

# ===== FLOPs / Params =====
from thop import profile

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
class LiteTopKMambaMIL(nn.Module):
    def __init__(
        self,
        in_dim,
        n_classes,
        k_ratio=0.1,
        dropout=0.25,
        act="gelu",
    ):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU() if act == "gelu" else nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ===== instance scorer (very cheap) =====
        self.scorer = nn.Linear(512, 1)

        # ===== single-layer Mamba =====
        self.mamba = nn.Sequential(
            nn.LayerNorm(512),
            SRMamba(
                d_model=512,
                d_state=16,
                d_conv=4,
                expand=2,
            )
        )

        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.norm = nn.LayerNorm(512)
        self.classifier = nn.Linear(512, n_classes)
        self.k_ratio = k_ratio

        self.apply(initialize_weights)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        h = self.fc(x)                    # [B, N, 512]
        scores = self.scorer(h).squeeze(-1)  # [B, N]

        # ===== Top-k selection =====
        k = max(1, int(h.size(1) * self.k_ratio))
        topk_idx = torch.topk(scores, k, dim=1)[1]
        h = torch.gather(
            h, 1, topk_idx.unsqueeze(-1).expand(-1, -1, h.size(-1))
        )  # [B, k, 512]

        # ===== Light Mamba =====
        h = self.mamba(h)
        h = self.norm(h)

        # ===== Attention Pooling =====
        A = self.attention(h).transpose(1, 2)
        A = F.softmax(A, dim=-1)
        h = torch.bmm(A, h).squeeze(1)

        logits = self.classifier(h)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]

        return logits, Y_prob, Y_hat, None, None

# ============================================================
# Test for LiteTopKMambaMIL
# ============================================================
if __name__ == "__main__":
    import torch
    from thop import profile

    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ================= 配置 =================
    in_dim = 1024
    n_classes = 2
    n_instances = 1000
    batch_size = 1
    k_ratio = 0.1   # Top-k = 100

    # ================= 模型 =================
    model = LiteTopKMambaMIL(
        in_dim=in_dim,
        n_classes=n_classes,
        k_ratio=k_ratio,
        dropout=0.25,
        act="gelu",
    ).to(device)

    model.eval()

    # ================= 输入 =================
    x = torch.randn(batch_size, n_instances, in_dim).to(device)

    # ================= Forward =================
    with torch.no_grad():
        logits, Y_prob, Y_hat, A_raw, results_dict = model(x)

    print("\n=== LiteTopKMambaMIL Forward Test ===")
    print("Input shape:", x.shape)
    print("logits shape:", logits.shape)
    print("Y_prob shape:", Y_prob.shape)
    print("Y_hat:", Y_hat)

    # ================= 参数量 =================
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params / 1e6:.3f} M")

    # ================= FLOPs =================
    try:
        flops, params = profile(model, inputs=(x,), verbose=False)
        print(f"Estimated FLOPs: {flops / 1e9:.3f} GFLOPs")
    except Exception as e:
        print("[Warning] FLOPs estimation failed")
        print("Reason:", e)
        print("Theoretical complexity: O(N·D + k·D), k = N * k_ratio")

    # ================= 对比提示 =================
    print("\n[Info]")
    print(f"Top-k instances: {int(n_instances * k_ratio)} / {n_instances}")
    print("Expected FLOPs reduction ≈ {:.1f}x".format(1 / k_ratio))
