"""dgr_prior.py
Prior association 模块，替代原始高斯核，注入点在 AnomalyTransformer.forward()。

提供两种实现，通过 USE_STATIC_DGR 开关切换：
  USE_STATIC_DGR = True  → StaticDGRPrior（推荐，实验 E4）
                           可学习静态先验，与当前输入无关，测试时不随异常偏移
  USE_STATIC_DGR = False → DGRPrior（消融对照，实验 E2）
                           动态先验，基于当前窗口输入计算，异常时会自适应（有害）

归一化保证：softmax 后每行 sum=1，满足 KL 散度要求。
solver.py 中的 prior/row_sum 归一化对已归一化输出无影响（除以1）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 切换开关
USE_STATIC_DGR = True


class DGRPrior(nn.Module):
    """
    动态 DGR Prior（消融对照 E2）。
    输入: x_seq (B, W, C)
    输出: prior (B, H, W, W)，每行 sum=1
    """
    def __init__(self, in_channels: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = max(in_channels // n_heads, 8)
        self.scale = self.head_dim ** -0.5
        self.proj = nn.Linear(in_channels, n_heads * self.head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.proj.weight, std=0.01)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, W, C = x_seq.shape
        H = self.n_heads
        feat = self.proj(x_seq).view(B, W, H, self.head_dim).permute(0, 2, 1, 3)
        feat = self.dropout(feat)
        sim = torch.matmul(feat, feat.transpose(-1, -2)) * self.scale
        return F.softmax(sim, dim=-1)


class StaticDGRPrior(nn.Module):
    """
    静态可学习 DGR Prior（推荐，实验 E4）。
    prior 是 (H, W, W) 可学习参数，与输入完全无关，测试时固定不变。
    输入: x_seq (B, W, C) — 只用于取 B，数值被忽略
    输出: prior (B, H, W, W)，每行 sum=1
    """
    def __init__(self, win_size: int, n_heads: int):
        super().__init__()
        self.prior_logits = nn.Parameter(torch.zeros(n_heads, win_size, win_size))
        nn.init.normal_(self.prior_logits, std=0.01)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B = x_seq.shape[0]
        prior = F.softmax(self.prior_logits, dim=-1)
        return prior.unsqueeze(0).expand(B, -1, -1, -1)
