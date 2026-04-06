import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
# B = Batch Size (每次并行处理样本数)
# W = Window length (每个样本里有多少时间步)
# C = Channels (传感器/特征通道数)
# D = Embedding dimension (编码后的潜变量维度)

class VitEncoder(nn.Module):
    def __init__(
        self,
        out_dim=256,
        channels: int = 25,
        model_name='tiny_graph',
        pretrained=False,
        image_size=224,
        freeze_backbone=False,
        use_layernorm=True,
        tiny_graph_layers=2,
        tiny_graph_heads=4,
        tiny_graph_ff_mult=2,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.channels = int(channels)
        self.image_size = image_size
        self.freeze_backbone = freeze_backbone
        self.model_name = str(model_name).lower()
        self.use_tiny_graph = self.model_name in {'tiny_graph', 'graph_transformer'}

        if self.use_tiny_graph:
            # 用基于相关矩阵的图卷积编码器替代随机 TransformerEncoder。
            # CorrelationGCNEncoder 直接处理 (B,1,C,C) 输入并输出 (B, D)，
            # 内部已包含投影和 LayerNorm，因此 self.proj 和 self.norm 退化为 Identity。
            self.backbone = CorrelationGCNEncoder(
                channels=self.channels,
                out_dim=out_dim,
                hidden_dim=None,
                dropout=0.1,
                use_layernorm=True,
            )
            # proj 和 norm 退化为 Identity，保留属性避免 optimizer 报 KeyError
            self.proj = nn.Identity()
            self.norm = nn.Identity()
        else:
            # 使用 timm 开源 ViT，num_classes=0 让模型输出特征而非分类 logits。
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool='avg',
            )
            self.proj = nn.Linear(self.backbone.num_features, out_dim)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.norm = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        # 期望 x_img: (B, 1, C, C)
        if x_img.ndim != 4:
            raise ValueError(f"VitEncoder 输入必须是 4D，当前 {tuple(x_img.shape)}")

        if x_img.shape[-1] != x_img.shape[-2]:
            raise ValueError(
                f"VitEncoder 输入的空间尺寸必须是方阵，当前是 {tuple(x_img.shape[-2:])}"
            )

        if self.use_tiny_graph:
            # CorrelationGCNEncoder 直接接受 (B, 1, C, C) 输入
            return self.backbone(x_img)

        # timm ViT 路径：默认保持兼容，单通道复制到 3 通道后按 image_size 插值。
        if x_img.shape[1] == 1:
            x_img = x_img.repeat(1, 3, 1, 1)
        elif x_img.shape[1] != 3:
            raise ValueError(f"VitEncoder 通道数必须是 1 或 3，当前是 {x_img.shape[1]}")

        if x_img.shape[-1] != self.image_size or x_img.shape[-2] != self.image_size:
            x_img = F.interpolate(
                x_img,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            )

        feat = self.backbone(x_img)   # (B, F)
        out = self.proj(feat)         # (B, D)
        out = self.norm(out)          # (B, D)
        return out


class CorrelationGCNEncoder(nn.Module):
    """基于 DGR 相关矩阵的轻量图卷积编码器。

    将 (B, 1, C, C) 的相关矩阵视为带权无向图：
      - 节点数 = C（传感器数）
      - 节点特征 = 该传感器与所有其他传感器的相关系数向量，形状 (C,)
      - 邻接矩阵 = 相关矩阵本身（对称，值域 [0,1] 经 normalize_image 后）

    两层谱域图卷积（Kipf & Welling 2017 简化版）：
        H^(l+1) = ReLU( D^{-1/2} A D^{-1/2} H^(l) W^(l) )
    其中 A = corr + I（加自环），D 为度矩阵。
    最终全局均值池化输出图级嵌入 (B, D)。
    """

    def __init__(
        self,
        channels: int,
        out_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.channels = int(channels)
        self.out_dim = int(out_dim)
        hidden = int(hidden_dim) if hidden_dim is not None else max(out_dim, channels * 2)

        # 两层图卷积权重矩阵
        self.gcn1 = nn.Linear(channels, hidden, bias=False)
        self.gcn2 = nn.Linear(hidden, out_dim, bias=False)

        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()

        # 输出投影
        self.out_proj = nn.Linear(out_dim, out_dim)

        # 参数初始化
        nn.init.xavier_uniform_(self.gcn1.weight)
        nn.init.xavier_uniform_(self.gcn2.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        """对称归一化邻接矩阵：D^{-1/2} (A+I) D^{-1/2}。"""
        eye = torch.eye(adj.shape[-1], dtype=adj.dtype, device=adj.device).unsqueeze(0)
        a_hat = adj + eye                                        # (B, C, C)
        deg = a_hat.sum(dim=-1, keepdim=True).clamp(min=1e-6)   # (B, C, 1)
        d_inv_sqrt = deg.pow(-0.5)                               # (B, C, 1)
        a_norm = a_hat * d_inv_sqrt                              # (B, C, C)
        a_norm = a_norm * d_inv_sqrt.transpose(1, 2)             # (B, C, C)
        return a_norm

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_img: (B, 1, C, C)，来自 build_dgr_batch 的相关矩阵图像
        Returns:
            (B, D) 图级嵌入
        """
        if x_img.ndim != 4:
            raise ValueError(f"CorrelationGCNEncoder 输入必须是 4D (B,1,C,C)，当前 {tuple(x_img.shape)}")

        corr = x_img[:, 0]                                      # (B, C, C)
        a_norm = self._normalize_adj(corr)                      # (B, C, C)
        h = corr                                                 # (B, C, C)

        # 第一层图卷积
        h = torch.bmm(a_norm, h)                                # (B, C, C)
        h = self.gcn1(h)                                        # (B, C, hidden)
        h = self.act(h)
        h = self.drop(h)

        # 第二层图卷积
        h = torch.bmm(a_norm, h)                                # (B, C, hidden)
        h = self.gcn2(h)                                        # (B, C, out_dim)

        # 全局均值池化
        out = h.mean(dim=1)                                     # (B, out_dim)
        out = self.out_proj(out)                                 # (B, out_dim)
        out = self.norm(out)
        return out


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1,
            self.conv2, self.chomp2, self.relu2, self.drop2
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.out_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.out_relu(out + res)


class TcnEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 25,
        out_dim: int = 256,
        hidden_channels=(64, 128, 128, 128, 128),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        # 感受野公式：RF = 1 + 2*(kernel_size-1)*sum(2^i for i in range(n_layers))
        # kernel_size=3 时：3层→RF=29步，4层→RF=61步，5层→RF=125步
        # 要完整覆盖 W=100 的窗口，需要 n_layers >= 5（默认已调整为5层）
        super().__init__()
        self.out_dim = out_dim
        self.in_channels = in_channels

        layers = []
        prev = in_channels
        for i, ch in enumerate(hidden_channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_ch=prev,
                    out_ch=ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            prev = ch

        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(prev, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x_seq: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        # 期望 x_seq: (B, W, C)
        if x_seq.ndim != 3:
            raise ValueError(f"TcnEncoder 输入必须是 3D，当前 {tuple(x_seq.shape)}")
        if x_seq.shape[-1] != self.in_channels:
            raise ValueError(
                f"TcnEncoder 输入通道数应为 {self.in_channels}，当前是 {x_seq.shape[-1]}"
            )

        # Conv1d 期望 (B, C, W)
        x = x_seq.transpose(1, 2)
        h = self.tcn(x)

        if return_sequence:
            # 返回时序特征: (B, W, D)
            h_seq = h.transpose(1, 2)     # (B, W, H)
            y_seq = self.proj(h_seq)      # (B, W, D)
            y_seq = self.norm(y_seq)
            return y_seq

        h = self.pool(h).squeeze(-1)   # (B, H)
        y = self.proj(h)               # (B, D)
        y = self.norm(y)
        return y
    