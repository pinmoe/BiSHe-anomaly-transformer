import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding
from .dgr_prior import DGRPrior, StaticDGRPrior, MultiScaleDGRPrior


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
        if self.norm is not None:
            x = self.norm(x)
        return x, series_list, prior_list, sigma_list


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True,
                 use_dgr_prior=False, dgr_mode='none'):
        """
        dgr_mode 参数说明（优先级高于 use_dgr_prior）：
          'none'       → E1，原始高斯先验
          'dynamic'    → E2，DGRPrior 动态先验
          'multiscale' → E3，MultiScaleDGRPrior 多尺度动态先验
          'static'     → E4，StaticDGRPrior 静态可学习先验
        """
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # 统一由 dgr_mode 决定结构，use_dgr_prior 仅作 fallback
        if dgr_mode == 'none':
            self.dgr_mode = 'none'
        else:
            self.dgr_mode = dgr_mode

        # 如果外部只传了 use_dgr_prior=True 但没传 dgr_mode，
        # 默认退回 dynamic（E2），保持向后兼容
        if self.dgr_mode == 'none' and use_dgr_prior:
            self.dgr_mode = 'dynamic'

        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout,
                                         output_attention=output_attention),
                        d_model, n_heads),
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

        # 根据 dgr_mode 实例化对应模块，参数键名因此与 checkpoint 一一对应
        if self.dgr_mode == 'multiscale':
            self.dgr_priors = nn.ModuleList(
                [MultiScaleDGRPrior(enc_in, win_size, n_heads, dropout=dropout)
                 for _ in range(e_layers)]
            )
        elif self.dgr_mode == 'static':
            self.dgr_priors = nn.ModuleList(
                [StaticDGRPrior(win_size, n_heads) for _ in range(e_layers)]
            )
        elif self.dgr_mode == 'dynamic':
            self.dgr_priors = nn.ModuleList(
                [DGRPrior(enc_in, n_heads, dropout=dropout) for _ in range(e_layers)]
            )
        else:
            self.dgr_priors = None  # E1：不挂载任何 DGR 模块

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.dgr_priors is not None:
            if self.dgr_mode == 'static':
                prior = [self.dgr_priors[u](x) for u in range(len(prior))]
            else:
                x_smooth = x.mean(dim=1, keepdim=True).expand_as(x)
                prior = [self.dgr_priors[u](x_smooth) for u in range(len(prior))]

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out
