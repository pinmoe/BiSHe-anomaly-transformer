import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding
from .dgr_prior import DGRPrior, StaticDGRPrior, USE_STATIC_DGR


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
                 dropout=0.0, activation='gelu', output_attention=True, use_dgr_prior=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention
        self.use_dgr_prior = use_dgr_prior

        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # 原始 encoder，attn 完全不改动
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

        # DGR prior 模块：每层独立，注入点在 forward()
        if use_dgr_prior:
            if USE_STATIC_DGR:
                # 静态可学习先验（推荐，实验 E4）
                self.dgr_priors = nn.ModuleList(
                    [StaticDGRPrior(win_size, n_heads) for _ in range(e_layers)]
                )
            else:
                # 动态先验（消融对照，实验 E2）
                self.dgr_priors = nn.ModuleList(
                    [DGRPrior(enc_in, n_heads, dropout=dropout) for _ in range(e_layers)]
                )
        else:
            self.dgr_priors = None

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        # 用 DGR prior 替换高斯 prior，attn.py 零侵入
        if self.dgr_priors is not None:
            # StaticDGRPrior: x_input 数值被忽略，只取 B
            # DGRPrior:       用窗口均值输入，降低对瞬时异常的敏感性（方案1）
            x_smooth = x.mean(dim=1, keepdim=True).expand_as(x)
            prior = [
                self.dgr_priors[u](x_smooth if not USE_STATIC_DGR else x)
                for u in range(len(prior))
            ]

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out
