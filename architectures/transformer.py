# -*- coding: utf-8 -*-
"""Custom Transformer encoder for Doppler-based activity recognition.

Integrates architectural improvements:
    1. Multi-Head Self-Attention (MHSA) with Learnable Attention Bias
    2. Position-wise Feed-Forward Network (FFN) with GELU
    3. Transformer Encoder Layer with PRE-NORM architecture
    4. Sinusoidal Positional Encoding
    5. Full Transformer Encoder stack with deepcopy fix
    6. Convolutional Stem 1D for sequence length reduction
    7. ViT-style CLS Token for classification
"""

import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout_rate: float = 0.1, max_seq_len: int = 1000):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

        # Learnable attention bias (positional relationships)
        self.attention_bias = nn.Parameter(torch.zeros(nhead, max_seq_len, max_seq_len))

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        Q = self.wq(query).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        K = self.wk(key).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        V = self.wv(value).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)

        # Compute Attention scores
        num = Q @ K.transpose(-2, -1)
        denom = self.head_dim ** 0.5
        z_i = num / denom

        # Add attention bias with bounds check / dynamic interpolation
        if seq_len <= self.attention_bias.shape[1]:
            z_i = z_i + self.attention_bias[:, :seq_len, :seq_len]
        else:
            bias = F.interpolate(
                self.attention_bias.unsqueeze(0),
                size=(seq_len, seq_len),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            z_i = z_i + bias

        if mask is not None:
            z_i = z_i.masked_fill(mask == 0, float('-1e20'))

        attention = F.softmax(z_i, dim=-1)
        x = self.dropout(attention) @ V

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.fc_out(x)


# ---------------------------------------------------------------------------
# 2. Position-wise FFN
# ---------------------------------------------------------------------------

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout_rate: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# ---------------------------------------------------------------------------
# 3. Encoder Layer (Pre-Norm Architecture)
# ---------------------------------------------------------------------------

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int,
                 dim_feedforward: int, dropout_rate: float = 0.1, max_seq_len: int = 1000):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout_rate, max_seq_len)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, src, src_mask=None):
        # Pre-norm architecture
        norm_src = self.norm1(src)
        attn_output = self.self_attn(norm_src, norm_src, norm_src, mask=src_mask)
        src = src + self.dropout1(attn_output)

        norm_src = self.norm2(src)
        ff_output = self.feed_forward(norm_src)
        src = src + self.dropout2(ff_output)
        return src


# ---------------------------------------------------------------------------
# 4. Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 5. Transformer Encoder
# ---------------------------------------------------------------------------

class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        super().__init__()
        # Deepcopy to ensure independent parameters per layer
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


# ---------------------------------------------------------------------------
# 6. Convolutional Stem 2D
# ---------------------------------------------------------------------------

class ConvStem2D(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        # 2D CNN Frontend to extract local spatial features
        self.stem = nn.Sequential(
            # Input: (B, 1, F, T)
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        
        # input_dim = 340. After two stride=2 layers, F_out = 340 // 4 = 85.
        f_out = input_dim // 4
        self.projection = nn.Sequential(
            nn.Linear(32 * f_out, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        # x is (B, T, F). Transform to (B, 1, F, T) for 2D Conv
        batch_size = x.size(0)
        x = x.transpose(1, 2).unsqueeze(1)
        
        # Apply 2D CNN
        x = self.stem(x) # (B, 32, F_out, T_out)
        
        # Flatten spatial dimensions and move Time to sequence dimension
        x = x.permute(0, 3, 1, 2).contiguous() # (B, T_out, 32, F_out)
        x = x.view(batch_size, x.size(1), -1)  # (B, T_out, 32 * F_out)
        
        # Project to Transformer embedding dimension (d_model)
        x = self.projection(x)
        return x


# ---------------------------------------------------------------------------
# 7. Full classification model
# ---------------------------------------------------------------------------

class CustomFullTransformerModel(nn.Module):
    """Transformer-based classifier for Doppler sequences using CLS token / Mean pooling and Conv Stem."""

    def __init__(self, input_dim: int, d_model: int, nhead: int,
                 num_encoder_layers: int, dim_feedforward: int,
                 dropout: float = 0.1, num_classes: int = 5, max_seq_len: int = 1000,
                 pooling: str = "cls"):
        super().__init__()
        self.pooling = pooling
        
        # Hybrid Architecture: 2D Convolutional Stem before Transformer
        self.embedding = ConvStem2D(input_dim=input_dim, d_model=d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout_rate=dropout, max_len=max_seq_len)

        enc_layer = CustomTransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout_rate=dropout, max_seq_len=max_seq_len
        )
        self.transformer_encoder = CustomTransformerEncoder(enc_layer, num_encoder_layers)
        self.d_model = d_model

        # ViT style CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.classification_layer = nn.Linear(self.d_model, num_classes)

        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.classification_layer.bias)
        nn.init.xavier_uniform_(self.classification_layer.weight)
        nn.init.xavier_uniform_(self.cls_token)

    def forward(self, src, src_mask=None):
        batch_size = src.shape[0]

        # Convolutional Stem
        src = self.embedding(src)

        # Append CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        src = torch.cat((cls_tokens, src), dim=1)

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)

        # Extract representation for classification
        if self.pooling == "mean":
            # Mean pool sequence tokens (excluding CLS token) for metric learning / fine gait details
            output = output[:, 1:].mean(dim=1) if output.shape[1] > 1 else output[:, 0]
        else:
            # Extract CLS token
            output = output[:, 0]
            
        return self.classification_layer(output)
