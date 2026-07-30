import torch.nn as nn
from architectures import ResNet8, SHARPInceptionNet
from architectures.transformer import CustomFullTransformerModel

def build_model(arch: str, num_classes: int, dropout: float, task: str = "activity") -> nn.Module:
    if arch == "resnet8":
        return ResNet8(num_classes=num_classes, dropout_rate=dropout)
    elif arch == "inception":
        return SHARPInceptionNet(num_classes=num_classes, dropout_rate=dropout)
    elif arch == "transformer":
        pooling = "mean" if task == "person_id" else "cls"
        return CustomFullTransformerModel(
            input_dim=340,        # Doppler feature dimension
            d_model=128,
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=256,
            dropout=dropout,
            num_classes=num_classes,
            max_seq_len=1000,
            pooling=pooling,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
