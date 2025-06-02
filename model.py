
import torch
import torch.nn as nn

class Simple1DCNN(nn.Module):
    """A lightweight 1‑D CNN for four‑class cotton‑impurity classification."""
    def __init__(self, input_len: int = 288, n_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        conv_out_len = input_len // 4   # 两次 2× 下采样
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * conv_out_len, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, input_len)
        x = self.features(x)
        x = self.classifier(x)
        return x
