import torch
import torch.nn as nn
from .network_base import NetworkBase


class AttentionNetwork(NetworkBase):
    def __init__(
        self,
        num_input_dim,
        num_output_dim,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        orthogonal_init=False,
        normalizer=None,
    ):
        super().__init__(num_input_dim, num_output_dim, orthogonal_init, normalizer)
        self.embed = nn.Linear(num_input_dim, hidden_dim)
        self.attn_layers = nn.ModuleList(
            [nn.MultiheadAttention(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim, num_output_dim)
        self.activation = nn.ReLU()

    def pure_forward(self, x):
        # x expected shape: (batch, seq_len, dim)
        x = self.embed(x)
        x = x.transpose(0, 1)  # seq_len, batch, hidden
        for attn in self.attn_layers:
            x, _ = attn(x, x, x)
        x = x.mean(dim=0)
        x = self.activation(x)
        return self.head(x)
