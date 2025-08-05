import torch
import torch.nn as nn
from .network_base import NetworkBase


class AttentionNetwork(NetworkBase):
    def __init__(
        self,
        num_input_dim,
        num_output_dim,
        num_heads=1,
        orthogonal_init=False,
        normalizer=None,
        obs_history_length=1,
    ):
        super().__init__(num_input_dim, num_output_dim, orthogonal_init, normalizer)
        self.obs_history_length = obs_history_length
        self.attention_input_dim = obs_history_length
        self.single_obs_dim = num_input_dim // obs_history_length
        
        self.atten = nn.MultiheadAttention(self.attention_input_dim, num_heads)
        self.head = nn.Linear(self.attention_input_dim, num_output_dim)
        self.activation = nn.ReLU()

    def pure_forward(self, x):
        """
        Pure forward method for the attention network.

        Args:
            x (tensor): Input tensor of shape (batch_size, contact_input_dim)

        Returns:
            (tensor, tensor): Output tensor and attention weights

        Description:
            This method reshapes the input tensor from (batch_size, contact_input_dim) to (batch_size, obs_history_length, single_obs_dim),
            then transposes it to (batch_size, single_obs_dim, obs_history_length), so that the last dimension corresponds to the observation history for each feature.
            Then, it applies multi-head attention to the reshaped input, computes the mean across the sequence length, applies a ReLU activation,
            and finally passes the result through a linear layer to produce the output.
            The attention weights are also returned.
        """
        x = x.reshape(
            -1, self.obs_history_length, self.single_obs_dim
        )   # (batch_size, obs_history_length, single_obs_dim)
        x = x.transpose(1, 2)   # (batch_size, single_obs_dim, obs_history_length)
        x = x.transpose(0, 1)   # (obs_history_length, batch_size, single_obs_dim) to fit attention input format

        x, attention_weight = self.atten(x, x, x)
        x = x.mean(dim=0)
        x = self.activation(x)
        return self.head(x), attention_weight

