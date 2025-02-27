import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerIntegrator(nn.Module):
    def __init__(self, num_layers, hidden_dim, integration_type="static"):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.integration_type = integration_type
        
        # static weight for test
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.projection = None
        
    def forward(self, hidden_states):
        """
        input:
            hidden_states: list with all hidden state [batch_size, seq_len, hidden_dim]
        output:
            integrated_state: onr hidden state after integration [batch_size, seq_len, hidden_dim]
            weight: for analysis
        """
        # Normalization weight
        weights = F.softmax(self.layer_weights, dim=0)
        
        # Relable all the hidden state into one
        x = sum(w * state for w, state in zip(weights, hidden_states))
        
        # Possible projection layer
        if self.projection is not None:
            x = self.projection(x)
        
        normalized_x = F.layer_norm(x, x.size()[1:])
            
        return normalized_x, weights