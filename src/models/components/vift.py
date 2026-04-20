"""Minimal VIFT-style causal transformer head over concatenated visual+IMU latents.

:class:`PoseTransformer` matches the core of VIFT ``components/pose_transformer.py``
(https://github.com/ybkurt/vift): linear embed, sinusoidal position, causal
``TransformerEncoder``, 6-DoF head. The upstream file may contain additional variants;
this repo keeps a small copy for compatibility (e.g. ``safe_globals``).
"""

from torch import nn
import torch
import math

class PoseTransformer(nn.Module):
    def __init__(self, input_dim=768, embedding_dim=128, num_layers=2, nhead=8, dim_feedforward=512, dropout=0.1):
        super(PoseTransformer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.embedding_dim, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ), 
            num_layers=num_layers
        )
        # Add the fully connected layer
        self.fc2 = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.embedding_dim, 6))
    
    def positional_embedding(self, seq_length):
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * -(math.log(10000.0) / self.embedding_dim))
        pos_embedding = torch.zeros(seq_length, self.embedding_dim)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        pos_embedding = pos_embedding.unsqueeze(0)
        return pos_embedding

    def generate_square_subsequent_mask(self, sz, device=None, dtype=None):
        """Generate a square causal mask for sequence.
        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        return torch.triu(
                torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
                diagonal=1
        )


    def forward(self, batch, gt):
        visual_inertial_features, _, _ = batch
        seq_length = visual_inertial_features.size(1)

        # Generate causal mask
        pos_embedding = self.positional_embedding(seq_length).to(visual_inertial_features.device)
        visual_inertial_features = self.fc1(visual_inertial_features)
        visual_inertial_features += pos_embedding

        
        # Passing through the transformer encoder with the mask
        mask = self.generate_square_subsequent_mask(seq_length, visual_inertial_features.device)
        output = self.transformer_encoder(visual_inertial_features, mask=mask, is_causal=True)
        #output = self.transformer_encoder(visual_inertial_features, mask=None)

        # Pass the output through the fully connected layer
        output = self.fc2(output)

        return output
