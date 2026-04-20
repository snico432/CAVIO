from torch import nn
import torch
import math


class BidirectionalCrossAttentionPoseTransformer(nn.Module):
    """
    Bidirectional cross-attention fusion block.

    Each layer keeps visual and IMU streams at ``embedding_dim`` each, runs
    cross-attention in both directions, concatenates the updated streams, and
    applies a fused FFN so the next layer receives a ``2 * embedding_dim`` token.
    """

    def __init__(
        self,
        input_dim=768,
        v_f_len=512,
        i_f_len=256,
        embedding_dim=512,
        num_layers=4,
        nhead=8,
        dim_feedforward=1024,
        attn_dropout=0.0,
        residual_dropout=0.0,
        ffn_dropout=0.0,
        return_attention_weights=False,
    ):
        super().__init__()

        if v_f_len + i_f_len != input_dim:
            raise ValueError(
                f"Expected input_dim={input_dim} to equal v_f_len+i_f_len={v_f_len+i_f_len}."
            )

        self.return_attention_weights = return_attention_weights
        self.v_f_len = v_f_len
        self.i_f_len = i_f_len
        self.embedding_dim = embedding_dim
        self.fused_dim = 2 * embedding_dim

        self.fc_visual = nn.Linear(v_f_len, embedding_dim)
        self.fc_imu = nn.Linear(i_f_len, embedding_dim)
        self.residual_dropout = nn.Dropout(residual_dropout)

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "visual_queries_imu": nn.MultiheadAttention(
                            embed_dim=embedding_dim,
                            num_heads=nhead,
                            dropout=attn_dropout,
                            batch_first=True,
                        ),
                        "imu_queries_visual": nn.MultiheadAttention(
                            embed_dim=embedding_dim,
                            num_heads=nhead,
                            dropout=attn_dropout,
                            batch_first=True,
                        ),
                        "visual_ln": nn.LayerNorm(embedding_dim),
                        "imu_ln": nn.LayerNorm(embedding_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(self.fused_dim, dim_feedforward),
                            nn.ReLU(),
                            nn.Dropout(ffn_dropout),
                            nn.Linear(dim_feedforward, self.fused_dim),
                        ),
                        "fused_ln": nn.LayerNorm(self.fused_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.fused_dim, embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(embedding_dim, 6),
        )

    def positional_embedding(self, seq_length: int) -> torch.Tensor:
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2).float()
            * -(math.log(10000.0) / self.embedding_dim)
        )
        pos_embedding = torch.zeros(seq_length, self.embedding_dim)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        return pos_embedding.unsqueeze(0)

    def generate_square_subsequent_mask(self, sz: int, device=None, dtype=None) -> torch.Tensor:
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        return torch.triu(
            torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
            diagonal=1,
        )

    def forward(self, batch, gt):
        visual_inertial_features, _, _ = batch
        seq_length = visual_inertial_features.size(1)

        visual = visual_inertial_features[..., : self.v_f_len]
        imu = visual_inertial_features[..., self.v_f_len : self.v_f_len + self.i_f_len]

        visual = self.fc_visual(visual)
        imu = self.fc_imu(imu)

        pos_embedding = self.positional_embedding(seq_length).to(visual.device)
        visual = visual + pos_embedding
        imu = imu + pos_embedding

        attn_mask = self.generate_square_subsequent_mask(
            seq_length, device=visual.device, dtype=visual.dtype
        )

        need_w = self.return_attention_weights
        visual_to_imu_weights = []
        imu_to_visual_weights = []

        for layer in self.layers:
            visual_cross, visual_w = layer["visual_queries_imu"](
                query=visual,
                key=imu,
                value=imu,
                attn_mask=attn_mask,
                need_weights=need_w,
                average_attn_weights=True,
            )
            imu_cross, imu_w = layer["imu_queries_visual"](
                query=imu,
                key=visual,
                value=visual,
                attn_mask=attn_mask,
                need_weights=need_w,
                average_attn_weights=True,
            )

            visual = layer["visual_ln"](visual + self.residual_dropout(visual_cross))
            imu = layer["imu_ln"](imu + self.residual_dropout(imu_cross))

            fused = torch.cat([visual, imu], dim=-1)
            fused = layer["fused_ln"](fused + self.residual_dropout(layer["ffn"](fused)))
            visual, imu = torch.split(fused, self.embedding_dim, dim=-1)

            if need_w:
                visual_to_imu_weights.append(visual_w)
                imu_to_visual_weights.append(imu_w)

        out = self.fc2(fused)
        if need_w:
            return out, {
                "visual_queries_imu": visual_to_imu_weights,
                "imu_queries_visual": imu_to_visual_weights,
            }
        return out
