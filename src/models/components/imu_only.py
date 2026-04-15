from torch import nn
import torch
import math

class IMUOnlyPoseTransformer(nn.Module):
    """
    Pose transformer variant that only uses IMU features.
    """

    def __init__(
        self,
        input_dim=768,
        v_f_len=512,
        i_f_len=256,
        embedding_dim=768,
        num_layers=2,
        nhead=8,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()

        if v_f_len + i_f_len != input_dim:
            raise ValueError(
                f"Expected input_dim={input_dim} to equal v_f_len+i_f_len={v_f_len+i_f_len}."
            )

        self.v_f_len = v_f_len
        self.i_f_len = i_f_len
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.imu_embed = nn.Linear(i_f_len, embedding_dim)

        self.dropout = nn.Dropout(dropout)
        self.transformer_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": nn.MultiheadAttention(
                            embed_dim=embedding_dim,
                            num_heads=nhead,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "norm_after_self_attn": nn.LayerNorm(embedding_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(embedding_dim, dim_feedforward),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(dim_feedforward, embedding_dim),
                        ),
                        "norm_after_ffn": nn.LayerNorm(embedding_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        self.pose_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.embedding_dim, 6),
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
        pos_embedding = pos_embedding.unsqueeze(0)  # (1, S, E)
        return pos_embedding

    def generate_square_subsequent_mask(self, sz: int, device=None, dtype=None) -> torch.Tensor:
        # Additive attn mask with -inf above the diagonal.
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        return torch.triu(
            torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
            diagonal=1,
        )

    def forward(self, batch, gt):
        visual_inertial_features, _, _ = batch  # (B, S, v_f_len+i_f_len)
        seq_length = visual_inertial_features.size(1)

        imu = visual_inertial_features[..., self.v_f_len : self.v_f_len + self.i_f_len]

        # (B, S, E)
        imu = self.imu_embed(imu)

        pos_embedding = self.positional_embedding(seq_length).to(imu.device)
        imu = imu + pos_embedding

        # Causal mask so timestep t cannot attend to future IMU tokens.
        attn_mask = self.generate_square_subsequent_mask(
            seq_length, device=imu.device, dtype=imu.dtype
        )

        for layer in self.transformer_layers:
            self_attn_out, _ = layer["self_attn"](
                query=imu,
                key=imu,
                value=imu,
                attn_mask=attn_mask,
                need_weights=False,
            )
            imu = layer["norm_after_self_attn"](imu + self.dropout(self_attn_out))

            ffn_out = layer["ffn"](imu)
            imu = layer["norm_after_ffn"](imu + self.dropout(ffn_out))

        # (B, S, 6)
        return self.pose_head(imu)