"""IMU-only self-attention pose head (CAVIO ablation); not in upstream VIFT."""

from torch import nn
import torch
import math

class IMUOnlyPoseTransformer(nn.Module):
    """
    Pose transformer variant that performs self-attention on IMU latent tokens:
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
        self.num_layers = num_layers

        self.fc_imu = nn.Linear(i_f_len, embedding_dim)

        self.residual_dropout = nn.Dropout(residual_dropout)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": nn.MultiheadAttention(
                            embed_dim=embedding_dim,
                            num_heads=nhead,
                            dropout=attn_dropout,
                            batch_first=True,
                        ),
                        "ln1": nn.LayerNorm(embedding_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(embedding_dim, dim_feedforward),
                            nn.ReLU(),
                            nn.Dropout(ffn_dropout),
                            nn.Linear(dim_feedforward, embedding_dim),
                        ),
                        "ln2": nn.LayerNorm(embedding_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        self.fc2 = nn.Sequential(
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
        visual_inertial_features, _, _ = batch  # (B, S, v_f_len+i_f_len); visual slice unused
        seq_length = visual_inertial_features.size(1)

        imu = visual_inertial_features[..., self.v_f_len : self.v_f_len + self.i_f_len]

        # (B, S, E)
        imu = self.fc_imu(imu)

        pos_embedding = self.positional_embedding(seq_length).to(imu.device)
        imu = imu + pos_embedding

        # Causal mask so timestep t cannot attend to future visual or IMU tokens.
        attn_mask = self.generate_square_subsequent_mask(
            seq_length, device=imu.device, dtype=imu.dtype
        )

        need_w = self.return_attention_weights
        self_weights_list = []

        for layer in self.layers:
            self_attn_out, self_w = layer["self_attn"](
                query=imu,
                key=imu,
                value=imu,
                attn_mask=attn_mask,
                need_weights=need_w,
                average_attn_weights=True,
            )
            imu = layer["ln1"](imu + self.residual_dropout(self_attn_out))

            ffn_out = layer["ffn"](imu)
            imu = layer["ln2"](imu + self.residual_dropout(ffn_out))

            if need_w:
                self_weights_list.append(self_w)

        out = self.fc2(imu)
        if need_w:
            return out, {"self_attn": self_weights_list}
        return out