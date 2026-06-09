import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.act1 = nn.ReLU(True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.act2 = nn.ReLU(True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = out + x
        out = self.act2(out)
        return out

class MicroactionDecoderTCN(nn.Module):
    """Decode latent tokens into hand coordinates.

    Args:
        z: (B, M, L, D) where L = num_joints * num_hands and D = embedding_dim.
    Returns:
        coords: (B, M, C, T, J, E) where C=coord_dim and T=num_frames.
    """
    def __init__(
        self,
        num_frames: int,          # MicroactionEncoder  num_frames
        num_joints: int,
        num_hands: int,
        embedding_dim: int = 256, # Encoder  embedding_dim
        coord_dim: int = 3,       # 2  3
        width: int = 256,
        num_upsamples: int = 3,   # 2^num_upsamples  num_frames
        kernel_size: int = 3,
        use_final_interp: bool = True
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.num_hands = num_hands
        self.embedding_dim = embedding_dim
        self.coord_dim = coord_dim

        self.proj_in = nn.Linear(embedding_dim, width)

        blocks = []
        # 1
        for _ in range(num_upsamples):
            blocks.append(ResidualBlock1D(width, kernel_size=kernel_size, dilation=1))
            blocks.append(nn.ConvTranspose1d(width, width, kernel_size=2, stride=2))
            blocks.append(nn.ReLU(True))
        blocks.append(ResidualBlock1D(width, kernel_size=kernel_size, dilation=1))
        self.tcn = nn.Sequential(*blocks)

        self.head = nn.Conv1d(width, coord_dim, kernel_size=1)
        self.use_final_interp = use_final_interp

    def forward(self, z):  # (B, M, L, D)
        B, M, L, D = z.shape
        assert D == self.embedding_dim, f"D mismatch: {D}!={self.embedding_dim}"
        assert L == self.num_joints * self.num_hands, f"L mismatch: {L}!={self.num_joints*self.num_hands}"

        z = z.view(B * M * L, D)                 # (BML, D)
        x = self.proj_in(z)                      # (BML, width)
        x = x.unsqueeze(-1)                      # (BML, width, 1)

        x = self.tcn(x)                          # (BML, width, T_dec)
        x = self.head(x)                         # (BML, C, T_dec)

        T_dec = x.shape[-1]
        if self.use_final_interp and T_dec != self.num_frames:
            x = F.interpolate(x, size=self.num_frames, mode="linear", align_corners=False)  # (BML, C, T)

        # (B, M, L, C, T) -> (B, M, C, T, J, E)
        x = x.view(B, M, L, self.coord_dim, self.num_frames)                    # (B, M, L, C, T)
        x = x.view(B, M, self.num_joints, self.num_hands, self.coord_dim, self.num_frames)
        coords = x.permute(0, 1, 4, 5, 2, 3).contiguous()                       # (B, M, C, T, J, E)
        return coords


class SimpleHandDecoderTCN(nn.Module):

    def __init__(
        self,
        num_frames: int = 16,
        input_frames: int = 2,
        embedding_dim: int = 768,
        mlp_dim: int = 512,  # MLPTCN
        width: int = 512,    # TCN
        num_upsamples: int = 3,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.input_frames = input_frames

        # MLP - TCN
        self.wrist_mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, width),  # widthTCN
        )

        # MLP - TCN
        self.finger_mlp = nn.Sequential(
            nn.Linear(10 * embedding_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, width),  # widthTCN
        )

        # TCN - width
        self.tcn = self._build_tcn(width, num_upsamples, kernel_size)


        self.wrist_head = nn.Conv1d(width, 18, kernel_size=1)  # 2*9=18
        self.finger_head = nn.Conv1d(width, 30, kernel_size=1) # 10*3=30

    def _build_tcn(self, width, num_upsamples, kernel_size):
        blocks = []
        for i in range(num_upsamples):
            blocks.append(ResidualBlock1D(width, kernel_size=kernel_size, dilation=1))
            blocks.append(nn.ConvTranspose1d(width, width, kernel_size=2, stride=2))
            blocks.append(nn.ReLU(True))
        blocks.append(ResidualBlock1D(width, kernel_size=kernel_size, dilation=1))
        return nn.Sequential(*blocks)

    def forward(self, z):  # (B, 2, 12, 768)
        B, T_in, N, D = z.shape


        wrist_input = z[:, :, :2, :]   # (B, 2, 2, 768)
        finger_input = z[:, :, 2:, :]  # (B, 2, 10, 768)


        wrist_flat = wrist_input.reshape(B, T_in, -1)  # (B, 2, 2*768=1536)
        wrist_mlp_out = self.wrist_mlp(wrist_flat)     # (B, 2, width=512)
        wrist_reshaped = wrist_mlp_out.permute(0, 2, 1)  # (B, width, 2)


        finger_flat = finger_input.reshape(B, T_in, -1)  # (B, 2, 10*768=7680)
        finger_mlp_out = self.finger_mlp(finger_flat)    # (B, 2, width=512)
        finger_reshaped = finger_mlp_out.permute(0, 2, 1)  # (B, width, 2)


        wrist_upsampled = self.tcn(wrist_reshaped)     # (B, width, 16)
        finger_upsampled = self.tcn(finger_reshaped)   # (B, width, 16)


        wrist_output = self.wrist_head(wrist_upsampled)  # (B, 18, 16)
        finger_output = self.finger_head(finger_upsampled)  # (B, 30, 16)

        wrist_output = wrist_output.permute(0, 2, 1)
        finger_output = finger_output.permute(0, 2, 1)
        B, T, C = wrist_output.shape
        wrist_output = wrist_output.reshape(B, T, 2, 9)
        finger_output = finger_output.reshape(B, T, 10, 3)

        return wrist_output, finger_output