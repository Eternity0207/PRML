import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels),
        )
        self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.block(x)
        down = self.downsample(skip)
        return down, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            nn.Conv1d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.size(-1) != skip.size(-1):
            x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ResUNet(nn.Module):
    def __init__(self, in_channels: int = 1, base_f: int = 32):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, base_f)
        self.enc2 = EncoderBlock(base_f, base_f * 2)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_f * 2, base_f * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_f * 4),
            nn.ReLU(inplace=True),
            ResidualBlock(base_f * 4),
        )
        self.dec2 = DecoderBlock(base_f * 4, base_f * 2, base_f * 2)
        self.dec1 = DecoderBlock(base_f * 2, base_f, base_f)
        self.out_conv = nn.Conv1d(base_f, 1, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        e1, s1 = self.enc1(noisy)
        e2, s2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d2 = self.dec2(b, s2)
        d1 = self.dec1(d2, s1)
        residual = self.out_conv(d1)
        return torch.clamp(noisy + residual, -1.0, 1.0)


class DenoiseLoss(nn.Module):
    def __init__(self, n_fft: int = 256, hop_length: int = 128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.l1 = nn.L1Loss()
        self.register_buffer("_stft_window", torch.hann_window(self.n_fft), persistent=False)

    def _stft_mag(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)
        window = self._stft_window.to(device=x.device, dtype=x.dtype)
        s = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
        )
        return torch.abs(s)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wav_loss = self.l1(pred, target)
        spec_loss = self.l1(self._stft_mag(pred), self._stft_mag(target))
        total = wav_loss + 0.5 * spec_loss
        return total, wav_loss.detach(), spec_loss.detach()
