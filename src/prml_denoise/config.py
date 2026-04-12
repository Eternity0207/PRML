from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommonConfig:
    target_sr: int = 16_000
    frame_size: int = 512
    hop_size: int = 256
    mix_snr_db: float = 0.0
    random_seed: int = 42
    output_dir: Path = Path("output")


@dataclass
class PcaConfig(CommonConfig):
    variance_thresh: float = 0.95
    num_samples: int = 5


@dataclass
class ResUNetConfig(CommonConfig):
    num_samples: int = 10
    batch_size: int = 16
    epochs: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    train_ratio: float = 0.8
