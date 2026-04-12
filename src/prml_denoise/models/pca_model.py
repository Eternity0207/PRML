from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")


class PcaSpeechDenoiser:
    def __init__(self, variance_thresh: float = 0.95, device: torch.device | None = None):
        self.variance_thresh = variance_thresh
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean_: torch.Tensor | None = None
        self.v_s: torch.Tensor | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.k_: int | None = None

    def fit(self, x_np: np.ndarray) -> "PcaSpeechDenoiser":
        x = torch.from_numpy(x_np).to(self.device).to(torch.float64)
        _, n = x.shape
        self.mean_ = x.mean(dim=1, keepdim=True)
        x_c = x - self.mean_
        c = (x_c @ x_c.T) / n
        eigenvalues, eigenvectors = torch.linalg.eigh(c)
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.eigenvalues_ = eigenvalues.detach().cpu().numpy()

        total = eigenvalues.sum() + 1e-12
        cumulative = torch.cumsum(eigenvalues, dim=0) / total
        k_idx = torch.where(cumulative >= self.variance_thresh)[0]
        self.k_ = int(k_idx[0].item() + 1) if len(k_idx) > 0 else int(x.shape[0])
        self.v_s = eigenvectors[:, : self.k_].to(torch.float32)
        return self

    def denoise(self, x_np: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.v_s is None:
            raise RuntimeError("Call fit before denoise")
        x = torch.from_numpy(x_np).to(self.device).to(torch.float32)
        x_c = x - self.mean_.to(torch.float32)
        x_proj = self.v_s @ (self.v_s.T @ x_c)
        x_hat = x_proj + self.mean_.to(torch.float32)
        return x_hat.detach().cpu().numpy()

    def save_scree_plot(self, save_path: Path, max_components: int = 64) -> None:
        if self.eigenvalues_ is None:
            raise RuntimeError("Call fit before scree plot")
        n = min(max_components, len(self.eigenvalues_))
        total = float(np.sum(self.eigenvalues_)) + 1e-12
        indiv = self.eigenvalues_[:n] / total * 100.0
        cumul = np.cumsum(self.eigenvalues_[:n]) / total * 100.0
        x_axis = np.arange(1, n + 1)

        fig, ax1 = plt.subplots(figsize=(9, 4))
        ax2 = ax1.twinx()
        ax1.bar(x_axis, indiv, color="#1f77b4", alpha=0.65)
        ax2.plot(x_axis, cumul, "o-", color="#d62728", ms=3)
        ax2.axhline(self.variance_thresh * 100.0, ls="--", color="#2ca02c")
        ax1.set_xlabel("Principal Component Index")
        ax1.set_ylabel("Individual Variance (%)")
        ax2.set_ylabel("Cumulative Variance (%)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=160)
        plt.close()
