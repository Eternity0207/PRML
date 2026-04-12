# SPeech Noise Reduction Using PCA

## Abstract

This project studies speech denoising with PCA as the primary method and ResUNet as a comparison method.
The objective is to recover clean and intelligible speech from noisy recordings while preserving natural voice characteristics.

## Package Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Current required packages:

- kagglehub>=0.3.7
- librosa>=0.10.2
- soundfile>=0.12.1
- pesq>=0.0.4
- matplotlib>=3.8.0
- numpy>=1.26.0
- torch>=2.2.0
- ffmpeg
- imageio-ffmpeg

Notes:

- For video rendering in `scripts/denoise_with_pca_weights.py`, a working `ffmpeg` executable is required.
- On macOS, install system ffmpeg with `brew install ffmpeg` if needed.

## Setup and Run Instructions

### 1. Environment setup

```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run PCA pipeline

```bash
python scripts/run_pca.py --num-samples 5 --mix-snr-db 0 --output-dir outputs/pca
```

### 3. Run ResUNet baseline

```bash
python scripts/run_resunet.py --num-samples 10 --epochs 8 --output-dir outputs/resunet
```

### 4. Denoise with saved PCA weights (inference)

```bash
python scripts/denoise_with_pca_weights.py
```

### 5. Test command (signal/noise subfolders, no video)

```bash
python scripts/denoise_with_pca_weights.py \
	--input-dir input \
	--signal-subdir signal \
	--noise-subdir noise \
	--mix-snr-db 0 \
	--output-dir output \
	--skip-video
```

### 6. Generate project presentation (optional)

```bash
python ppt_generator/generate_pca_presentation.py
```

## 1. Problem Statement

Real-world speech is often corrupted by environmental noise.
Given a noisy signal, the task is to estimate the underlying clean speech signal.

At sample level:

$$
x[t] = s[t] + v[t]
$$

where:
- $x[t]$ is noisy speech,
- $s[t]$ is clean speech,
- $v[t]$ is additive noise.

## 2. Project Objectives

1. Build controlled noisy speech using known SNR.
2. Apply PCA-based denoising on framed signals.
3. Reconstruct time-domain speech from denoised frames.
4. Compare quality against a ResUNet baseline.
5. Measure improvements with SNR and PESQ.

## 3. End-to-End Methodology

1. Load speech and noise audio.
2. Convert to mono and common sample rate.
3. Mix at target SNR.
4. Split waveform into overlapping short-time frames.
5. Denoise frames:
- PCA pipeline: linear subspace projection.
- ResUNet pipeline: learned nonlinear mapping.
6. Reconstruct final waveform using overlap-add.
7. Evaluate and compare quality.

## 4. Signal Analysis Before PCA

### 4.1 Resampling and normalization

All files are brought to one sampling rate (default 16 kHz) and peak-normalized.
This ensures consistent statistics across files.

### 4.2 SNR-controlled mixing

Define speech and noise average powers over $T$ samples:

$$
P_s = \frac{1}{T}\sum_{t=1}^{T} s[t]^2,
\qquad
P_n = \frac{1}{T}\sum_{t=1}^{T} n[t]^2
$$

Target SNR relation:

$$
\mathrm{SNR}_{\mathrm{dB}} = 10\log_{10}\left(\frac{P_s}{P_n}\right)
$$

To enforce a desired SNR, noise is scaled by:

$$
\alpha = \sqrt{\frac{P_s}{P_n\,10^{\mathrm{SNR}_{\mathrm{dB}}/10}}}
$$

Then noisy mixture is:

$$
x[t] = s[t] + \alpha n[t]
$$

### 4.3 Framing and overlap

Speech is non-stationary over long durations, but can be approximated as locally stationary in short windows.
So we process short overlapping frames.

For frame length $D$ and hop $H$, frame $i$ is:

$$
x_i = [x[iH], x[iH+1], \ldots, x[iH + D - 1]]^T \in \mathbb{R}^{D}
$$

Collect $N$ frames as columns:

$$
X = [x_1, x_2, \ldots, x_N] \in \mathbb{R}^{D \times N}
$$

## 5. PCA Mathematics in Detail

### 5.1 Mean centering

Column mean vector:

$$
\mu = \frac{1}{N}\sum_{i=1}^{N} x_i
$$

Centered frame matrix:

$$
\tilde{X} = X - \mu\mathbf{1}^T
$$

where $\mathbf{1}$ is an all-ones vector of length $N$.

### 5.2 Covariance matrix

$$
C = \frac{1}{N}\tilde{X}\tilde{X}^T \in \mathbb{R}^{D\times D}
$$

Interpretation:
- diagonal entries: variance at each frame position,
- off-diagonal entries: correlation between frame positions.

### 5.3 Eigen decomposition

Solve:

$$
Cv_i = \lambda_i v_i,
\qquad
\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_D \ge 0
$$

where:
- $v_i$ is principal direction,
- $\lambda_i$ is variance captured by that direction.

### 5.4 Explained variance and choosing $k$

Explained variance ratio of first $k$ components:

$$
r_k = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{j=1}^{D} \lambda_j}
$$

Choose smallest $k$ such that:

$$
r_k \ge \tau
$$

with default threshold $\tau = 0.95$.

### 5.5 Projection and denoised reconstruction

Define principal basis matrix:

$$
V_s = [v_1, v_2, \ldots, v_k] \in \mathbb{R}^{D\times k}
$$

Project to PCA subspace:

$$
Z = V_s^T\tilde{X}
$$

Reconstruct centered frames:

$$
\tilde{X}_{\mathrm{rec}} = V_s Z = V_sV_s^T\tilde{X}
$$

Add mean back:

$$
\hat{X} = \tilde{X}_{\mathrm{rec}} + \mu\mathbf{1}^T
$$

Each column of $\hat{X}$ is a denoised frame.

### 5.6 Why PCA denoising works

PCA keeps dominant shared structure and removes weak components.
For speech frames, dominant components often preserve speech shape, while low-energy components tend to contain more random/noise variation.

Equivalent optimality statement:

$$
\hat{X}_k = \arg\min_{\mathrm{rank}(Y)\le k}\|\tilde{X} - Y\|_F^2
$$

So PCA gives the best rank-$k$ linear approximation under Frobenius norm.

## 6. Overlap-Add Reconstruction Mathematics

After denoising frame sequence $\{\hat{x}_i\}$:

$$
y[t] = \sum_i \hat{x}_i[t-iH] \cdot \mathbf{1}_{0\le t-iH < D}
$$

$$
c[t] = \sum_i \mathbf{1}_{0\le t-iH < D}
$$

$$
\hat{s}[t] = \frac{y[t]}{\max(c[t],1)}
$$

This averaging avoids amplitude inflation in overlapping regions.

## 7. Fourier / STFT: Where It Is Used and Not Used

1. PCA denoising step is performed in time-domain frame space.
2. PCA projection is not performed on Fourier bins.
3. STFT is used for spectrogram visualization/analysis.
4. In the ResUNet path, STFT magnitude is used in the training loss to improve spectral fidelity.

So Fourier analysis is used for interpretation and auxiliary training objectives, not for the core PCA projection itself.

## 8. Parameter-by-Parameter Analysis

### target_sr
Controls time resolution and high-frequency content preservation.
16 kHz is a standard speech-focused setting.

### frame_size ($D$)
Larger $D$ captures broader context but increases covariance dimension.
Smaller $D$ gives better locality but less stable statistics.

### hop_size ($H$)
Smaller $H$ increases overlap and smoothness of reconstruction, but increases compute.

### variance_thresh ($\tau$)
Controls retained PCA dimension:
- lower $\tau$: stronger suppression, higher speech-loss risk,
- higher $\tau$: more speech preserved, possibly more residual noise.

### mix_snr_db
Controls difficulty level of denoising examples.
Lower SNR means harder denoising conditions.

### num_samples
More sample pairs improve covariance estimation and PCA basis stability.

## 9. Evaluation Metrics

### SNR

For reference clean signal $s[t]$ and estimate $\hat{s}[t]$:

$$
\mathrm{SNR}(s,\hat{s}) = 10\log_{10}\left(\frac{\sum_t s[t]^2}{\sum_t (s[t]-\hat{s}[t])^2 + \varepsilon}\right)
$$

Higher is better.

### PESQ

PESQ estimates perceptual speech quality (how listeners perceive clarity).
Higher is better.

## 10. PCA vs ResUNet (Project-Level Comparison)

| Criterion | PCA | ResUNet |
|---|---|---|
| Core principle | Linear subspace model | Nonlinear learned mapping |
| Training need | Minimal | Significant |
| Compute cost | Lower | Higher |
| Interpretability | High | Medium |
| Hard-noise handling | Good baseline | Usually stronger |

Observed practical trend:
1. Both methods improve noisy speech.
2. PCA gives reliable baseline gains quickly.
3. ResUNet can achieve larger gains in difficult conditions when sufficiently trained.

## 11. Conclusion

PCA is a mathematically grounded, efficient, and interpretable denoising baseline.
It is especially useful when you need strong explainability and low computational cost.
ResUNet provides a stronger nonlinear alternative for higher-capacity denoising.
Together, both methods provide a complete baseline-to-advanced framework for speech noise reduction studies.

## 12. Worked Numerical Example (Mini PCA Denoising)

This toy example shows the same math with tiny numbers.

Assume 2-sample frames and 4 frames:

$$
X = \begin{bmatrix}
2 & 3 & 4 & 5 \\
1 & 2 & 1 & 2
\end{bmatrix}
$$

### Step 1: Mean vector

$$
\mu = \frac{1}{4}\sum_{i=1}^{4}x_i
= \begin{bmatrix}3.5 \\ 1.5\end{bmatrix}
$$

### Step 2: Centered matrix

$$
	ilde{X} = X - \mu\mathbf{1}^T
= \begin{bmatrix}
-1.5 & -0.5 & 0.5 & 1.5 \\
-0.5 & 0.5 & -0.5 & 0.5
\end{bmatrix}
$$

### Step 3: Covariance

$$
C = \frac{1}{4}\tilde{X}\tilde{X}^T
= \begin{bmatrix}
1.25 & 0.25 \\
0.25 & 0.25
\end{bmatrix}
$$

### Step 4: Eigenvalues (approx)

For this matrix, eigenvalues are approximately:

$$
\lambda_1 \approx 1.309,
\qquad
\lambda_2 \approx 0.191
$$

Total variance is $1.5$.
Explained variance of first component:

$$
\frac{\lambda_1}{\lambda_1+\lambda_2} \approx \frac{1.309}{1.5} \approx 0.873
$$

So one component explains about $87.3\%$ variance.

### Step 5: Keep top-1 component

If we choose $k=1$, we project on the first principal direction and reconstruct:

$$
\hat{X} = v_1v_1^T\tilde{X} + \mu\mathbf{1}^T
$$

This gives a rank-1 approximation of frame data.
Effect:
1. dominant shared pattern is preserved,
2. weaker variation is suppressed,
3. denoising occurs by dimensionality reduction.

This is exactly what the full pipeline does, but on larger speech frame matrices.

## 13. Quick Run

```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
python scripts/run_pca.py --num-samples 5 --mix-snr-db 0 --output-dir outputs/pca
python scripts/run_resunet.py --num-samples 10 --epochs 8 --output-dir outputs/resunet
```

## 14. Technical Architecture

Technical module and layer details are documented in architecture.md.
