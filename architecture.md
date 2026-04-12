# Architecture and Technical Details

This document contains the complete technical explanation of the project, including pipeline behavior, mathematical formulation, model architecture, and evaluation definitions.

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

1. Build controlled noisy speech examples using known SNR.
2. Apply PCA-based denoising on framed signals.
3. Reconstruct time-domain speech from denoised frames.
4. Compare quality against a ResUNet baseline.
5. Measure improvements using SNR and PESQ.

## 3. Project Layering

The repository is organized in five logical layers:

1. Entry layer
- scripts/run_pca.py
- scripts/run_resunet.py
- scripts/denoise_with_pca_weights.py

2. Pipeline layer
- src/prml_denoise/pipelines/pca_pipeline.py
- src/prml_denoise/pipelines/resunet_pipeline.py

3. Model layer
- src/prml_denoise/models/pca_model.py
- src/prml_denoise/models/resunet.py

4. Signal and data layer
- src/prml_denoise/data_io.py
- src/prml_denoise/dsp.py

5. Config layer
- src/prml_denoise/config.py

## 4. End-to-End Methodology

1. Load speech and noise audio.
2. Convert to mono and a common sample rate.
3. Mix at target SNR.
4. Split waveform into overlapping short-time frames.
5. Denoise frames:
- PCA pipeline: linear subspace projection.
- ResUNet pipeline: learned nonlinear mapping.
6. Reconstruct final waveform using overlap-add.
7. Evaluate and compare quality.

## 5. Signal Analysis Before PCA

### 5.1 Resampling and normalization

All files are brought to one sampling rate (default 16 kHz) and peak-normalized.
This ensures consistent statistics across files.

### 5.2 SNR-controlled mixing

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

### 5.3 Framing and overlap

Speech is non-stationary over long durations, but can be approximated as locally stationary in short windows.
So processing is done on short overlapping frames.

For frame length $D$ and hop $H$, frame $i$ is:

$$
x_i = [x[iH], x[iH+1], \ldots, x[iH + D - 1]]^T \in \mathbb{R}^{D}
$$

Collect $N$ frames as columns:

$$
X = [x_1, x_2, \ldots, x_N] \in \mathbb{R}^{D \times N}
$$

## 6. PCA Mathematics in Detail

### 6.1 Mean centering

Column mean vector:

$$
\mu = \frac{1}{N}\sum_{i=1}^{N} x_i
$$

Centered frame matrix:

$$
\tilde{X} = X - \mu\mathbf{1}^T
$$

where $\mathbf{1}$ is an all-ones vector of length $N$.

### 6.2 Covariance matrix

$$
C = \frac{1}{N}\tilde{X}\tilde{X}^T \in \mathbb{R}^{D\times D}
$$

Interpretation:

- diagonal entries: variance at each frame position,
- off-diagonal entries: correlation between frame positions.

### 6.3 Eigen decomposition

Solve:

$$
Cv_i = \lambda_i v_i,
\qquad
\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_D \ge 0
$$

where:

- $v_i$ is principal direction,
- $\lambda_i$ is variance captured by that direction.

### 6.4 Explained variance and choosing $k$

Explained variance ratio of first $k$ components:

$$
r_k = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{j=1}^{D} \lambda_j}
$$

Choose smallest $k$ such that:

$$
r_k \ge \tau
$$

with default threshold $\tau = 0.95$.

### 6.5 Projection and denoised reconstruction

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

### 6.6 Why PCA denoising works

PCA keeps dominant shared structure and removes weak components.
For speech frames, dominant components often preserve speech shape, while low-energy components tend to contain more random or weak noise variation.

Equivalent optimality statement:

$$
\hat{X}_k = \arg\min_{\mathrm{rank}(Y)\le k}\|\tilde{X} - Y\|_F^2
$$

So PCA gives the best rank-$k$ linear approximation under Frobenius norm.

### 6.7 Matrix-Operator View (Detailed Derivation)

Define the sample-centering operator over $N$ frames:

$$
H = I_N - \frac{1}{N}\mathbf{1}\mathbf{1}^T
$$

Then centered data can be written compactly as:

$$
	ilde{X} = XH
$$

and covariance is:

$$
C = \frac{1}{N}\tilde{X}\tilde{X}^T
= \frac{1}{N}XHX^T
$$

because $H$ is symmetric and idempotent ($H^T = H$, $H^2 = H$).

With eigendecomposition:

$$
C = V\Lambda V^T,
\quad
\Lambda = \mathrm{diag}(\lambda_1,\ldots,\lambda_D),
\quad
\lambda_1 \ge \cdots \ge \lambda_D
$$

choose top-$k$ basis $V_s = [v_1,\ldots,v_k]$ and define projection operator:

$$
P_k = V_sV_s^T
$$

Then PCA denoising is exactly:

$$
\hat{X} = P_k\tilde{X} + \mu\mathbf{1}^T
$$

So each frame is orthogonally projected onto a $k$-dimensional subspace and then shifted back by the mean.

### 6.8 Error Decomposition and Variance Retention

If singular values of $\tilde{X}$ are $\sigma_i$, then:

$$
\sigma_i^2 = N\lambda_i
$$

Best rank-$k$ reconstruction error is:

$$
\|\tilde{X} - \hat{X}_k\|_F^2
= \sum_{i=k+1}^{D}\sigma_i^2
= N\sum_{i=k+1}^{D}\lambda_i
$$

Hence retained variance ratio is:

$$
r_k = \frac{\sum_{i=1}^{k}\lambda_i}{\sum_{j=1}^{D}\lambda_j}
$$

and discarded variance fraction is $1-r_k$.

This gives a direct quantitative tradeoff between denoising strength and signal distortion.

### 6.9 Probabilistic Interpretation (Intuition)

PCA can be interpreted as fitting a low-dimensional linear latent subspace to dominant data structure.

- high-variance directions: dominant shared structure (often speech form)
- low-variance directions: weaker fluctuations (often noise-like variation)

Denoising via PCA assumes that useful speech structure concentrates more in top components than nuisance noise.

### 6.10 Assumptions and Failure Modes

Core assumptions behind PCA denoising:

1. speech frames are approximately low-rank in short-time windows,
2. noise is less coherent than speech across frame dimensions,
3. linear subspace model is sufficient for the noise regime.

Potential failure cases:

1. nonstationary or structured noise with high variance can occupy top components,
2. aggressive truncation (small $k$) can remove speech details,
3. speech and noise subspaces may overlap strongly at low SNR.

This is why PCA is a strong interpretable baseline, while nonlinear models (for example ResUNet) can outperform it in harder conditions.

## 7. Overlap-Add Reconstruction Mathematics

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

## 8. Fourier or STFT Usage

1. PCA denoising step is performed in time-domain frame space.
2. PCA projection is not performed on Fourier bins.
3. STFT is used for spectrogram visualization and analysis.
4. In ResUNet training, STFT magnitude is used in the loss for spectral fidelity.

So Fourier analysis is used for interpretation and auxiliary objectives, not for the core PCA projection.

## 9. ResUNet Architecture

Implemented as a 1D encoder-decoder with residual blocks.

### 9.1 Building blocks

1. ResidualBlock
- Conv1d -> BatchNorm1d -> ReLU -> Conv1d -> BatchNorm1d
- Residual skip: output = input + block(input)

2. EncoderBlock
- Feature extraction conv stack
- Residual refinement
- Strided Conv1d downsampling
- Returns downsampled tensor and skip tensor

3. DecoderBlock
- ConvTranspose1d upsampling
- Concatenate with skip features
- Conv stack with residual refinement
- Shape mismatch handling with interpolation

4. Output head
- 1x1 Conv predicts residual
- Final output: clamp(noisy + residual)

### 9.2 ResUNet loss

DenoiseLoss combines:

- waveform L1 loss,
- STFT magnitude L1 loss with weight 0.5.

Total loss:

$$
L = L_{\text{wave}} + 0.5\,L_{\text{spectral}}
$$

## 10. Parameter-by-Parameter Analysis

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

## 11. Evaluation Metrics

### 11.1 SNR

For reference clean signal $s[t]$ and estimate $\hat{s}[t]$:

$$
\mathrm{SNR}(s,\hat{s}) = 10\log_{10}\left(\frac{\sum_t s[t]^2}{\sum_t (s[t]-\hat{s}[t])^2 + \varepsilon}\right)
$$

Higher is better.

### 11.2 PESQ

PESQ estimates perceptual speech quality.
Higher is better.

## 12. Runtime and Output Contracts

### PCA runtime

script -> pca_pipeline.run -> data_io + dsp -> PcaSpeechDenoiser -> outputs

PCA outputs:

- outputs/pca/representative_case/*.wav
- outputs/pca/representative_spectrogram.png
- outputs/pca/scree_plot.png
- pca_weights.npz at repository root

### ResUNet runtime

script -> resunet_pipeline.run -> data_io + dsp -> ResUNet + training loop -> outputs

ResUNet outputs:

- outputs/resunet/best_resunet.pt
- outputs/resunet/*_clean.wav
- outputs/resunet/*_noisy.wav
- outputs/resunet/*_denoised.wav

### Testing runtime

script -> denoise_with_pca_weights.py using local signal and noise folders

Testing outputs:

- metrics_summary.csv
- metrics_aggregate.csv
- plots/snr_per_case.png
- plots/pesq_per_case.png
- plots/metric_deltas_per_case.png
- per-case output audio (and optional MP4)

## 13. Worked Numerical Example (Mini PCA)

Assume 2-sample frames and 4 frames:

$$
X = \begin{bmatrix}
2 & 3 & 4 & 5 \\
1 & 2 & 1 & 2
\end{bmatrix}
$$

Step 1 mean vector:

$$
\mu = \frac{1}{4}\sum_{i=1}^{4}x_i
= \begin{bmatrix}3.5 \\ 1.5\end{bmatrix}
$$

Step 2 centered matrix:

$$
X_c = X - \mu\mathbf{1}^T
= \begin{bmatrix}
-1.5 & -0.5 & 0.5 & 1.5 \\
-0.5 & 0.5 & -0.5 & 0.5
\end{bmatrix}
$$

Step 3 covariance:

$$
C = \frac{1}{4}X_cX_c^T
= \begin{bmatrix}
1.25 & 0.25 \\
0.25 & 0.25
\end{bmatrix}
$$

Step 4 eigenvalues (approx):

$$
\lambda_1 \approx 1.309,
\qquad
\lambda_2 \approx 0.191
$$

Total variance is 1.5.
Explained variance by the first component:

$$
\frac{\lambda_1}{\lambda_1+\lambda_2} \approx \frac{1.309}{1.5} \approx 0.873
$$

Step 5 keep top-1 component:

$$
\hat{X} = v_1v_1^TX_c + \mu\mathbf{1}^T
$$

This is a rank-1 approximation of frame data that preserves dominant structure and suppresses weaker variation.

## 14. PCA vs ResUNet (Project-Level Comparison)

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
