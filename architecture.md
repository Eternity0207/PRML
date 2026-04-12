# Architecture (Technical Details)

This file contains the technical structure of the project, including model layers, pipeline flow, and module responsibilities.

## 0. Documentation Mirror (README Preservation)

To preserve key operational README content in a second file, this section mirrors setup requirements and execution steps.

### Package Requirements (mirrored)

- kagglehub>=0.3.7
- librosa>=0.10.2
- soundfile>=0.12.1
- pesq>=0.0.4
- matplotlib>=3.8.0
- numpy>=1.26.0
- torch>=2.2.0
- ffmpeg
- imageio-ffmpeg

System note:

- A working `ffmpeg` executable is needed for video creation in `scripts/denoise_with_pca_weights.py`.
- On macOS, use `brew install ffmpeg` if the executable is not found.

### Run Instructions (mirrored)

1. Setup environment
- `chmod +x setup.sh`
- `./setup.sh`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

2. Run PCA pipeline
- `python scripts/run_pca.py --num-samples 5 --mix-snr-db 0 --output-dir outputs/pca`

3. Run ResUNet pipeline
- `python scripts/run_resunet.py --num-samples 10 --epochs 8 --output-dir outputs/resunet`

4. Run PCA-weight inference script
- `python scripts/denoise_with_pca_weights.py`

5. Run testing command (signal/noise subfolders, no video)
- `python scripts/denoise_with_pca_weights.py --input-dir input --signal-subdir signal --noise-subdir noise --mix-snr-db 0 --output-dir output --skip-video`

6. Generate slides (optional)
- `python ppt_generator/generate_pca_presentation.py`

## 1. Project Layering

The project is organized in five logical layers:

1. Entry Layer
- scripts/run_pca.py
- scripts/run_resunet.py

2. Pipeline Layer
- src/prml_denoise/pipelines/pca_pipeline.py
- src/prml_denoise/pipelines/resunet_pipeline.py

3. Model Layer
- src/prml_denoise/models/pca_model.py
- src/prml_denoise/models/resunet.py

4. Signal/Data Layer
- src/prml_denoise/data_io.py
- src/prml_denoise/dsp.py

5. Config Layer
- src/prml_denoise/config.py

## 2. PCA Pipeline Architecture

Flow:
1. Download datasets
2. Load and normalize audio
3. Mix speech and noise at configured SNR
4. Convert waveform to overlapping frame matrix (column-major)
5. Fit PCA basis on noisy frame matrix
6. Project frames onto speech subspace
7. Reconstruct waveform via overlap-add
8. Compute SNR and PESQ
9. Save waveforms, metrics CSV, and scree plot

### PCA Model Internals

The PCA model class stores:
- mean vector
- selected subspace basis V_s
- eigenvalue spectrum
- selected component count k

Primary operations:
- fit(X): covariance, eigendecomposition, variance-threshold component selection
- denoise(X): projection and reconstruction in selected subspace

## 3. ResUNet Architecture

Implemented as a 1D encoder-decoder with residual blocks.

### Building Blocks

1. ResidualBlock
- Conv1d -> BN -> ReLU -> Conv1d -> BN
- Residual skip: output = input + block(input)

2. EncoderBlock
- Feature extraction conv stack
- Residual refinement
- Strided Conv1d downsampling
- Returns: downsampled tensor + skip tensor

3. DecoderBlock
- ConvTranspose1d upsampling
- Concatenate with corresponding skip features
- Conv + residual refinement
- Handles shape mismatch with linear interpolation

4. Output head
- 1x1 Conv predicts residual
- Final output: clamp(noisy + residual)

### ResUNet Loss

DenoiseLoss combines:
- waveform L1 loss
- STFT magnitude L1 loss with weight 0.5

Total:

L = L1_wave + 0.5 * L1_spectral

## 4. Data and DSP Utilities

src/prml_denoise/data_io.py
- dataset download via kagglehub
- recursive audio file discovery
- load/resample/mono/normalize utilities

src/prml_denoise/dsp.py
- SNR-controlled mixing
- frame extraction (column and row layouts)
- overlap-add reconstruction (column and row layouts)
- SNR and PESQ scoring

## 5. End-to-End Runtime

PCA runtime:
- script -> pca_pipeline.run -> data_io + dsp -> PcaSpeechDenoiser -> outputs

ResUNet runtime:
- script -> resunet_pipeline.run -> data_io + dsp -> ResUNet + training loop -> outputs

## 6. Output Contracts

PCA output directory contains:
- representative_case waveforms
- metrics_summary.csv
- representative spectrogram plot
- scree plot

ResUNet output directory contains:
- best_resunet.pt
- metrics.csv
- per-case clean/noisy/denoised waveforms

## 7. Extension Path

To add a new denoiser:
1. Add model in src/prml_denoise/models
2. Add pipeline in src/prml_denoise/pipelines
3. Add script wrapper in scripts
4. Reuse existing data_io and dsp utilities for consistency
