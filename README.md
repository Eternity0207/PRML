# Speech Noise Reduction (PCA + ResUNet)

This README is run-focused: setup, training, testing, commands, and flags.
For full mathematics and deep technical explanation, see architecture.md.

## 1. Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Required packages:

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

- ffmpeg executable is required only for MP4 generation in testing when skip-video is not used.
- On macOS, install ffmpeg with brew install ffmpeg.

## 2. Environment Setup

```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Training Commands

### 3.1 Train PCA

```bash
python scripts/run_pca.py --num-samples 1000 --mix-snr-db 0.8 --output-dir outputs/pca
```

What this run produces:

- outputs/pca/representative_case/*.wav
- outputs/pca/scree_plot.png
- outputs/pca/representative_spectrogram.png
- pca_weights.npz (project root, used by inference/testing)

PCA training flags:

- --target-sr: audio sample rate used during loading/resampling
- --frame-size: frame length for PCA processing
- --hop-size: hop length for framing and overlap-add
- --variance-thresh: PCA explained variance threshold for component selection
- --mix-snr-db: target SNR when creating noisy mixtures
- --num-samples: number of speech-noise pairs to use
- --random-seed: random seed for sample selection
- --output-dir: output folder for PCA artifacts

### 3.2 Train ResUNet

```bash
python scripts/run_resunet.py --num-samples 1000 --epochs 50 --output-dir outputs/resunet
```

What this run produces:

- outputs/resunet/best_resunet.pt
- outputs/resunet/*_clean.wav
- outputs/resunet/*_noisy.wav
- outputs/resunet/*_denoised.wav

ResUNet training flags:

- --target-sr: audio sample rate used during loading/resampling
- --frame-size: frame length used by the model
- --hop-size: hop length used for frame extraction/reconstruction
- --mix-snr-db: target SNR for noisy-mixture generation
- --num-samples: number of speech-noise pairs to use
- --random-seed: random seed
- --batch-size: training batch size
- --epochs: number of training epochs
- --learning-rate: AdamW learning rate
- --weight-decay: AdamW weight decay
- --train-ratio: train/validation split ratio
- --output-dir: output folder for checkpoint and audio outputs

## 4. Testing Commands

Input folder convention:

```text
input/
  signal/
  noise/
```

The signal and noise folders must contain equal numbers of audio files.

### 4.1 Test PCA only (no video)

```bash
python scripts/denoise_with_pca_weights.py \
  --input-dir input \
  --signal-subdir signal \
  --noise-subdir noise \
  --mix-snr-db 0 \
  --output-dir output \
  --skip-video
```

### 4.2 Test and compare PCA vs ResUNet (no video)

```bash
python scripts/denoise_with_pca_weights.py \
  --input-dir input \
  --signal-subdir signal \
  --noise-subdir noise \
  --mix-snr-db 0 \
  --output-dir output \
  --compare-resunet \
  --resunet-weights-path outputs/resunet/best_resunet.pt \
  --skip-video
```

Testing flags:

- --input-dir: root folder containing signal/noise subfolders
- --signal-subdir: clean signal folder name under input-dir
- --noise-subdir: noise folder name under input-dir
- --output-dir: output folder for audio, CSV, and plots
- --weights-path: PCA weights path (default pca_weights.npz)
- --target-sr: sample rate used during test-time loading
- --hop-size: PCA overlap-add hop size
- --mix-snr-db: SNR used to create noisy test mixtures
- --num-samples: number of random pairs to evaluate (0 means all)
- --random-seed: seed used when sampling pairs
- --skip-video: disable MP4 generation
- --compare-resunet: enable PCA vs ResUNet comparison
- --resunet-weights-path: checkpoint path for ResUNet comparison
- --resunet-frame-size: frame size used for ResUNet inference path
- --resunet-hop-size: hop size used for ResUNet inference path

Testing outputs:

- output/metrics_summary.csv
- output/metrics_aggregate.csv
- output/plots/snr_per_case.png
- output/plots/pesq_per_case.png
- output/plots/metric_deltas_per_case.png
- output/case_XX_* audio files (and optional MP4 when skip-video is not used)

## 5. Technical Reference

For architecture, module layering, signal-processing details, PCA mathematics, metric definitions, and worked derivations, see architecture.md.
