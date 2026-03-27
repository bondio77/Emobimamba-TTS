# EGCA Speech Synthesis

Emotional speech synthesis based on FastSpeech2 with **Emotion-Guided Context Attention (EGCA)** and **Mamba** architecture. Supports emotion intensity control (min / med / max) for expressive TTS.

## Architecture

- **Backbone**: FastSpeech2 with Conformer encoder/decoder
- **Context Module**: Mamba-based sequence modeling
- **Emotion Control**: Intensity-aware emotion embedding (5 emotions x 3 intensity levels)
- **Discriminator**: JCU (Joint Conditional/Unconditional) adversarial training (optional)
- **Vocoder**: Parallel WaveGAN

### Supported Emotions
| ID | Emotion |
|----|---------|
| 0 | Neutral |
| 1 | Angry |
| 2 | Sad |
| 3 | Happy |
| 4 | Surprise |

## Project Structure

```
EGCA-sepech-synthesis/
├── train.py                  # Training (base model)
├── train_jcu.py              # Training (with JCU discriminator)
├── infer.py                  # Inference script
├── configs/esd/              # Model & preprocessing configs
├── vocoder/                  # Pre-trained Parallel WaveGAN vocoder
├── nets/                     # Network building blocks (Conformer, Transformer, Mamba, ...)
├── tts/                      # TTS models (FastSpeech2, GST, Meta-Style, Equalizer, ...)
├── inference/                # Inference modules (Text2Speech)
├── data/                     # Dataset loader
├── text/                     # Text frontend (G2P, cleaners, symbols)
├── audio/                    # Audio processing utilities
├── feats_extract/            # Feature extraction (mel, pitch, energy)
├── utils/                    # Utilities & torch helpers
├── training/                 # Training utilities (scheduler, optimizer, reporter)
└── requirements.txt
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# Mamba (requires CUDA)
pip install mamba-ssm causal-conv1d
```

### 2. Prepare Dataset (ESD)

Download the [Emotional Speech Database (ESD)](https://github.com/HLTSingapore/Emotional-Speech-Data).

### 3. Montreal Forced Aligner (MFA)

MFA를 사용하여 음성-텍스트 alignment를 추출합니다. 이 alignment로부터 phoneme-level pitch, energy, duration GT를 얻습니다.

```bash
# Install MFA
conda install -c conda-forge montreal-forced-aligner

# Download English acoustic model & dictionary
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

# Run alignment
mfa align /path/to/ESD/wavs english_us_arpa english_us_arpa /path/to/output/TextGrid
```

### 4. Preprocessing

MFA에서 얻은 TextGrid alignment를 기반으로 pitch, energy, duration, mel-spectrogram을 추출합니다.

`configs/esd/preprocess.yaml`에서 경로를 수정한 후:

```yaml
path:
  corpus_path: "/path/to/ESD"
  raw_path: "/path/to/ESD/wavs"
  preprocessed_path: "/path/to/output/preprocessed/"
  alignment_path: "/path/to/output/TextGrid/"
  stats_path: "/path/to/output/preprocessed/stats.json"
  symbol_path: "/path/to/output/preprocessed/symbols.json"
```

Preprocessor를 실행하면 다음 feature들이 생성됩니다:

| Feature | Directory | Description |
|---------|-----------|-------------|
| `mel/` | `{spk}-{emo}-mel-{id}.npy` | 80-dim log-mel spectrogram |
| `pitch_per_phoneme/` | `{spk}-{emo}-pitch_per_phoneme-{id}.npy` | Phoneme-level average F0 |
| `pitch_per_frame/` | `{spk}-{emo}-pitch_per_frame-{id}.npy` | Frame-level F0 (DIO) |
| `pitch_max_per_phoneme/` | `{spk}-{emo}-pitch_max_per_phoneme-{id}.npy` | Phoneme-level max F0 |
| `energy/` | `{spk}-{emo}-energy-{id}.npy` | Phoneme-level energy |
| `energy_frame/` | `{spk}-{emo}-energy_frame-{id}.npy` | Frame-level energy |
| `duration/` | `{spk}-{emo}-duration-{id}.npy` | Phoneme duration (from MFA) |

또한 `speakers.json`, `symbols.json`, `stats.json`, `train.txt`, `val.txt` 파일이 필요합니다.

## Training

### Base Model (EGCA)

```bash
python train.py \
    --preprocess_config ./configs/esd/preprocess.yaml \
    --model_config ./configs/esd/model_6L.yaml \
    --option_config ./configs/esd/option.yaml \
    --result_path ./model_checkpoint/egca/
```

### With JCU Discriminator

```bash
python train_jcu.py \
    --preprocess_config ./configs/esd/preprocess.yaml \
    --model_config ./configs/esd/model_8L.yaml \
    --option_config ./configs/esd/option.yaml \
    --result_path ./model_checkpoint/egca_jcu/
```

### Config Options

| Config | File | Description |
|--------|------|-------------|
| `model.yaml` | 4L encoder/decoder | Base model |
| `model_6L.yaml` | 6L encoder/decoder | Recommended |
| `model_8L.yaml` | 8L encoder/decoder | Larger model |
| `model_equal.yaml` | With style equalizer | Equalizer variant |

## Inference

```bash
python infer.py \
    -p ./configs/esd/preprocess.yaml \
    -m ./configs/esd/model_6L.yaml
```

`infer.py` 안에서 모델 checkpoint 경로, vocoder 경로, 출력 디렉토리를 수정하세요.

Intensity level을 `min`, `med`, `max`로 설정하여 감정 강도를 조절할 수 있습니다.

## Preprocessing Config

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sampling rate | 24000 Hz | Audio sampling rate |
| FFT size | 2048 | STFT filter length |
| Hop length | 300 | STFT hop size |
| Win length | 1200 | STFT window size |
| Mel channels | 80 | Number of mel bins |
| Mel fmin/fmax | 80 / 7600 Hz | Mel frequency range |
| F0 max | 400 Hz | Maximum pitch |
| Pitch/Energy | phoneme_level | Feature granularity |
