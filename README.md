# VAD-SLI-ASR

Python scripts for a speech processing pipeline with Voice Activity Detection (VAD), Spoken Language Identification (SLI), and Automatic Speech Recognition (ASR). Our use case involves using VAD to detect time regions in a language documentation recording where someone is speaking, then using SLI to classify each region as either English (eng) or Muruwari (zmu), and then using an English ASR model to transcribe regions detected as English. This pipeline outputs an ELAN .eaf file with the following tier structure (`_vad`, `_sli`, and `_asr`):

<p align="center">
![](docs/elan-eg.png)
</p>

## Set up

```
pip install -r requirements.txt
```

### Data

```
├── data
│   ├── sli-train      <- Training data for SLI (one folder per language)
│   │   ├── eng/       <- .wav files (English utterances)
│   │   ├── zmu/       <- .wav files (Muruwari utterances)
│   ├── asr-train      <- Intermediate data that has been transformed.
│   │   ├── eng.tsv    <- transcriptions
│   │   ├── eng/       <- .wav files (English utterances)
```

## Usage

### VAD

```bash
# VAD
python scripts/run_vad-by-silero.py myrecording.wav
```

### SLI

```bash
# To train a classifier using your own clips and then save it:
python scripts/train_sli-by-sblr.py data/sli-train models/zmu-eng_sli_k10.pkl

# Use trained model to classify VAD-detected regions as eng or zmu
python scripts/run_sli-by-sblr.py models/zmu-eng_sli_k10.pkl myrecording.wav
```

### ASR

```bash
# To fine-tune a wav2vec 2.0 model and save the checkpoint:
python scripts/train_asr-by-w2v2.py data/asr-train data/checkpoints/no-lm_b10

# Transcribe using trained model 
python scripts/run_asr-by-w2v2.py data/checkpoints/no-lm_b10 myrecording.wav

```
