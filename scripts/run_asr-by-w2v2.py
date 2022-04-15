from argparse import ArgumentParser
from datasets import Dataset
from helpers.asr import configure_w2v2_for_inference
from transformers import logging

import pandas as pd
import pympi.Elan as Elan
import os
import re
import torchaudio

parser = ArgumentParser(
    prog='run_asr-by-w2v2',
    description='Run automatic speech recognition (ASR) using wav2vec 2.0',
)
parser.add_argument('repo_path_or_name', help = "Pre-trained wav2vec 2.0 model, local path or HuggingFace repo name")
parser.add_argument('wav_file',  help = "wav file to perform ASR on")

parser.add_argument('--roi_tier', default="_sli", help = "Tier containing speech regions of interest to transcribe")
parser.add_argument('--roi_filter', default="eng", help = "Regular expression to filter regions of interest")

parser.add_argument('--asr_tier',  default="_asr", help = "Tier to write transcriptions to")
parser.add_argument('--overwrite', help = "overwrite _asr tier on existing .eaf file?", dest='overwrite', action='store_true')

parser.add_argument('--cache_dir',  default="tmp/cache", help = "Directory for downloading pre-trained models")

parser.set_defaults(overwrite=False)

args = parser.parse_args()

logging.set_verbosity(40)

assert os.path.exists(args.wav_file), f"Specified wav file does not exist: {args.wav_file}"

eaf_path   = os.path.splitext(args.wav_file)[0] + ".eaf"
eaf_exists = os.path.exists(eaf_path)

assert eaf_exists is True, f"Expected eaf file does not exist at: {eaf_path}"

eaf_data   = Elan.Eaf(file_path=eaf_path if eaf_exists else None)
eaf_tiers  = eaf_data.tiers.keys()

if args.asr_tier not in eaf_data.tiers.keys():
    # Add _asr tier if it doesn't already exist
    eaf_data.add_tier(args.asr_tier)

else:
    # If _asr tier already exists, check if we should overwrite
        # If overwrite, clear the tier first
        # If not, exit script
    if args.overwrite is True:
        eaf_data.remove_all_annotations_from_tier(args.asr_tier, clean=True)

    else:
        print(f"Skipping ASR on {eaf_path}, _asr tier already exists and overwite is set to False (use --overwrite to set to True)")
        exit()

# All conditions met to actually do ASR
waveform, sample_rate = torchaudio.load(args.wav_file)

if sample_rate != 16_000:
    print("Resampling audio to 16 kHz ...")
    samp_to_16k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16_000)
    waveform    = samp_to_16k(waveform)

def get_speech_excerpts(batch):
    start_sample = batch["start_ms"] * 16
    end_sample   = batch["end_ms"] * 16

    batch["speech"] = waveform[:, start_sample:end_sample].squeeze().numpy()

    return batch

_, _, transcribe_speech = configure_w2v2_for_inference(args.repo_path_or_name, cache_dir=args.cache_dir)

roi_annots  = eaf_data.get_annotation_data_for_tier(args.roi_tier)
roi_annots  = [ r for r in roi_annots if bool(re.search(args.roi_filter, r[2])) ]

roi_dataset = pd.DataFrame(roi_annots, columns=['start_ms', 'end_ms', 'annotation'])
roi_dataset = Dataset.from_pandas(roi_dataset).remove_columns(['annotation'])

print("Gathering speech regions into a dataset ...")

roi_dataset = roi_dataset.map(get_speech_excerpts)

print("Running ASR on each region of interest ...")

roi_dataset = roi_dataset.map(transcribe_speech)

for index, region in enumerate(roi_annots):

    start_ms, end_ms, annot = region

    annot = roi_dataset['transcription'][index] if annot == 'eng' else ''

    eaf_data.add_annotation(args.asr_tier, start=start_ms, end=end_ms, value=annot)

eaf_data.to_file(eaf_path)
print(f"Transcribed regions written to {args.asr_tier} tier in {eaf_path}")
