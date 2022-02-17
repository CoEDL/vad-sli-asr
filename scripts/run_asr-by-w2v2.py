from argparse import ArgumentParser
from datasets import Dataset
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm
from transformers import logging, AutoModelForCTC, AutoProcessor

import glob
import pandas as pd
import pympi.Elan as Elan
import os
import re
import torch
import torchaudio

parser = ArgumentParser(
    prog='run_asr-by-w2v2',
    description='Run automatic speech recognition (ASR) using wav2vec 2.0',
)

parser.add_argument('w2v2_cp_dir', help = "Directory containing model checkpoint")
parser.add_argument('wav_file',  help = "wav file to perform ASR on")

parser.add_argument('--roi_tier', default="_sli", help = "Tier containing speech regions of interest to transcribe")
parser.add_argument('--roi_filter', default="eng", help = "Regular expression to filter regions of interest")

parser.add_argument('--asr_tier',  default="_asr", help = "Tier to write transcriptions to")
parser.add_argument('--overwrite', help = "overwrite _asr tier on existing .eaf file?", dest='overwrite', action='store_true')

parser.add_argument('--cuda', help = "send model and data to GPU", dest='cuda', action='store_true')

parser.set_defaults(overwrite=False, cuda=False)

args = parser.parse_args()

logging.set_verbosity(40)

assert os.path.isdir(args.w2v2_cp_dir), f"Checkpoint directory does not exist at: {args.logreg_pkl}"
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

def map_to_array(batch):
    start_sample = max(0, batch["start_ms"] - 500) * 16
    end_sample   = batch["end_ms"] * 16

    batch["speech"] = waveform[:, start_sample:end_sample].squeeze().numpy()

    return batch

def make_map_to_pred(decode_with_lm=False):

    def map_to_pred(batch):
        input_values = processor(batch["speech"], return_tensors="pt", padding="longest", sampling_rate=16_000).input_values

        with torch.no_grad():
            logits = model(input_values.to("cuda")).logits if args.cuda else model(input_values).logits

        if decode_with_lm is False:
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)
        else:
            transcription = processor.batch_decode(logits.cpu().numpy()).text

        batch["transcription"] = transcription

        return batch
    
    return map_to_pred

waveform, sample_rate = torchaudio.load(args.wav_file)

if sample_rate != 16_000:
    print("Resampling audio to 16 kHz ...")
    samp_to_16k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16_000)
    waveform    = samp_to_16k(waveform)

model_path = glob.glob(os.path.join(args.w2v2_cp_dir, 'checkpoint-*'))[0]
model      = AutoModelForCTC.from_pretrained(model_path)
processor  = AutoProcessor.from_pretrained(args.w2v2_cp_dir)

has_lm_dir = os.path.isdir(os.path.join(args.w2v2_cp_dir, 'language_model'))
map_to_pred = make_map_to_pred(decode_with_lm=has_lm_dir)

if args.cuda:
    model.to("cuda")

roi_annots  = eaf_data.get_annotation_data_for_tier(args.roi_tier)
roi_annots  = [ r for r in roi_annots if bool(re.search(args.roi_filter, r[2])) ]

roi_dataset = pd.DataFrame(roi_annots, columns=['start_ms', 'end_ms', 'annotation'])
roi_dataset = Dataset.from_pandas(roi_dataset).remove_columns(['annotation'])

print("Gathering speech regions into a dataset ...")

roi_dataset = roi_dataset.map(map_to_array)

print("Running ASR on each region of interest ...")

result = roi_dataset.map(map_to_pred, batched=True, batch_size=1, remove_columns=["speech"])

for r in result:
    eaf_data.add_annotation(args.asr_tier, start=r['start_ms'], end=r['end_ms'], value=r['transcription'])

eaf_data.to_file(eaf_path)
print(f"Transcribed regions written to {args.asr_tier} tier in {eaf_path}")