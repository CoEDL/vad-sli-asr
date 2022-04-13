from argparse import ArgumentParser
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm

import pickle
import pympi.Elan as Elan
import os
import torch
import torchaudio

parser = ArgumentParser(
    prog='run_sli-by-sblr',
    description='Spoken language identification (SLI) using SpeechBrain embeddings as input to a logistic regression classifier.',
)

parser.add_argument('logreg_pkl', help = "pickle file containing a trained logistic regression classifier")
parser.add_argument('wav_file',  help = "wav file to perform SLI on")

parser.add_argument('--vad_tier',  default="_vad", help = "Tier containing speech regions to classify")
parser.add_argument('--sli_tier',  default="_sli", help = "Tier to write classified speech regions")

parser.add_argument('--overwrite', help = "overwrite _vad tier on existing .eaf file?", dest='overwrite', action='store_true')
parser.add_argument('--rm_vad_tier', help = "remove _vad tier after finishing SLI task", dest='rm_vad_tier', action='store_true')

parser.add_argument('--cache_dir',  default="tmp/cache", help = "Directory for downloading pre-trained models")

parser.set_defaults(overwrite=False, rm_vad_tier=False)

args = parser.parse_args()

assert os.path.exists(args.logreg_pkl), f"Pickle file does not exist at: {args.logreg_pkl}"
assert os.path.exists(args.wav_file), f"Specified wav file does not exist: {args.wav_file}"

eaf_path   = os.path.splitext(args.wav_file)[0] + ".eaf"
eaf_exists = os.path.exists(eaf_path)

assert eaf_exists is True, f"Expected eaf file does not exist at: {eaf_path}"

eaf_data   = Elan.Eaf(file_path=eaf_path if eaf_exists else None)
eaf_tiers  = eaf_data.tiers.keys()

assert args.vad_tier in eaf_tiers, f"VAD tier '{args.vad_tier}' does not exist in {eaf_path}"

if args.sli_tier not in eaf_data.tiers.keys():
    # Add _sli tier if it doesn't already exist
    eaf_data.add_tier(args.sli_tier)

else:
    # If _sli tier already exists, check if we should overwrite
        # If overwrite, clear the tier first
        # If not, exit script
    if args.overwrite is True:
        eaf_data.remove_all_annotations_from_tier(args.sli_tier, clean=True)

    else:
        print(f"Skipping SLI on {eaf_path}, _sli tier already exists and overwite is set to False (use --overwrite to set to True)")
        exit()

# All conditions met to actually do SLI

waveform, sample_rate = torchaudio.load(args.wav_file)

if sample_rate != 16_000:
    print("Resampling audio to 16 kHz ...")
    samp_to_16k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16_000)
    waveform    = samp_to_16k(waveform)

sb_embd = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir=args.cache_dir)
sli_clf = pickle.load(open(args.logreg_pkl, 'rb'))

vad_regions = eaf_data.get_annotation_data_for_tier(args.vad_tier)

print("Starting spoken language identification ...")

for start_ms, end_ms, _ in tqdm(vad_regions):

    start_sample = max(0, start_ms - 250) * 16
    end_sample   = end_ms   * 16

    clip = waveform[:, start_sample:end_sample]
    emb  = sb_embd.encode_batch(clip).reshape((1, 256))
    lang = sli_clf.predict(emb)[0]
    
    eaf_data.add_annotation(args.sli_tier, start=start_ms, end=end_ms, value=lang)

if args.rm_vad_tier is True:
    eaf_data.remove_tier(args.vad_tier)

eaf_data.to_file(eaf_path)
print(f"Identified languages written to {args.sli_tier} tier in {eaf_path}")
