from argparse import ArgumentParser
import pympi.Elan as Elan
import os
import sys
import torch
import torchaudio

from helpers.eaf import get_eaf_file

parser = ArgumentParser(
    prog='run_vad-by-silero',
    description='Voice activity detection with Silero. Writes intervals onto _vad tier in sidecar file.',
)

parser.add_argument('wav_file',  help = "wav file to perform VAD on")

parser.add_argument('--eaf_file',  help = "write to sidecar file (e.g. myfile.eaf for myfile.wav) unless specified.")

parser.add_argument('--vad_tier',  default="_vad", help = "Tier containing speech regions to classify")
parser.add_argument('--overwrite', help = "overwrite _vad tier on existing .eaf file?", dest='overwrite', action='store_true')

parser.set_defaults(overwrite=False)

args = parser.parse_args()

assert os.path.exists(args.wav_file), f"Specified wav file does not exist: {args.wav_file}"

eaf_path, eaf_exists = get_eaf_file(args.wav_file)

eaf_data   = Elan.Eaf(file_path=eaf_path if eaf_exists else None)
eaf_tiers  = eaf_data.tiers.keys()

if eaf_exists is not True:
    # Add wav file as linked file to newly created eaf object
    eaf_data.add_linked_file(args.wav_file, relpath=os.path.basename(args.wav_file))

    # Remove 'default' tier from newly created eaf object
    eaf_data.remove_tier('default')

if args.vad_tier not in eaf_data.tiers.keys():
    # Add _vad tier if it doesn't already exist
    eaf_data.add_tier(args.vad_tier)

else:
    # If _vad tier already exists, check if we should overwrite
        # If overwrite, clear the tier first
        # If not, exit script
    if args.overwrite is True:
        eaf_data.remove_all_annotations_from_tier(args.vad_tier, clean=True)

    else:
        print(f"Skipping VAD on {eaf_path}, _vad tier already exists and overwite is set to False (use --overwrite to set to True)")
        exit()

# All conditions met to actually do VAD

waveform, sample_rate = torchaudio.load(args.wav_file)

if sample_rate != 16_000:
    print("Resampling audio to 16 kHz ...")
    samp_to_16k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16_000)
    waveform    = samp_to_16k(waveform)

print("Loading VAD model ...")

vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = vad_utils

print(f"Detecting speech regions in {args.wav_file} ...")

speech_timestamps = get_speech_timestamps(waveform, vad_model, sampling_rate=16_000)

for ts in speech_timestamps:

    # Basically (sample_number/16000) * 1000 or just sample_number/16
    start_ms, end_ms = [ts['start']/16, ts['end']/16]

    eaf_data.add_annotation(args.vad_tier, start=round(start_ms), end=round(end_ms), value='')

eaf_data.to_file(eaf_path)
print(f"Detected speech regions written to {args.vad_tier} tier in {eaf_path}")
