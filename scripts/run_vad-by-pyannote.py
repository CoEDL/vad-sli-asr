from pyannote.audio.pipelines import VoiceActivityDetection
from argparse import ArgumentParser
import pympi.Elan as Elan
import os
import sys
import torch

from helpers.eaf import get_eaf_file

parser = ArgumentParser(
    prog='run_vad-by-pyannote',
    description='Voice activity detection with PyAnnote. Writes intervals onto _vad tier in sidecar file.',
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

print(f"Detecting speech regions in {args.wav_file} ...")

vad = VoiceActivityDetection({ 
    "checkpoint":"pyannote/segmentation",
    "map_location": torch.device("cuda" if torch.cuda.is_available() else "cpu")
  }) \
  .instantiate({
    "onset": 0.5,
    "offset": 0.5,
    "min_duration_on": 0.0,
    "min_duration_off": 0.0
  })

segments = vad(args.wav_file)

for segment in segments.itersegments():
  start_ms = int(segment.start * 1000)
  end_ms   = int(segment.end * 1000)
  
  eaf_data.add_annotation(args.vad_tier, start=start_ms, end=end_ms, value='')

eaf_data.to_file(eaf_path)
print(f"Detected speech regions written to {args.vad_tier} tier in {eaf_path}")
