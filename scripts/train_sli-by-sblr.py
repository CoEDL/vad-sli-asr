from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm

import glob
import os
import pandas as pd
import pickle
import torch
import torchaudio

parser = ArgumentParser(
    prog='train_sli-by-sblr',
    description='Train a logistic regression classifier for spoken language identification using SpeechBrain embeddings as input',
)

parser.add_argument('clips_dir', help = "directory containing audio clips as wav files (1 subdirectory for each language)")
parser.add_argument('logreg_pkl', help = "path to save fitted logistic regression classifier")

parser.add_argument('--logreg_maxiter', default=1000, help="Maximum number of iterations for fitting classifier")

args = parser.parse_args()

language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp/")

def get_sb_emb(wav_path):
    waveform, sample_rate = torchaudio.load(wav_path)

    if sample_rate != 16_000:
        print("Resampling audio to 16 kHz ...")
        samp_to_16k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16_000)
        waveform    = samp_to_16k(waveform)

    emb =  language_id.encode_batch(waveform)

    return emb.reshape((1, 256))

wav_files = glob.glob(os.path.join(args.clips_dir, "*", "*.wav"))
langs     = [ os.path.basename(os.path.dirname(f)) for f in wav_files ]

print("Extracting features...")

embds     = pd.concat([ pd.DataFrame(get_sb_emb(f)) for f in tqdm(wav_files) ])

langs, embds = shuffle(langs, embds, random_state=0)

print("Fitting classifier...")
clf = LogisticRegression(random_state=0, max_iter=args.logreg_maxiter).fit(embds, langs)

pickle.dump(clf, open(args.logreg_pkl, 'wb'))
print(f"Saved classifier to {args.logreg_pkl}")
