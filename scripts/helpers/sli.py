import glob
import os
import numpy as np
import pandas as pd
import torch
import torchaudio

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.utils._testing import ignore_warnings

from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm

def get_sli_df(sli_train_dir):

    langs    = os.listdir(sli_train_dir)
    lang_dfs = {}

    for l in langs:
        wav_paths = glob.glob(os.path.join(sli_train_dir, l, "*.wav"))
        
        lang_dfs[l] = pd.DataFrame.from_dict({
            'wav_path' : wav_paths,
            'lang' : l
        })

    sli_df = pd.concat(lang_dfs.values(), ignore_index=True)

    return sli_df

def get_sb_encoder(save_dir="tmp"):
    sb_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir=save_dir,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu" }
    )

    return sb_encoder

def add_sbemb_cols(sli_df, sb_encoder):

    def enc_helper(wav_path):
        waveform, sample_rate = torchaudio.load(wav_path)

        if sample_rate != 16_000:
            samp_to_16k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16_000)
            waveform    = samp_to_16k(waveform)

        emb =  sb_encoder.encode_batch(waveform)

        return emb.reshape((1, 256)).cpu().detach().numpy()

    sbemb_df = pd.concat([ pd.DataFrame(enc_helper(f)) for f in tqdm(sli_df["wav_path"].to_list()) ])
    sli_df   = pd.concat([sli_df.reset_index(drop=True), sbemb_df.reset_index(drop=True)], axis=1)

    return sli_df

def colsplit_feats_labels(sli_df):
    # Split data frame columns, return features and labels separately
    return sli_df.iloc[:, -256:], sli_df.lang

@ignore_warnings(category=ConvergenceWarning)
def get_logreg_f1(train_df, test_df):

    train_feats, train_labels = colsplit_feats_labels(shuffle(train_df))
    test_feats, test_labels   = colsplit_feats_labels(test_df)

    logreg = LogisticRegression(class_weight='balanced', max_iter = 1000, random_state=0)
    logreg.fit(train_feats, train_labels)
    test_pred = logreg.predict(test_feats)

    results_dict = classification_report(test_labels, test_pred, output_dict=True, zero_division=0)
    f1 = round(results_dict['weighted avg']['f1-score'], 3)

    return f1, test_pred
