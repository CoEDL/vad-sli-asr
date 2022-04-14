import os
import pandas as pd

from helpers.exp import output_path_maker
from helpers.sli import (
    add_sbemb_cols,
    get_logreg_f1,
    get_sb_encoder,
    get_sli_df
)

from multiprocessing import cpu_count
from tqdm.contrib.concurrent import process_map
from sklearn.model_selection import train_test_split

EXP_PARAMS = {
    "exp_name": "20220218_sli",
    "train_clips_dir": "data/clips",
    "output_dir": "data/exps/sli",
    "n_bootstraps": 5000,
    # Note using -1 to indicate 'All data'
    "train_ks": [1, 5, 10, 25, 50, -1]
}

sli_df = get_sli_df(EXP_PARAMS["train_clips_dir"])

print("Extracting SpeechBrain SLI embeddings ...")

sli_df = add_sbemb_cols(sli_df, get_sb_encoder())

def bootstrap_f1(i, sli_df=sli_df):

    train_utts, test_utts   = train_test_split(sli_df, test_size=0.2, shuffle=True, random_state=i)

    kshot_f1s = []

    for k in EXP_PARAMS['train_ks']:

        train_subset = train_utts if k < 0 else train_utts.groupby('lang').apply(lambda x: x.sample(n=k, random_state=i)).reset_index(drop=True)

        f1, _ = get_logreg_f1(train_subset, test_utts)

        kshot_f1s.append({"i" : i, "k" : k, "f1": f1})

    return pd.DataFrame(kshot_f1s)

print("Running k-shot experiments ...")

exp1_results = process_map(bootstrap_f1, range(EXP_PARAMS['n_bootstraps']), max_workers=cpu_count(), chunksize=1)

all_f1s_df   = pd.concat(exp1_results, ignore_index=True)

# Write out SpeechBrain embeddings as experiment artefacts 
sli_df.to_pickle(os.path.join(EXP_PARAMS['output_dir'], "sli-df.pkl"))

# Write out experiment results
all_f1s_df.to_csv(os.path.join(EXP_PARAMS['output_dir'], "kshot-f1s.csv"), index=False)

print(f"Experiments finished. Outputs written to: {EXP_PARAMS['output_dir']}")
