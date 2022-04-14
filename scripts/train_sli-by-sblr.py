import pickle

from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

from helpers.sli import get_sli_df, get_sb_encoder, add_sbemb_cols, colsplit_feats_labels

parser = ArgumentParser(
    prog='train_sli-by-sblr',
    description='Train a logistic regression classifier for spoken language identification using SpeechBrain embeddings as input',
)

parser.add_argument('clips_dir', help = "directory containing audio clips as wav files (1 subdirectory for each language)")
parser.add_argument('logreg_pkl', help = "path to save fitted logistic regression classifier")

parser.add_argument('--logreg_maxiter', default=1000, help="Maximum number of iterations for fitting classifier")

args = parser.parse_args()

sli_df = get_sli_df(args.clips_dir)

print("Extracting features...")

sli_df = add_sbemb_cols(sli_df, sb_encoder=get_sb_encoder())

feats, labels = colsplit_feats_labels(sli_df)
feats, labels = shuffle(feats, labels, random_state=0)

print("Fitting classifier...")
clf = LogisticRegression(random_state=0, max_iter=args.logreg_maxiter).fit(feats, labels)

pickle.dump(clf, open(args.logreg_pkl, 'wb'))
print(f"Saved classifier to {args.logreg_pkl}")
