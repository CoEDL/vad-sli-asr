import os
import pandas as pd
import torchaudio

from datasets import Dataset
from helpers.asr import configure_w2v2_for_inference
from jiwer import wer, cer

EVAL_MODELS_DATASETS = [
    ("facebook/wav2vec2-large-robust-ft-swbd-300h", "data/exps/asr/datasets/bootstrap-1-test20.tsv"),
    ("data/exps/asr/checkpoints/bootstrap/no-lm/b-1", "data/exps/asr/datasets/bootstrap-1-test20.tsv")
]

EVAL_RESULTS = []

def read_clip(batch):
    batch['speech'] = torchaudio.load(batch['path'])[0]
    return batch

def make_all_lowercase(batch):
    batch["sentence"] = batch["sentence"].lower()
    batch["transcription"] = batch["transcription"].lower()

    return batch

for model_path, testset_path in EVAL_MODELS_DATASETS:

    print(f"Reading in data from {testset_path} ...")
    test_ds = Dataset.from_pandas(pd.read_csv(testset_path, sep = '\t'))
    test_ds = test_ds.map(read_clip)

    _, processor, transcribe_speech = configure_w2v2_for_inference(model_path)

    print(f"Obtaining predictions using model from {model_path} ...")
    test_ds = test_ds.map(transcribe_speech, remove_columns=["speech"])
    test_ds = test_ds.map(make_all_lowercase)

    EVAL_RESULTS.append({
        "model" : os.path.basename(model_path),
        "model_lm" : type(processor).__name__ == 'Wav2Vec2ProcessorWithLM',
        "testset" : os.path.basename(testset_path),
        "wer" : round(wer(test_ds['sentence'], test_ds['transcription']), 2),
        "cer" : round(cer(test_ds['sentence'], test_ds['transcription']), 2)
    })

results_df = pd.DataFrame(EVAL_RESULTS)
results_df.to_csv("data/exps/asr/asr_wer-csr.csv")

print("Results written to data/exps/asr/asr_wer-csr.csv")
