#!/usr/bin/bash

# Data subset experiments 
declare -a subsets=("train-100" "train-80" "train-60" "train-40" "train-20" "train-10" "train-05" "train-01")

for i in "${subsets[@]}"
do
   python scripts/train_asr-by-w2v2-ft.py \
   facebook/wav2vec2-large-robust-ft-swbd-300h \
   "data/exps/asr/checkpoints/$i" \
   "data/exps/asr/datasets/$i.tsv" \
   data/exps/asr/datasets/test.tsv \
   --use_target_vocab False
done

# Cross-validation experiments without language model
for j in {1..10}
do
   python scripts/train_asr-by-w2v2-ft.py \
   facebook/wav2vec2-large-robust-ft-swbd-300h \
   "data/exps/asr/checkpoints/bootstrap/no-lm/b-$j" \
   "data/exps/asr/datasets/bootstrap-$j-train01.tsv" \
   "data/exps/asr/datasets/bootstrap-$j-test20.tsv" \
   --use_target_vocab False
done

# Cross-validation experiments with a bigram language model
for k in {1..10}
do
   python scripts/train_asr-by-w2v2-ft.py \
   facebook/wav2vec2-large-robust-ft-swbd-300h \
   "data/exps/asr/checkpoints/bootstrap/lm/b-$k" \
   "data/exps/asr/datasets/bootstrap-$k-train01.tsv" \
   "data/exps/asr/datasets/bootstrap-$k-test20.tsv" \
   --lm_arpa data/exps/asr/datasets/20220422_2gram-correct.arpa \
   --use_target_vocab False
done
