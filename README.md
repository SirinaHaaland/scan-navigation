# SCAN Navigation Instructions

University machine learning project investigating systematic
generalization in sequence-to-sequence models.

## Project objective

Compare an LSTM encoder-decoder and a Transformer encoder-decoder
on the official SCAN random, length, and add-primitive splits.

## Dataset

SCAN: https://github.com/brendenlake/SCAN

Download the official dataset:

git clone https://github.com/brendenlake/SCAN.git data/scan

## Initial data verification

From the project root, run:

python src/dataset.py

The script loads the random-split training data, constructs
input and output vocabularies, prints example translations,
and checks that tokenization can be reversed correctly.

## Status

Initial data inspection and vocabulary construction completed.
Model training and evaluation are under development.