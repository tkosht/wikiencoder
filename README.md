# The project for WikiEncoder in PyTorch
This project is to create the encoder for wikipedia(enwiki).
The main script is named `seq2vec.py`, which encode the sequence of vectors (for wiki texts embeded by word2vec)
to the single vector, and to learn to decode the original sequence of word vectors
using core model `SequenceEncoder` in `sequoder.py`, which consists of LSTM and Linear models simply.

## Prerequisites
- Ubuntu 16.04
- Python 3.6.5
- CPU or NVIDIA GPU

## Run all
```
make all
```
### to run seperatively
```
make preprocess
make vectorizer
make run-visdom-server seq2vec
```

## Stop the visdom server
you should kill visdom server after run.
```
make kill-visdom-server
```
