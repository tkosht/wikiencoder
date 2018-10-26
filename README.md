# The project for WikiEncoder in PyTorch
This project is to create the encoder for wikipedia(enwiki).
The main script is `project/seq2vec.py`, which encode the title sequence of word vectors
(embeded wiki texts by word2vec)
to the single vector, and to learn to decode the original sequence of word vectors
using core model `SequenceEncoder` in `project/sequoder.py`, which consists of LSTM and Linear models simply.

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
### also use 'run' target
instead of specifying all of vectorizer, run-visdom-server, seq2vec,
you can use only `run` target, like just run `make run` as follows.
```
make preprocess
make run
```

### output example of seq2vec
output format is as follows:
```
word1 word2 ...
    -> predicted_word1(similarity1) predicted_word2(similarity2) ... 
```

after 1000 epochs, every similarity(cosine similarity) is 1.00,
which means seq2vec decoded completely.
but, note that in this example, just only around 100 titles to encode.
if you need more titles to encode, you have to spend the time(i.e. epochs) to train seq2vec encoder.

```
loss[1000]: 1.026806
mikhail vanyov __eos__ `` `` `` ``
        -> jhonny(1.00) street(1.00) __eos__(1.00)

watling chase community forest __eos__ `` ``
        -> watling(1.00) chase(1.00) community(1.00) forest(1.00) __eos__(1.00)                                        

la chapelle-urée __eos__ `` `` `` ``
        -> la(1.00) chapelle-urée(1.00) __eos__(1.00)

kske __eos__ `` `` `` `` ``
        -> kske(1.00) __eos__(1.00)

syria–turkey relations __eos__ `` `` `` ``
        -> syria–turkey(1.00) relations(1.00) __eos__(1.00)

mikhail vanyov __eos__ `` `` `` ``
        -> jhonny(1.00) street(1.00) __eos__(1.00)

watling chase community forest __eos__ `` ``
        -> watling(1.00) chase(1.00) community(1.00) forest(1.00) __eos__(1.00)                                        

la chapelle-urée __eos__ `` `` `` ``
        -> la(1.00) chapelle-urée(1.00) __eos__(1.00)

kske __eos__ `` `` `` `` ``
        -> kske(1.00) __eos__(1.00)

syria–turkey relations __eos__ `` `` `` ``
        -> syria–turkey(1.00) relations(1.00) __eos__(1.00)
```

## Stop the visdom server
you should kill visdom server after run-visdom-server.
```
make kill-visdom-server
```


