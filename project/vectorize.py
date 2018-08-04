#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import argparse
from nltk import tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from allennlp.data.tokenizers.word_tokenizer import SpacyWordSplitter


import project.decorator as deco
from project.wikitext import WikiTextLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False,
                        help="if you specified, execute as debug mode. default: 'False'")
    parser.add_argument("--trace", action="store_true", default=False,
                        help="if you specified, execute as trace mode. default: 'False'")
    parser.add_argument("-i", "--indir", type=str, default="data/parsed",
                        help="you can specify the string of the input directory"
                        " must includes subdir 'doc/', and 'title/'. default: 'data/parsed'")
    parser.add_argument("-b", "--batch-size", type=int, default="4096",
                        help="you can specify the number of mini batch size."
                        " default: '32'")
    args = parser.parse_args()
    return args


class Vectorize(object):
    def __init__(self, indir, batch_size):
        assert batch_size > 0
        self.indir = indir
        self.batch_size = batch_size

        self.tagdocs = []
        self.model = None

    def run(self):
        self.load()
        self.train()
        self.save()

    @deco.trace
    def load(self):
        allen_tokenizer = SpacyWordSplitter(pos_tags=True)
        wt = WikiTextLoader(self.indir, batch_size=self.batch_size, with_title=False, residual=True)
        self.tagdocs = [
            TaggedDocument(doc, [f"{re.sub('{self.indir}/doc/', '', str(path))}"]
            )
            for m, (docs, paths) in enumerate(wt.iter())
            for k, (doc, path) in enumerate(zip(docs, paths))
            ]

    @deco.trace
    def train(self):
        self.model = Doc2Vec(self.tagdocs, vector_size=128, window=3, min_count=1,
                             dm=1, hs=0, negative=10, dbow_words=1,
                             workers=4)

    @deco.trace
    def save(self, model_file="model/doc2vec.model"):
        self.model.save(model_file)


@deco.trace
@deco.excep
def main():
    args = get_args()
    v = Vectorize(indir=args.indir, batch_size=args.batch_size)
    v.run()


if __name__ == '__main__':
    r = main()
    if r != 0:
        logfile = deco.logger.logger.handlers[0].baseFilename
        print(f"Abort with error. see logfile '{logfile}'")
    exit(r)
