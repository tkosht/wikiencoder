#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import argparse
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


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
    parser.add_argument("-b", "--batch-size", type=int, default="1024",
                        help="you can specify the number of mini batch size. default: '1024'")
    parser.add_argument("--samples", type=int, default=2*1000*1000,
                        help="you can specify the number of samples which is the set of paths in indir."
                        " default: '2*1000*1000', if 0 or negative number, will use all of samples.")
    args = parser.parse_args()
    return args


class Vectorize(object):
    def __init__(self, indir, batch_size, n_samples):
        assert batch_size > 0
        self.indir = indir
        self.batch_size = batch_size
        self.n_samples = n_samples

        self.tagdocs = []
        self.model = None

    def run(self):
        self.train()
        self.save()

    @deco.trace
    def train(self):
        class WikiDocuments(object):
            def __init__(self, indir, batch_size, n_samples):
                self.wt = WikiTextLoader(indir, batch_size=batch_size, n_samples=n_samples,
                                    do_tokenize=True, with_title=False, residual=True)
            def __iter__(self):
                for docs, paths in self.wt.iter():
                    for doc, path in zip(docs, paths):
                        yield TaggedDocument(doc, [f"{re.sub('{self.indir}/doc/', '', str(path))}"])
                        del doc
                    del docs

        wikidocs = WikiDocuments(self.indir, self.batch_size, self.n_samples)
        self.model = Doc2Vec(wikidocs, vector_size=128, window=7, min_count=1,
                             dm=1, hs=0, negative=10, dbow_words=1,
                             workers=4)

    @deco.trace
    def save(self, model_file="model/doc2vec.model"):
        self.model.save(model_file)


@deco.trace
@deco.excep
def main():
    args = get_args()
    v = Vectorize(indir=args.indir, batch_size=args.batch_size, n_samples=args.samples)
    v.run()


if __name__ == '__main__':
    r = main()
    if r != 0:
        logfile = deco.logger.logger.handlers[0].baseFilename
        print(f"Abort with error. see logfile '{logfile}'")
    exit(r)
