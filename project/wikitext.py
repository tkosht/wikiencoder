#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
import numpy
from nltk import tokenize
from allennlp.data.tokenizers.word_tokenizer import SpacyWordSplitter


def _get_title_p(doc_p):
    title_file = doc_p.as_posix().replace("/doc/", "/title/")
    title_p = pathlib.Path(title_file)
    return title_p


class WikiTextLoader(object):
    def __init__(self, indir: str="data/parsed", do_tokenize=True,
                 batch_size: int=32, with_title=False, residual: bool=False
                 ):
        self.indir = indir
        self.batch_size = numpy.inf if batch_size == -1 else batch_size
        self.do_tokenize = do_tokenize
        self.with_title = with_title
        self.residual = residual
        assert batch_size > 0

    def iter(self):
        tokenizer = SpacyWordSplitter(pos_tags=True)
        doc_list = pathlib.Path(self.indir).glob("doc/**/*.txt")
        batch = []
        docs = []
        titles = []
        paths = []
        for idx, doc_p in enumerate(doc_list):
            with doc_p.open("r") as fr:
                doc = fr.read().rstrip()
            if self.do_tokenize:
                doc = [tokenizer.split_words(s) for s in tokenize.sent_tokenize(doc)]
            docs.append(doc)
            paths.append(doc_p)
            batch = docs, paths
            assert len(docs) == len(paths)

            if self.with_title:
                title_p = _get_title_p(doc_p)
                with title_p.open("r") as fr:
                    title = fr.read().rstrip()
                if self.do_tokenize:
                    title = tokenizer.split_words(title)
                assert len(docs) == len(titles)
                titles.append(title)
                batch = titles, docs, paths

            if idx % self.batch_size == self.batch_size - 1:
                yield batch
                titles = []
                docs = []
                paths = []
        assert (not self.with_title) or (len(docs) == len(titles))
        if self.residual and docs != []:
            yield batch
