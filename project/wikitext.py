# -*- coding: utf-8 -*-

import os
import pathlib
import numpy
import string
from tqdm import tqdm
from nltk import tokenize
from nltk.corpus import stopwords
import project.deco as deco

stop_words = set(stopwords.words("english")) & set(list(string.punctuation))


def _get_title_p(doc_p):
    title_file = doc_p.as_posix().replace("/doc/", "/title/")
    title_p = pathlib.Path(title_file)
    return title_p


@deco.trace
def save_samples_list(docs, indir):
    p = pathlib.Path(indir)
    with p.joinpath("samples.list").open("w") as f:
        for elm in docs:
            line = f"{str(elm)}{os.linesep}"
            f.writelines(line)

def tokenize_word(sentence):
    _words = tokenize.word_tokenize(sentence)
    words = [str(w).lower() for w in _words if w not in stop_words]
    words.append("__eos__")
    return words


class WikiTextLoader(object):
    def __init__(self, indir: str="data/parsed", do_tokenize: bool=False, n_samples: int=0,
                 batch_size: int=32, with_title: bool=False, residual: bool=False
                 ):
        self.indir = indir
        self.batch_size = numpy.inf if batch_size == -1 else batch_size
        self.do_tokenize = do_tokenize
        self.n_samples = n_samples
        self.with_title = with_title
        self.residual = residual
        assert batch_size > 0

    @deco.trace
    def filter_samples(self, doc_list):
        docs = numpy.array([p for p in doc_list])
        n = len(docs)
        n_samples = min(n, self.n_samples)
        filter_flags = numpy.random.binomial(1, n_samples/n, n)
        filter_flags = filter_flags.astype(numpy.bool8)
        doc_list = docs[filter_flags]
        save_samples_list(doc_list, self.indir)
        return doc_list

    def load(self):
        doc_list = pathlib.Path(self.indir).glob("doc/**/*.txt")
        if self.n_samples > 0:
            doc_list = self.filter_samples(doc_list)
        self.doc_list = doc_list

    def iter(self):
        batch = []
        docs = []
        titles = []
        paths = []
        for idx, doc_p in tqdm(enumerate(self.doc_list)):
            with doc_p.open("r") as fr:
                doc = fr.read().rstrip()
            if self.do_tokenize:
                doc = tokenize_word(doc)

            docs.append(doc)
            paths.append(doc_p)
            batch = docs, paths
            assert len(docs) == len(paths)

            if self.with_title:
                title_p = _get_title_p(doc_p)
                with title_p.open("r") as fr:
                    title = fr.read().rstrip()
                if self.do_tokenize:
                    title = tokenize_word(title)
                titles.append(title)
                assert len(docs) == len(titles)
                batch = titles, docs, paths

            if idx % self.batch_size == self.batch_size - 1:
                yield batch
                titles = []
                docs = []
                paths = []
        assert (not self.with_title) or (len(docs) == len(titles))
        if self.residual and docs != []:
            yield batch
