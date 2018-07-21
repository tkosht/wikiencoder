#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
import numpy


def _get_title_p(doc_p):
    title_file = doc_p.as_posix().replace("/doc/", "/title/")
    title_p = pathlib.Path(title_file)
    return title_p


class WikiTextLoader(object):
    def __init__(self, indir: str="data/parsed", batch_size: int=32, with_title=False, residual: bool=False):
        self.indir = indir
        self.batch_size = numpy.inf if batch_size == -1 else batch_size
        self.with_title = with_title
        self.residual = residual
        assert batch_size > 0

    def iter(self):
        doc_list = pathlib.Path(self.indir).glob("doc/**/*.txt")
        batch = []
        docs = []
        titles = []
        for idx, doc_p in enumerate(doc_list):
            with doc_p.open("r") as fr:
                doc = fr.read().rstrip()
            docs.append(doc)
            batch = docs

            if self.with_title:
                title_p = _get_title_p(doc_p)
                with title_p.open("r") as fr:
                    title = fr.read().rstrip()
                assert len(docs) == len(titles)
                titles.append(title)
                batch = titles, docs

            if idx % self.batch_size == self.batch_size - 1:
                yield batch
                titles = []
                docs = []
        assert (not self.with_title) or (len(docs) == len(titles))
        if self.residual and docs != []:
            yield batch
