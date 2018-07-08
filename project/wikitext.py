#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pathlib


class WikiTextLoader(object):
    def __init__(self, indir: str="data/parsed", batch_size: int=32, residual: bool=False):
        self.indir = indir
        self.batch_size = batch_size
        self.residual = residual
        assert batch_size > 0

    def iter(self):
        doc_list = pathlib.Path(self.indir).glob("doc/**/*.txt")
        docs = []
        titles = []
        for idx, doc_p in enumerate(doc_list):
            def _get_title_p(doc_p):
                title_file = doc_p.as_posix().replace("/doc/", "/title/")
                title_p = pathlib.Path(title_file)
                return title_p
            title_p = _get_title_p(doc_p)
            with doc_p.open("r") as fr:
                doc = fr.read().rstrip()
            with title_p.open("r") as fr:
                title = fr.read().rstrip()
            docs.append(doc)
            titles.append(title)
            assert len(docs) == len(titles)
            if idx % self.batch_size == self.batch_size - 1:
                yield titles, docs
                titles = []
                docs = []
        assert len(docs) == len(titles)
        if self.residual and docs != []:
            yield titles, docs
