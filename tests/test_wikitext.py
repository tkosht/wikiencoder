#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pathlib
from project.wikitext import WikiTextLoader


class TestWikiTextLoader(object):
    @pytest.fixture(scope='module')
    def loader(self):
        params = {
            "indir": "data/tests/parsed",
            "batch_size": 5,
            "n_samples": 7,
            "with_title": True,
            "do_tokenize": True,
        }
        wtl = WikiTextLoader(**params)
        wtl = wtl.load()
        return wtl

    def test_iter(self, loader):
        assert not loader.residual
        assert len(loader.doc_list) < 13
        for titles, docs, paths in loader.iter():
            assert len(titles) == loader.batch_size
            assert len(docs) == loader.batch_size
            assert len(paths) == loader.batch_size
            for t, d, p in zip(titles, docs, paths):
                assert isinstance(t, list)
                assert isinstance(d, list)
                assert isinstance(p, pathlib.PosixPath)

    def test_iter_residual(self, loader):
        loader.n_samples = -1
        loader.load()
        loader.residual = True
        for idx, (titles, docs, paths) in enumerate(loader.iter()):
            if idx < 2:
                assert len(titles) == loader.batch_size
                assert len(docs) == loader.batch_size
                assert len(paths) == loader.batch_size
            else:
                assert len(titles) == len(docs)
                assert len(paths) == len(docs)
                assert len(docs) == 3
            for t, d, p in zip(titles, docs, paths):
                assert isinstance(t, list)
                assert isinstance(d, list)
                assert isinstance(p, pathlib.PosixPath)

    def test_iter_non_title(self, loader):
        params = {
            "indir": "data/tests/parsed",
            "batch_size": 5,
            "n_samples": 7,
            "with_title": False,
            "do_tokenize": False,
        }
        loader = WikiTextLoader(**params)
        loader.load()
        for docs, paths in loader.iter():
            for d, p in zip(docs, paths):
                assert isinstance(d, str)
                assert isinstance(p, pathlib.PosixPath)
