#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from project.wikitext import WikiTextLoader


class TestWikiTextLoader(object):
    @pytest.fixture(scope='module')
    def loader(self):
        return WikiTextLoader(indir="data/test/parsed", batch_size=5)

    def test_iter(self, loader):
        assert not loader.residual
        for titles, docs in loader.iter():
            assert len(titles) == loader.batch_size
            assert len(docs) == loader.batch_size

    def test_iter_residual(self, loader):
        loader.residual = True
        for idx, (titles, docs) in enumerate(loader.iter()):
            if idx < 2:
                assert len(titles) == loader.batch_size
                assert len(docs) == loader.batch_size
            else:
                assert len(titles) == len(docs)
                assert len(docs) == 3
