# -*- coding: utf-8 -*-

import pytest
import project.deco as deco
from project.config import Config


class TestConfig(object):
    @pytest.fixture(scope='module')
    def config(self):
        return Config("tests/config/project.ini", "tests/config/project.spc")

    def test_get(self, config):
        assert config.get("encoder", "gpu") == 1
        assert config.get("encoder", "epochs") == 10
        assert config.get("encoder", "lr") == 0.001
        assert config.get("encoder", "weight_decay") == 0.01

    def test_setup(self, config):
        keys = [
                ["encoder", "gpu"],
                ["encoder", "epochs"],
                ["encoder", "lr"],
                ["encoder", "weight_decay"],
                ]
        config.setup(keys)
        assert config.gpu == 1
        assert config.epochs == 10
        assert config.lr == 0.001
        assert config.weight_decay == 0.01

    def test_invalid(self, config):
        @deco.excep(with_raise=True)
        def _load_invalid():
            cfg = Config("tests/config/invalid.ini", "tests/config/project.spc")
        try:
            _load_invalid()
            assert False
        except Exception as e:
            print(e)
            assert True
