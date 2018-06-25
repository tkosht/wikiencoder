#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import pytest
import traceback
import tests.test_util as test_util
from project.logger import Logger

logger_name = test_util.logger_name
pattern_time = r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}.\d{3}"
pattern_normal = f"^{pattern_time} {logger_name} %s [\w\.:]+ %s$"
pattern_except = f"^{pattern_time} {logger_name} %s [\w\.:]+ %s %s$"


class TestLogger(object):
    @pytest.fixture(scope='module')
    def logger(self):
        return Logger(logger_name)

    def test_debug(self, logger):
        logger.debug('Hello')
        logged = test_util.get_logged(n_tails=1)
        pattern = pattern_normal % ('DEBUG', 'Hello')
        assert re.search(pattern, logged[0])

    def test_info(self, logger):
        logger.info('Hello')
        logged = test_util.get_logged(n_tails=1)
        pattern = pattern_normal % ('INFO', 'Hello')
        assert re.search("^" + pattern_time, logged[0])
        assert re.search(pattern, logged[0])
        assert not re.search(r"Params: ", logged[0])
        assert not re.search(r"KeyParams: ", logged[0])

    def test_info_with_args(self, logger):
        tpl = (1, 2, 3)
        logger.info('Hello', args=tpl)
        logged = test_util.get_logged(n_tails=1)
        assert re.search("Params: .*(\(1, 2, 3\))", logged[0])

    def test_info_with_kwargs(self, logger):
        dct = {'a': 1, 'b': 'xyz', 'c': ['world', 3]}
        logger.info('Hello', kwargs=dct)
        logged = test_util.get_logged(n_tails=1)
        assert re.search("KeyParams: .*(\{'a': 1, 'b': 'xyz', 'c': \['world', 3\]\})", logged[0])

    def test_info_with_params(self, logger):
        tpl = (1, 2, 3)
        dct = {'a': 1, 'b': 'xyz', 'c': ['world', 3]}
        logger.info('Hello', tpl, dct)
        logged = test_util.get_logged(n_tails=1)
        assert re.search("Params: .*(\(1, 2, 3\))", logged[0])
        assert re.search("KeyParams: .*(\{'a': 1, 'b': 'xyz', 'c': \['world', 3\]\})", logged[0])
        assert re.search("Params: .*KeyParams: ", logged[0])

    def test_warn(self, logger):
        tpl = (1, 2, 3)
        lst = list(tpl)
        dct = {'a': 1, 'b': 'xyz', 'c': ['world', 3]}
        try:
            1 / 0
        except Exception as e:
            logger.warn('Hello', (), locals(), e, traceback.format_exc())
        logged = test_util.get_logged(n_tails=1)
        pattern_extra = r"/ KeyParams: .+ / Caught Exception: .+ ZeroDivisionError: division by zero\s*"
        pattern = pattern_except % ('WARNING', 'Hello', pattern_extra)
        assert re.search(pattern, logged[0])
        assert re.search(r"'dct': \{'a': 1, 'b': 'xyz', 'c': \['world', 3\]\}", logged[0])
        assert re.search(r"'lst': \[1, 2, 3\]", logged[0])
        assert re.search(r"'tpl': \(1, 2, 3\)", logged[0])

    def test_error(self, logger):
        try:
            1 / 0
        except Exception as e:
            logger.error('Hello', (), locals(), e, traceback.format_exc())
        logged = test_util.get_logged(n_tails=1)
        pattern_extra = r"/ KeyParams: .+ / Caught Exception: .+ ZeroDivisionError: division by zero\s*"
        pattern = pattern_except % ('ERROR', 'Hello', pattern_extra)
        assert re.search(pattern, logged[0])

    def test_critical(self, logger):
        try:
            1 / 0
        except Exception as e:
            logger.critical('Hello', (), locals(), e, traceback.format_exc())
        logged = test_util.get_logged(n_tails=1)
        pattern_extra = r"/ KeyParams: .+ / Caught Exception: .+ ZeroDivisionError: division by zero\s*"
        pattern = pattern_except % ('CRITICAL', 'Hello', pattern_extra)
        assert re.search(pattern, logged[0])
