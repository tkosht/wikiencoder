#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import traceback
import tests.test_util as test_util
import project.decorator as decorator
from project.logger import Logger


decorator.change_logger(test_util.logger_name)


class TestDecorator(object):
    def test_decorator(self):
        @decorator.trace
        @decorator.excep
        def run_test(a=1, b=3, x='x', z='z'):
            1 / 0
        run_test()
        logged = test_util.get_logged(n_tails=3)
        assert re.search(r"INFO .+ Start ", logged[0])
        assert re.search(r"ERROR .+ ErrorOccured", logged[1])
        assert re.search(r"INFO .+ End ", logged[2])
