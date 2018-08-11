#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import tests.test_util as test_util
import project.deco as deco


deco.change_logger(test_util.logger_name)


class TestDecorator(object):
    def test_decorator(self):
        @deco.trace
        @deco.excep
        def run_test(a=1, b=3, x='x', z='z'):
            1 / 0
        run_test()
        logged = test_util.get_logged(n_tails=3)
        assert re.search(r"INFO .+ Start ", logged[0])
        assert re.search(r"ERROR .+ ErrorOccured", logged[1])
        assert re.search(r"INFO .+ End ", logged[2])

    def test_type_exc(self):
        @deco.trace
        @deco.excep()
        @deco.excep(type_exc=ZeroDivisionError, with_raise=True)
        def run_test(a=1, b=3, x='x', z='z'):
            1 / 0
        run_test()
        logged = test_util.get_logged(n_tails=4)
        assert re.search(r"INFO .+ Start ", logged[0])
        assert re.search(r"ERROR .+ ErrorOccured.+ZeroDivisionError.+", logged[1])  # excep(type_exc=*)
        assert re.search(r"ERROR .+ ErrorOccured.+ZeroDivisionError.+", logged[2])  # excep()
        assert re.search(r"INFO .+ End ", logged[3])

    def test_type_exc2(self):
        @deco.trace
        @deco.excep(type_exc=Exception)
        @deco.excep(type_exc=IndexError)
        def run_test(a=1, b=3, x='x', z='z'):
            1 / 0
        run_test()
        logged = test_util.get_logged(n_tails=3)
        assert re.search(r"INFO .+ Start ", logged[0])
        assert re.search(r"ERROR .+ ErrorOccured.+ZeroDivisionError.+", logged[1])
        assert re.search(r"INFO .+ End ", logged[2])

    def test_type_warn(self):
        @deco.trace
        @deco.excep()
        @deco.excep(type_exc=ZeroDivisionError, warn=True)
        def run_test(a=1, b=3, x='x', z='z'):
            1 / 0
        run_test()
        logged = test_util.get_logged(n_tails=3)
        assert re.search(r"INFO .+ Start ", logged[0])
        assert re.search(r"WARNING .+ Warned.+ZeroDivisionError.+", logged[1])
        assert re.search(r"INFO .+ End ", logged[2])
