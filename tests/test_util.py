#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib

""" Test Util Class """

logger_name = 'test'
test_logfile = 'log/test_project.log'


def get_logged(n_tails=1) -> list:
    p = pathlib.Path(test_logfile)
    logged = []
    with p.open('r') as f:
        logged = f.readlines()[-n_tails:]
    return logged
