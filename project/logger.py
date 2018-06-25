#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import traceback
import logging.config
from logging import getLogger


class Logger(object):
    def __init__(self, logger_name="lib", conf_file='config/logging.conf'):
        logging.config.fileConfig(conf_file)
        self.logger = getLogger(logger_name)

    def _format(self, msg: str, args: tuple=(), kwargs: dict={}, e: Exception=None, stacktrace: str=""):
        _msg = msg
        if args != ():
            _msg += " / Params: " + self._format_params(args)
        if kwargs != {}:
            _msg += " / KeyParams: " + self._format_params(kwargs)
        if stacktrace != "":
            _msg += " / Caught Exception: " + self._format_stacktrace(traceback.format_exc())
        return _msg

    def _format_params(self, params):
        return str(params)

    def _format_stacktrace(self, stacktrace: str):
        _stacktrace = re.sub(r"\r?\n", " ", stacktrace)
        return _stacktrace

    def debug(self, msg, args: tuple=(), kwargs: dict={}):
        _msg = self._format(msg, args, kwargs)
        self.logger.debug(_msg)

    def info(self, msg, args: tuple=(), kwargs: dict={}):
        _msg = self._format(msg, args, kwargs)
        self.logger.info(_msg)

    def warn(self, msg, args: tuple=(), kwargs: dict={}, e: Exception=None, stacktrace: str=""):
        _msg = self._format(msg, args, kwargs, e, stacktrace)
        self.logger.warn(_msg)

    def error(self, msg, args: tuple=(), kwargs: dict={}, e: Exception=None, stacktrace: str=""):
        _msg = self._format(msg, args, kwargs, e, stacktrace)
        self.logger.error(_msg)

    def critical(self, msg, args: tuple=(), kwargs: dict={}, e: Exception=None, stacktrace: str=""):
        _msg = self._format(msg, args, kwargs, e, stacktrace)
        self.logger.critical(_msg)
