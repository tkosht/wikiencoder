#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import traceback
from project.logger import Logger

logger_name = 'app'
logger = Logger(logger_name)

def change_logger(logger_name='test'):
    global logger
    logger = Logger(logger_name)


def trace(f: callable) -> callable:
    """ trace decorator

    Args:
        f: decorated function

    Returns:
        wrapper function

    >>> fn = lambda : print("hello")
    >>> lg = trace(fn)
    >>> type(lg)
    <class 'function'>
    >>> lg.__name__
    '<lambda>'
    """
    assert callable(f)

    @functools.wraps(f)
    def tracer(*args, **kwargs) -> None:
        """ tracer function for f

        write log before and after calling 'f'

        Args:
            args: args for the function f
            kwargs:kwargs for the function f
        """
        logger.info(f'Start {f.__qualname__}', args, kwargs)
        r = f(*args, **kwargs)
        logger.info(f'End {f.__qualname__}', args, kwargs)
        return r
    return tracer


def excep(f: callable) -> callable:
    """

    :param f: decorated function
    :return: wrapper function
    """
    assert callable(f)

    @functools.wraps(f)
    def exceptor(*args, **kwargs) -> None:
        try:
            f(*args, **kwargs)
        except Exception as e:
            err_info = traceback.format_exc(-1, chain=e)   # - just the last stack
            logger.error(f'ErrorOccured {f.__qualname__}', args=(), kwargs=locals(), e=e, stacktrace=err_info)
        return
    return exceptor
