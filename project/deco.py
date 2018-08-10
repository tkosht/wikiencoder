#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
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


def excep(f=None, type_exc: type=Exception, with_raise: bool=False, warn: bool=False, do_exit: bool=False) -> callable:
    """

    :param f: decorated function
    :return: wrapper function
    """
    if f is None:
        kwargs = {
            "type_exc": type_exc,
            "with_raise": with_raise,
            "warn": warn,
            "do_exit": do_exit,
        }
        return functools.partial(excep, **kwargs)

    assert callable(f)
    assert issubclass(type_exc, Exception)

    @functools.wraps(f)
    def exceptor(*args, **kwargs) -> None:
        ec = 0
        try:
            r = f(*args, **kwargs)
        except type_exc as e:
            err_info = traceback.format_exc(-1, chain=e)   # - just the last stack
            write_log = logger.warn if warn else logger.error
            write_log(f'{f.__qualname__}', args=(), kwargs=locals(), e=e, stacktrace=err_info)
            if with_raise:
                raise e
            ec = 2 if warn else 1
            if do_exit and not warn:
                logger.error(f'Abort', args=(), kwargs=locals(), e=e, stacktrace=err_info)
                sys.exit(ec)
        return r
    return exceptor
