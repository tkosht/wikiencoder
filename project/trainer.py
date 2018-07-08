#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import project.decorator as deco
from project.wikitext import WikiTextLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False,
                        help="if you specified, execute as debug mode. default: 'False'")
    parser.add_argument("--trace", action="store_true", default=False,
                        help="if you specified, execute as trace mode. default: 'False'")
    parser.add_argument("-i", "--indir", type=str, default="data/parsed",
                        help="you can specify the string of the input directory"
                        " must includes subdir 'doc/', and 'title/'. default: 'data/parsed'")
    parser.add_argument("-b", "--batch-size", type=int, default="32",
                        help="you can specify the number of mini batch size."
                        " default: '32'")
    args = parser.parse_args()
    return args


@deco.trace
@deco.excep
def main():
    args = get_args()
    wt = WikiTextLoader(args.indir, batch_size=args.batch_size)
    for titles, docs in wt.iter():
        print(titles)
        break


if __name__ == '__main__':
    r = main()
    if r != 0:
        logfile = deco.logger.logger.handlers[0].baseFilename
        print(f"Abort with error. see logfile '{logfile}'")
    exit(r)
