#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import click
import pathlib

@click.command()
@click.option('--infile', '-i', default='wikitxt/AA/wiki_00')
@click.option('--outdir', '-o', default='parsed')
def cmd(infile, outdir):
    p = pathlib.Path(infile)
    with p.open('r') as f:
        n_lines = 0
        title = ""
        doc = []
        n_docs = 0
        for line in f:
            s = re.search(r'<doc\s.+title="([^"]+)"', line)
            if s is not None:   # if matched
                #title = s.group(1)
                n_lines = 1
                n_docs += 1
                continue
            e = re.search(r'^\s*</doc>\s*$', line)
            if e is not None: # if matched
                d = p.parent.name
                output_file = f'{outdir}/doc/{d}/{p.name}/{n_docs:05d}.txt'
                wp = pathlib.Path(output_file)
                wp.parent.mkdir(parents=True, exist_ok=True)
                with wp.open('w') as fw:
                    for l in  doc:
                        fw.write(l)
                title = ""
                doc = []
                continue
            if n_lines == 1:    # title
                d = p.parent.name
                output_file = f'{outdir}/title/{d}/{p.name}/{n_docs:05d}.txt'
                wp = pathlib.Path(output_file)
                eos = "__EOS__"
                line = f"{line.strip()} {eos}{os.linesep}"
                wp.parent.mkdir(parents=True, exist_ok=True)
                with wp.open('w') as fw:
                    fw.write(line)
            elif n_lines == 2:
                pass
            else:
                doc.append(line)
            n_lines += 1
    
def main():
    cmd()

if __name__ == '__main__':
    main()
