#!/bin/sh

d=$(cd $(dirname $0) && pwd)
cd $d

#lang="ja"
lang="en"
url="https://dumps.wikimedia.org/${lang}wiki/latest"
url="$url/${lang}wiki-latest-pages-articles.xml.bz2"
outdir="../data"
if [ "$1" != "" ]; then
    outdir=$1
fi
outfile="$outdir/$(basename $url)"

curl $url -o $outfile
