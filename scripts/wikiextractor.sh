#!/bin/sh

d=$(cd $(dirname $0) && pwd)
cd $d

git clone https://github.com/attardi/wikiextractor.git \
    2> /dev/null

datadir="../data"
xmlbz2="$datadir/enwiki-latest-pages-articles.xml.bz2"
if [ "$1" != "" ]; then
    xmlbz2="$1"
fi
outdir="../data/wikitxt"
if [ "$2" != "" ]; then
    outdir="$2"
fi
python wikiextractor/WikiExtractor.py \
    -b 1M -o $outdir $xmlbz2 \
    > /dev/null 2>&1
