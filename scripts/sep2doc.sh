#!/bin/sh

d=$(cd $(dirname $0) && pwd)
cd $d

outdir="../data/parsed"
rm -rf $outdir

echo "`date +'%Y/%m/%d %T'` - Start"
for f in $(find ../data/wikitxt -type f)
do
    python sep2doc.py -i $f -o $outdir
done
echo "`date +'%Y/%m/%d %T'` - End"
