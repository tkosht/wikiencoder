#!/bin/sh

d=$(cd $(dirname $0) && pwd)
cd $d

outdir="../data/parsed"
echo "`date +'%Y/%m/%d %T'` - Start"
echo "`date +'%Y/%m/%d %T'` - Start - Clean old directories"
rm -rf $outdir
echo "`date +'%Y/%m/%d %T'` - End - Clean old directories"

echo "`date +'%Y/%m/%d %T'` - Start - Parse"
for f in $(find ../data/wikitxt -type f)
do
    python sep2doc.py -i $f -o $outdir
done
echo "`date +'%Y/%m/%d %T'` - End - Parse"
echo "`date +'%Y/%m/%d %T'` - End"
