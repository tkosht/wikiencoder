#!/bin/sh
set -eu
d=$(cd $(dirname $0) && pwd)
cd $d/../data

n=100000
n_samples=13
doc_list=$(find parsed/doc/ -type f  | head -n $n | shuf -n $n_samples /dev/stdin)

test_dir="tests"
rm -rf $test_dir
mkdir -p $test_dir
for doc_file in $doc_list
do
    title_file=$(echo $doc_file | perl -pe 's:/doc/:/title/:')
    doc_dir=$test_dir/$(dirname $doc_file)
    title_dir=$test_dir/$(dirname $title_file)
    echo "cp" $doc_file "-> $doc_dir" "/" $title_file "-> $title_dir"
    mkdir -p $doc_dir $title_dir
    cp $doc_file $doc_dir
    cp $title_file $title_dir
done
