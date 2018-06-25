#!/bin/sh
echo "====="
echo "PATH >>> $PATH"
echo "====="

base="."
if [ "$1" != "" ]; then
    base=$1
fi

url_list="
https://github.com/openai/baselines.git
https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"
for url in $url_list
do
    d=$base/$(basename $url | perl -pe 's/.git$//')
    git clone $url $d
    cd $d
    if [ -f "./requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    if [ -f "./setup.py" ]; then
        pip install -e .
    fi
done

cd $base/pytorch-a2c-ppo-acktr
ln -s ../baselines/baselines .
