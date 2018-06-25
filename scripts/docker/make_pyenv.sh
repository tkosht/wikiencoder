#!/bin/sh

d=$(cd $(dirname $0) && pwd)

env_name="lab"
if [ "$1" != "" ]; then
    env_name="$1"
fi

conda="miniconda3-latest"
pyenv install -s $conda
pyenv versions
pyenv virtualenv $conda $env_name

mkdir -p ~/pj/$env_name
cd ~/pj/$env_name
pyenv local $env_name

echo pip install -r $d/requirements.txt
pip install -r $d/../../requirements.txt
