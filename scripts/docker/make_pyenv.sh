#!/bin/sh

d=$(cd $(dirname $0) && pwd)

env_name="lab"
if [ "$1" != "" ]; then
    env_name="$1"
fi

#conda="miniconda3-latest"
conda="miniconda3-3.19.0"
pyenv install -s $conda


mkdir -p ~/pj/$env_name
cd ~/pj/$env_name
pyenv virtualenv $conda $env_name
pyenv local $env_name

echo conda install pytorch torchvision cuda90 -c pytorch -y
conda install pytorch torchvision cuda91 -c pytorch -y

echo pip install -r $d/requirements.txt
pip install -r $d/../../requirements.txt
