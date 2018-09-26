#!/bin/sh

d=$(cd $(dirname $0) && pwd)

env_name="lab"
if [ "$1" != "" ]; then
    env_name="$1"
fi

# pyversion="miniconda3-latest"
# pyversion="miniconda3-3.19.0"
pyversion="3.6.5"
pyenv install -s $pyversion


mkdir -p ~/pj/$env_name
cd ~/pj/$env_name
pyenv virtualenv $pyversion $env_name
pyenv local $env_name

pip install -U pip
pip install -y http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
pip install -y torchvision

echo pip install -r $d/requirements.txt
pip install -r $d/../../requirements.txt
