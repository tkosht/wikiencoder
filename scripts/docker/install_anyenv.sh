#!/bin/sh

d=$(cd $(dirname $0) && pwd)

rm -rf ~/.anyenv
git clone https://github.com/riywo/anyenv ~/.anyenv
mkdir -p ~/.anyenv/envs/pyenv
~/.anyenv/bin/anyenv install pyenv
git clone https://github.com/yyuu/pyenv-virtualenv.git ~/.anyenv/envs/pyenv/plugins/pyenv-virtualenv

# - for anyenv
bash_rc='
if [ -d $HOME/.anyenv ]
then
    export PATH="$PATH:$HOME/.anyenv/bin"
    eval "$(anyenv init -)"
fi
if [ -d ~/.anyenv/envs/pyenv/plugins/pyenv-virtualenv ]
then
    eval "$(pyenv virtualenv-init -)"
fi
'
echo "$bash_rc" >> ~/.bashrc

eval "$bash_rc"
sh $d/make_pyenv.sh lab
