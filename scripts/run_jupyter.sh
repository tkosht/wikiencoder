#!/bin/sh

d=$(cd $(dirname $0) && pwd)
base_dir=$(cd $d/../ && pwd)
cd $base_dir

cd notebook
jupyter notebook --no-browser > ../log/jupyter-notebook.log 2>&1  &
