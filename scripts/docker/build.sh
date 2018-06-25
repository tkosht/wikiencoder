#!/bin/sh

opt_cache="--no-cache=true"
if [ "$1" = "-d" -o "$1" = "--debug" ]; then
    opt_cache=""
fi

d=$(cd $(dirname $0)/../../ && pwd)
docker build -f $d/docker/Dockerfile $d -t gpuenv:gpuenv $opt_cache
