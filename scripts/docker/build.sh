#!/bin/sh

opt_cache="--no-cache=true"
if [ "$1" = "-d" -o "$1" = "--debug" ]; then
    opt_cache=""
fi

base_dir=$(cd $(dirname $0)/../../ && pwd)
. $(dirname $0)/init.sh

# d=$(cd $(dirname $0)/../../ && pwd)
cd $base_dir
docker build -f ./docker/Dockerfile -t $docker_image $opt_cache .
