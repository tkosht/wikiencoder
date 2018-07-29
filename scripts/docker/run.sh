#!/bin/sh
. $(dirname $0)/init.sh

opts=""
cmd=""
if [ "$1" = "-it" ]; then
    opts="-it"
    shift
else
    cmd="nvidia-smi"
    if [ "$1" != "" ]; then
        cmd="$@"
    fi
fi


docker run --runtime=nvidia --rm $opts $docker_image $cmd
