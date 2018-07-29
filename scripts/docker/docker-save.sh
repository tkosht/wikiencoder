#!/bin/sh

. $(dirname $0)/init.sh

docker_dir="../../docker"
docker save $docker_image | gzip > $docker_dir/${docker_image}.tar.gz
