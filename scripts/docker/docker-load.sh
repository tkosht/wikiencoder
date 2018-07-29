#!/bin/sh
. $(dirname $0)/init.sh
docker_dir="../../docker"
docker load < $docker_dir/${docker_image}.tar.gz
