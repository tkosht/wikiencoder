#!/bin/sh
d=$(cd $(dirname $0) && pwd)
cd $d

project_name=$(basename $(cd $d/../../ && pwd))
docker_image="${project_name}-image"
