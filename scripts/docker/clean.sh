#!/bin/sh

except='(nvidia/cuda|CONTAINER|REPOSITORY)'
runnnig_file=".runnnig"
docker ps | egrep -v "$except" | awk '{print $1}' > $runnnig_file
container_list=$(docker ps -a | egrep -v "$except" \
    | egrep -v -f $runnnig_file | awk '{print $1}')
for ctr in $container_list
do
    docker rm $ctr
done

image_list=$(docker images | egrep -v "$except" \
    | awk '{print $3}')
for img in $image_list
do
    docker rmi $img
done

