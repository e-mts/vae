#! /usr/bin/bash

cd ~
cd vae/docker

docker run --gpus all -it -p 8888:8888 -v /home/emts/vae/:/tf/vae vae

