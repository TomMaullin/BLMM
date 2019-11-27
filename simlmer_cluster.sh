#!/bin/bash

RealPath() {
    (echo $(cd $(dirname "$1") && pwd -P)/$(basename "$1"))
}

BLM_PATH=$(dirname $(RealPath "${BASH_SOURCE[0]}"))

i=1
n=100
while [ "$i" -le "$n" ]; do

  # Submit n simulations 
  qsub $BLM_PATH/lib/simlmer2.R 
  i=$(($i + 1))
done
