#!/bin/bash

RANGE="64 128 256 512 1024 2048 4096" #8192"
export OMP_NUM_THREADS=24

# Comment naive and order if tou want to run 8192 or more as it will run for hour(s)
for i in $RANGE; do echo $i && ./naive $i >> time.naive.txt; done
for i in $RANGE; do echo $i && ./order $i >> time.order.txt; done
for i in $RANGE; do echo $i && ./threads $i >> time.threads.txt; done
for i in $RANGE; do echo $i && ./target $i >> time.target.txt; done
for i in $RANGE; do echo $i && ./cuda $i >> time.cuda.txt; done

