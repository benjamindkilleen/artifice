#!/bin/bash

# get a unu GLK trusts
export PATH=/home/glk/teem-install/bin:$PATH

N=2000 # how many images to make

mkdir images
mkdir labels
rm -f labels/????.txt images/????.png
for I in $(seq $N); do
    echo $I/$N
    II=$(printf %04d $[$I+1000])
    ./genobjs.sh $II
    ./gendata.sh $II
done

