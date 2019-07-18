#!/bin/bash

# in rcc-land, get a unu GLK trusts
export PATH=/home/glk/teem-install/bin:$PATH

N=11000 # how many images to make

if [ ! -d images ]; then
    mkdir images
fi
if [ ! -d labels ]; then
    mkdir labels
fi
rm -f labels/????.txt images/????.png
for I in $(seq $N); do
    echo $I/$N
    II=$(printf %05d $[$I])
    ./genobjs.sh $II
    ./genimg.sh $II
done

