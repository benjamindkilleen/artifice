#!/bin/bash
set -o errexit
set -o nounset
shopt -s expand_aliases
JUNK=""
function junk { JUNK="$JUNK $@"; }
function cleanup { rm -rf $JUNK; }
trap cleanup err exit int term

export NRRD_STATE_DISABLE_CONTENT=true

PFX=$1 # prefix for input and output files
IN=labels/$PFX.txt
OUT=images/$PFX.png

SX=100  # output image resolution
SY=100
SH=9 # sharpness of object edge

RNG=$(cksum $IN | cut -d' ' -f 1) # RNG seed determined by input
NOISE=0.04 # gaussian noise on signals in [0,1]
QMIN=-0.2  # quantization range to 8-bit
QMAX=1.2

echo "-1 1 -1 1" |
unu reshape -s 2 2 |
unu resample -s $SX $SY -k tent -c node -o x; junk x
unu swap -i x -a 0 1 | unu flip -a 1 -o y; junk y

unu 2op x x 0 -o tmp; junk tmp

while IFS= read -r xysp; do
    xyspA=($xysp)
    XX=${xyspA[0]}
    YY=${xyspA[1]}
    SZ=${xyspA[2]}
    PP=${xyspA[3]}
    unu 2op - x $XX | unu 1op abs | unu 2op / - $SZ | unu 2op pow - $PP -o xp
    unu 2op - y $YY | unu 1op abs | unu 2op / - $SZ | unu 2op pow - $PP |
    unu 2op + xp - | # x^p + y^p
    unu 2op pow - `echo 1/$PP | bc -l` | # (x^p + y^p)^(1/p)
    unu 2op - 1 - | # pre thresholding
    unu 2op x - $SH | unu 2op x - $SZ | unu 2op x - 5 | unu 1op erf | # soft thresholding
    unu 2op max tmp - -o tmp
done < $IN

junk xp

unu 2op nrand tmp $NOISE -s $RNG |
unu quantize -b 8 -min $QMIN -max $QMAX -o $OUT
