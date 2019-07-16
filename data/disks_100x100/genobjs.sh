set -o errexit
set -o nounset
shopt -s expand_aliases
JUNK=""
function junk { JUNK="$JUNK $@"; }
function cleanup { rm -rf $JUNK; }
trap cleanup err exit int term

RNG=$1
OUT=labels/$RNG.txt

N=4  # number of objects
SZMIN=0.05  # range of sizes
SZMAX=0.15

# "shape" parameter; 1=diamond, 2=circle, inf=square
PMIN=0.8
PMAX=8

echo "0 0" | unu reshape -s 2 | unu axinsert -a 1 -s $N | # 2 by N array
unu 1op rand -s $RNG | unu affine 0 - 1 -1 1 -o xy; junk xy

echo 0 | unu reshape -s 1 | unu axinsert -a 1 -s $N | # 1 by N array
unu 1op rand -s $[$RNG+1] | unu affine 0 - 1 $SZMIN $SZMAX -o size; junk size

echo 0 | unu reshape -s 1 | unu axinsert -a 1 -s $N | # 1 by N array
unu 1op rand -s $[$RNG+2] | # uniform random
unu 2op + - 0.5 | unu 2op pow - 3 | # more smaller values HEY
unu affine 0.125 - 3.375 $PMIN $PMAX -o parm; junk parm

unu join -i xy size parm -a 0 -o $OUT

