#!/bin/bash

i=1
for gp in $(seq 0 0.02 0.18); do
  cat <<EOF > input_${i}.toml
N = 160
Ncut = 22
Nup = 32
Ndn = 32
U = 8.0
V = 0.0
w = 0.2
g = 0.25
g1 = $gp
bare_n = 100
init_n = 2
LBO_dims = [100,80,60,30,40,30,20,8]
pbc = false
EOF
  echo "Generated input_${i}.toml with g1 = $gp"
  i=$((i+1))
done

for i in {13..16}
do
    sbatch run_dmrg.sh $i
done
