#!/bin/bash

for i in {1..16}
do
    sbatch frontera_entwitness_julia.sh $i
done
