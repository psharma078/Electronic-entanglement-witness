#!/bin/bash
#SBATCH -p cpuq
#SBATCH -J 61-a-LUMO
#SBATCH -o output.slurm
#SBATCH -t 520:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100GB
#SBATCH -e error.slurm

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export JULIA_NUM_THREADS=8

julia main_groundState.jl input_$1.toml > out_L80_Nup32_Ndn32_input_$1.txt
