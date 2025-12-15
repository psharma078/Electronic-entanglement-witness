#!/bin/sh
#SBATCH -J dmrg_gs
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -e job.err
#SBATCH -o job.out
#SBATCH --mail-type=all
##SBATCH --mail-user=PSHAR50@emory.edu
#SBATCH -A DMR21001
#SBATCH -V
cd $SLURM_SUBMIT_DIR

rm job.*
##rm *.txt

export OMP_PROC_BIND=true
export OMP_PLACES=threads
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/00434/eijkhout/arpack/installation-3.7.0-intel/lib64
##export OMP_NUM_THREADS=28
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export JULIA_NUM_THREADS=8

julia main_groundState.jl > out_gs_L40_Nup20_Ndn20_U0_V0.txt 2>&1
wait
