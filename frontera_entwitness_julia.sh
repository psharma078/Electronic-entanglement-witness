#!/bin/sh
#SBATCH -J ent_V
#SBATCH -p normal
#SBATCH -N 2
#SBATCH -n 110
#SBATCH -t 48:00:00
#SBATCH -e job.err
#SBATCH -o job.out
#SBATCH --mail-type=all
##SBATCH --mail-user=PSHAR50@emory.edu
#SBATCH -A DMR21001
#SBATCH -V
cd $SLURM_SUBMIT_DIR

rm job.*

export OMP_PROC_BIND=true
export OMP_PLACES=threads
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/00434/eijkhout/arpack/installation-3.7.0-intel/lib64
##export OMP_NUM_THREADS=28
#export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=1
#export JULIA_NUM_THREADS=1

mpiexec -n 110 julia main_entanglement.jl > out_ent_L40_Nup20_Ndn20_U0_V0.txt 2>&1
wait
