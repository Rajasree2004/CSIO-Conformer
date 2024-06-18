#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -N gpu_task
#PBS -l select=1:ngpus=4:mpiprocs=36
#PBS -q gpu
#PBS -V

# Go to the directory from which you submitted the job
cd $PBS_O_WORKDIR

# Setup the compiler environment for Intel OneAPI 2023
module use /cm/sharked/modulefiles/oneapi-2023
module load debugger/latest
module load dpl/latest
module load compiler/latest
module load mpi/latest
module load mkl/latest

# List all the modules which are currently loaded
module list
module load /apps/modules/python-3.10.9
python3 --version
python3 -c"print('hello gpu')"

# /home/srikanth/miniconda3/bin/conda init
source ~/.bashrc
# Activate environment with full path (if necessary)
conda activate graspenv
python --version
export OMP_NUM_THREADS=1
python -c "import torch; print(torch.__version__)"

time mpirun -bootstrap=ssh -np 1 python conformerwwchitgprec.py
