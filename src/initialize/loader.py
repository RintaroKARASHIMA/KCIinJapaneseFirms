#!/bin/csh
#$ -cwd
#$ -V -S /bin/bash
#$ -N RK_data_processing_loader
#$ -q all.q@Cheryl
#$ -pe smp 1
#$ -o RK_data_processing_loader.txt
#$ -e RK_data_processing_loader_error

#%%
import numpy as np
import pandera as pa

#%%
    