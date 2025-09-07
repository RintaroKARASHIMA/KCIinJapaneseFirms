#! (root)/notebooks/05_calculation/5_MergeCorporationRegion.py python3
# -*- coding: utf-8 -*-

#%%
## Load Global Settings
%run ../../src/initialize/load_libraries.py
%run ../../src/initialize/initial_conditions.py

## Load Local Settings
%run 0_LoadLibraries.py

# %%
# Import Original Modules
from calculation import method_of_reflections as mor
# reload(mor)

# %%
# Initialize Global Variables
in_dir = f'{IN_IN_DIR}filtered_after_agg/'
out_dir = f'{PRO_IN_DIR}'
ex_dir = f'{PRO_EX_DIR}'

## Check the condition
print(CONDITION)
# %%
