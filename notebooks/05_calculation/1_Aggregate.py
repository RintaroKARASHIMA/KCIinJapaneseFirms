#! (root)/notebooks/3_calculate/1_Aggregate.py python3
# -*- coding: utf-8 -*-
#%%
# %load 0_LoadLibrary.py
## Import Library
### Processing Data
import pandas as pd
import numpy as np
import sys

### Visualization
from IPython.display import display

### Third Party
from ecomplexity import ecomplexity

### Set Visualization Parameters
pd.options.display.float_format = "{:.3f}".format

## Import Original Modules
sys.path.append("../../src")
import initial_condition
from process import weight
from visualize import rank as vr

### 
#%%
