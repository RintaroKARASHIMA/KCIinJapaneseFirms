#! (root)/src/initialize/load_libraries.py python3
# -*- coding: utf-8 -*-

## Import Library
### Systems
from importlib import reload
from glob import glob
import os
import sys

### Processing Data
import numpy as np
import pandas as pd

### Visualization
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick

### Third Party
from ecomplexity import ecomplexity, proximity

## Import Original Modules
sys.path.append('../../src')
from calculation import weight
from visualize import rank as vr

### Set Visualization Parameters
pd.options.display.float_format = '{:.3f}'.format

plt.rcParams['axes.axisbelow'] = True
plt.rcParams['font.family'] = 'Meiryo'
plt.rcParams['font.size'] = 25
plt.rcParams.update({'figure.autolayout': True})
