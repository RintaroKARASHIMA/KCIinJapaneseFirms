#! (root)/notebooks/02_merge_raw/0_LoadLibraries.py python3
# -*- coding: utf-8 -*-

## Import Library
### Processing Data
import sys
from glob import glob
import pandas as pd
import numpy as np
import time
from bs4 import BeautifulSoup
import urllib.request
import tarfile

### Visualization
from IPython.display import display

### Set Visualization Parameters
pd.options.display.float_format = "{:.3f}".format

## Import Original Modules
sys.path.append("../../src")
from process import weight
from visualize import rank as vr

## Initialize Global Variables
global DATA_DIR, EXCUTE_COUNT
DATA_DIR = "../../data/original/internal/"