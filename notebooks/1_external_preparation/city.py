#%%
## Import Libraries
import pandas as pd
import numpy as np
from glob import glob

#%%
## Load Raw Data
original_df = pd.read_csv('../Data/Original/geoshape-city-geolod.csv', 
                          encoding='utf-8', 
                          dtype=str, 
                          index_col=False)

#%%
df = original_df.copy()
df[df['body']!=df['body_variants']]
df[df['suffix'].str.len()!=2]