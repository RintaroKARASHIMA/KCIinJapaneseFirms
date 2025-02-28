#%%
%run 0_LoadLibraries.py
## Import Library
### Processing Data

### Visualization

### Third Party

### Set Visualization Parameters

## Import Original Modules

## Arrange variables
ipc_digit = 3
data_dir = '../../data/interim/internal/filtered_before_agg/'
filter_dir = '../../data/interim/internal/filter_after_agg/'
output_dir = '../../data/interim/internal/filtered_after_agg/'
all_df = pd.read_csv(f'{data_dir}japan.csv', 
                     encoding='utf-8', 
                     sep=',')

#%%
all_df
# %%
