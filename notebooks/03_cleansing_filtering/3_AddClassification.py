#! (root)/notebooks/03_cleansing_filtering/3_AddClassification.py python3
# -*- coding: utf-8 -*-


#%%
## Import Library
%run ../../src/initialize/load_libraries.py
%run 0_LoadLibraries.py

## Initialize Global Variables
global ex_data_dir, in_data_dir
ex_data_dir = '../../data/processed/external/'
in_data_dir = '../../data/interim/internal/filtered_before_agg/'


#%%
schmoch_df = pd.read_csv(ex_data_dir + 'schmoch/35.csv', 
                         encoding='utf-8', 
                         sep=',', 
                         dtype=str)
schmoch_df = schmoch_df.sort_values(by='IPC_code', 
                                    key=lambda s: s.str.len(), 
                                    ascending=True)
# schmoch_df['Field_en'].nunique()
ipc_to_schmoch35_dict = dict(zip(schmoch_df['IPC_code'], schmoch_df['Field_number']))
schmoch35_to_describe_dict = dict(zip(schmoch_df['IPC_code'], schmoch_df['Field_number']))
# ipc_to_schmoch35_dict

df = pd.read_csv(in_data_dir + 'filtered.csv')
df
df['schmoch35'] = ''
for ipc, schmoch35 in ipc_to_schmoch35_dict.items():
    df['schmoch35'] = np.where(df['ipc'].str.startswith(ipc), 
                                ipc_to_schmoch35_dict[ipc], 
                                df['schmoch35'])
df

df.to_csv(in_data_dir + 'addedclassification.csv', 
          encoding='utf-8', 
          index=False, 
          sep=',')
