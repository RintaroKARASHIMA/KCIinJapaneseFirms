#! (root)/notebooks/02_merge_raw/1_MeltBulk.py python3
# -*- coding: utf-8 -*-

# %%
# %load 0_LoadLibraries.py
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
global DATA_DIR
DATA_DIR = "../../data/raw/internal/"

# %%
# Get the Data
def get_data(master_name, coding="utf-8"):
    html_file = f"{DATA_DIR}bulk_html/{master_name}.html"
    output_dir = f"{DATA_DIR}bulk_targz/{master_name}/"

    with open(html_file, "r", encoding=coding) as f:
        html_text = f.read()
        f.close()

    soup = BeautifulSoup(html_text, "html.parser")
    href_list = list(
        set(
            href
            for href in [a.get("href") for a in soup.select("div a")]
            if master_name in href
        )
    )
    for href in href_list:
        urllib.request.urlretrieve(href, filename=output_dir + href.split("/")[-1])
        # print(f'Downloaded: {href.split("/")[-1]}')
        time.sleep(3)
    print(master_name, "got!")

# %%
# Melt the Data
def extract_data(master_name):
    needed_file_dict = {
        "JPWRP": [  # 登録マスタ
            "upd_mgt_info_p.tsv",
            "upd_right_person_art_p.tsv",
        ],
        "JPWIP": [  # IPCマスタ
            "upd_dsptch_fin_ipc.tsv",
        ],
        "JPWAP": [  # 出願マスタ
            "upd_pmac_g_app_case.tsv", 
            "upd_sinseinin.tsv"],
    }
    targz_list = glob(f"{DATA_DIR}bulk_targz/{master_name}/*")
    for targz in targz_list:
        with tarfile.open(name=targz, mode="r:gz") as tf:
            member_list = tf.getmembers()
            file_date = targz.split(f"{master_name}_")[-1].split(".")[0]

            for member in member_list:
                file_name = member.name.split("/")[-1]
                if file_name in needed_file_dict[master_name]:
                    needed_dir = file_name.split(".")[0]
                    member.name = (
                        f"{DATA_DIR}bulk/{master_name}/{needed_dir}/{file_date}.tsv"
                    )

                    # 誰に何と言われようと、バージョンに依存するメソッドは嫌いです。
                    tf.extract(member, path=".")
                    # print(f'Extracted: {folder_name}/{file_name}')
            tf.close()
    print(master_name, "melted!")


# %%
# Main
for mn in ["JPWRP", "JPWIP", "JPWAP"]:
    get_data(mn)
    extract_data(mn)
print("completed!")
