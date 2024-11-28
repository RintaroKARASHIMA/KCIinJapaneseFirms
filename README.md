<h1>KCIinJapaneseFirms</h1>

This project is undergoing. Thus this README is also incompleted.
<br/>日本語版（未完成）は[こちら](https://github.com/RintaroKARASHIMA/KCIinJapaneseFirms/blob/master/README_jp.md)から

<h2>About (TL;DR)</h2>

This project is a log for the research which applies [KCI (Knowledge Complexity Index)](https://) to Japanese corporations.
<br/>**Rintaro Karashima** is the only person who conducts the research and is responsible for this repository.

- The directory of this projects mainly consists of **four private** and **two public** folders as following.
  <br/>[1] archive (private):
  <br/>[2] data (private):
  <br/>[3] **notebooks** (public):
  <br/>[4] outputs (private):
  <br/>[5] **src** (public):
  <br/>[6] tests (private):

<h2>Directory Structure</h2>

<details><summary><h3>Entire Directory Structure</h3></summary>
<br/>```
root/
├───archive
├───data
│   ├───interim
│   │   ├───external
│   │   └───internal
│   │       ├───bulk
│   │       ├───filtered_after_agg
│   │       ├───filtered_before_agg
│   │       ├───filter_after_agg
│   │       ├───filter_before_agg
│   │       ├───fixed
│   │       ├───merged
│   │       └───stack
│   ├───raw
│   │   ├───external
│   │   │   ├───city
│   │   │   ├───eutci
│   │   │   ├───ipc
│   │   │   ├───ministry
│   │   │   ├───nistep
│   │   │   └───schmoch
│   │   └───internal
│   │       ├───bulk
│   │       │   ├───JPWAP
│   │       │   │   ├───upd_pmac_g_app_case
│   │       │   │   └───upd_sinseinin
│   │       │   ├───JPWIP
│   │       │   │   └───upd_dsptch_fin_ipc
│   │       │   └───JPWRP
│   │       │       ├───upd_mgt_info_p
│   │       │       └───upd_right_person_art_p
│   │       ├───bulk_html
│   │       │   ├───JPWAP_files
│   │       │   ├───JPWIP_files
│   │       │   └───JPWRP_files
│   │       ├───bulk_targz
│   │       │   ├───JPWAP
│   │       │   ├───JPWIP
│   │       │   └───JPWRP
│   │       └───stack
│   │           ├───JPAP
│   │           │   ├───upd_pmac_g_app_case
│   │           │   └───upd_sinseinin
│   │           ├───JPIP
│   │           │   └───upd_dsptch_fin_ipc
│   │           └───JPRP
│   │               ├───upd_mgt_info_p
│   │               └───upd_right_person_art_p
│   └───processed
│       ├───external
│       │   ├───abroad
│       │   ├───ipc
│       │   ├───letter
│       │   ├───ministry
│       │   ├───nistep
│       │   └───schmoch
│       └───internal
│           ├───corporations
│           ├───graph
│           ├───rta
│           └───tech
├───notebooks
│   ├───00_template
│   ├───01_merging_raw
│   ├───02_external_preparation
│   ├───03_cleansing_filtering
│   ├───04_observation
│   ├───05_calculation
│   ├───06_producer_long
│   ├───07_tech_long
│   ├───08_producer_sep
│   ├───09_tech_sep
│   ├───10_network_long
│   └───11_tech_r_national_comparison
├───output
│   ├───charts
│   │   ├───adjacency_matrix
│   │   ├───tech_long_ctci_vs_rtci
│   │   ├───tech_long_finer_vs_coarse
│   │   ├───tech_long_finer_vs_coarse_residual
│   │   ├───tech_long_tci_vs_avediversity
│   │   ├───tech_long_ubiquity_vs_avediversity
│   │   ├───tech_long_ubiquity_vs_tci
│   │   └───tech_sep_ranking
│   ├───networks
│   │   ├───product_space
│   │   └───projection
│   └───tables
├───src
│   ├───analysis
│   ├───cleansing_filtering
│   ├───process
│   └───visualize
└───tests
```
</details>


<h3>Data</h3>

- **datasets/education_conversations.csv**: 教育会話データセット。各行は個別の会話を表します。
- **datasets/annotations.json**: 会話データの注釈ファイル。

## データ前処理

- **scripts/preprocess.py**: データの前処理スクリプト。
- **scripts/analyze.py**: データ分析スクリプト。

## 計算と分析

- **scripts/preprocess.py**: データの前処理スクリプト。
- **scripts/analyze.py**: データ分析スクリプト。


[^1]: Here is My reference