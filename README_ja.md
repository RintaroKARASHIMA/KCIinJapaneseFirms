# KCIinJapaneseFirms

This project is undergoing. Thus this README is also incomplete.
<br>
日本語版は[こちら](https://github.com/RintaroKARASHIMA/KCIinJapaneseFirms/blob/master/README_en.md)から（未完成）

## 概要
KCIinJapaneseFirmsは、日本の法人が持つ複雑さと技術の複雑さを定量化することで評価を試みた

Rintaro Karashima is the only person who is responsible for this repository.

- The directory of this projects is mainly consist of **three** parts as following.
<br/>1. **data**: 
<br/>2. **notebooks**: 
<br/>3. **src**: 

## ディレクトリ構造

```html
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
│   ├───original
│   │   ├───external
│   │   │   ├───city
│   │   │   ├───eutci
│   │   │   ├───ipc
│   │   │   ├───ministry
│   │   │   ├───nistep
│   │   │   └───schmoch
│   │   └───internal
│   │       ├───bulk
│   │       │   ├───20230927
│   │       │   │   ├───JPWAP
│   │       │   │   ├───JPWIP
│   │       │   │   └───JPWRP
│   │       │   ├───20231004
│   │       │   │   ├───JPWAP
│   │       │   │   ├───JPWIP
│   │       │   │   └───JPWRP
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
│   │           ├───JPIP
│   │           └───JPRP
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
│   ├───08_tech_comparative_analysis
│   ├───09_tech_comparative_analysis
│   ├───10_tech_comparative_analysis
│   └───11_network_analysis
├───output
│   ├───figures
│   ├───reports
│   └───tables
├───src
│   ├───analysis
│   ├───cleansing_filtering
│   ├───process
│   └───visualize
└───tests
```

## データの内容

- **datasets/education_conversations.csv**: 教育会話データセット。各行は個別の会話を表します。
- **datasets/annotations.json**: 会話データの注釈ファイル。

## データ前処理

- **scripts/preprocess.py**: データの前処理スクリプト。
- **scripts/analyze.py**: データ分析スクリプト。

## 計算と分析

- **scripts/preprocess.py**: データの前処理スクリプト。
- **scripts/analyze.py**: データ分析スクリプト。