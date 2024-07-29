# プロジェクト名: KCIinJapaneseFirms

This project is undergoing. So this README is also incomplete.

## 概要

KCIinJapaneseFirmsは、日本の法人が持つ複雑さと技術の複雑さを定量化することで評価を試みた

Rintaro Karashima is the only person who is responsible for this repository.

## データの内容

- **datasets/education_conversations.csv**: 教育会話データセット。各行は個別の会話を表します。
- **datasets/annotations.json**: 会話データの注釈ファイル。

## データ前処理

- **scripts/preprocess.py**: データの前処理スクリプト。
- **scripts/analyze.py**: データ分析スクリプト。

## 計算と分析

- **scripts/preprocess.py**: データの前処理スクリプト。
- **scripts/analyze.py**: データ分析スクリプト。

## ディレクトリ構造

KCIinJapaneseFirms
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
│   ├───0_merge_raw
│   ├───1_external_prearation
│   ├───2_cleansing_filtering
│   ├───3_calculate
│   ├───4_regional_focus_analysis
│   ├───5_corporate_focus_analysis
│   ├───6_knowledge_comparative_analysis
│   ├───7_tech_comparative_analysis
│   └───8_network_analysis
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
