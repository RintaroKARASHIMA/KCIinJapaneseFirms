# KCIinJapaneseFirms

This project is undergoing. Thus this README is also incomplete.
`<br>`
日本語版は[こちら](https://github.com/RintaroKARASHIMA/KCIinJapaneseFirms/blob/master/README_en.md)から（未完成）

## 概要

KCIinJapaneseFirmsは、日本の法人が持つ複雑さと技術の複雑さを定量化することで評価を試みた

Rintaro Karashima is the only person who is responsible for this repository.

- The directory of this projects is mainly composed of **three** parts as following.
  <br/>
  1. **data**:
  <br/>
  2. **notebooks**:
  <br/>
  3. **src**:

<h2>ディレクトリ構成</h2>
本プロジェクトディレクトリ構成の全体像と、それを構成する各フォルダ/ファイルについて、定義を示す。
<br/>以下、`メモリ効率の高い処理`は、文字通り可読性よりもメモリ効率を優先した処理を指し、例えばpandasでは再帰代入（recursive query/function）が挙げられる。
<br/>同様に、`可読性の高い処理`は、メモリ効率よりも可読性を優先した処理を指し、例えばpandasではquery methodとmethod chainingが挙げられる。
<br/>なお、どちらの処理においても余計なコピーが発生しやすく可読性の低いBoolean Indexingや推奨されないchaining(chained indexing/assignment、あるいはhidden chaining)、mutatingといった処理は極力使わない。


<details><summary><h3>Entire Directory Structure</h3></summary>

<pre>
.
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
│   ├───01_external_preparation
│   ├───02_merging_raw
│   ├───03_cleansing_filtering
│   ├───04_observation
│   ├───05_calculation
│   ├───06_producer_long
│   ├───07_tech_long
│   ├───08_producer_sep
│   ├───09_tech_sep
│   ├───10_network_long
│   └───11_tech_r_national_comparison
├───outputs
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
</pre>

</details>

<h3>data</h3>
データレイク層、データクレンジング層、データマート層に分け、各層の中で特許データを格納するinternalと外部データを格納するexternalを分けている。

- **interim**: データクレンジング層。加工に係る中間生産のデータ。
- **raw**: データレイク層。加工前のデータ（note:tar.gzとそれを解凍したファイルを含む。）。
- **processed**: データマート層。分析に用いる加工後のデータ。

<h3>notebooks</h3>
dataの3つの層に対応する次のような11のフォルダが存在する。ここで、各フォルダは半角数字2桁でindexingされ、`00_template`のようにアンダースコアを挟んでスネークケースで一意に命名される。
<br/>各フォルダを構成する最小構成要素.pyファイルは、それぞれ半角数字1桁でindexingされ、0_LoadLibraries.py`のようにキャメルケースで命名される。
<br/>なお、`00_template`には、そのフォルダ内に共通して追加のimportが必要なlibrariesをloadする`0_LoadLibraries.py`と、.pyファイルのbaseとなる`1_Sample`が格納され、全てのフォルダはその使用に則り作成される。

<h4>外部データ</h4>
特許分類のマスタのマージ等、特に計算量が多い処理を含むものについてはメモリ効率の高い処理、それ以外のものについては可読性の高い処理を行う。

- notebooks/**01_external_preparation**: <br/>
  外部から読み込んだ生データを分析可能な状態にする。このフォルダに限り、データレイク層からデータマート層まで一気通貫した処理を行う。indexingはなく、スネークスケールで命名される。

<h4>データレイク層</h4>

- notebooks/**02_merging_raw**:

<h4>データクレンジング層</h4>

- notebooks/**03_cleansing_filtering**:
- notebooks/**04_observation**:

<h4>データマート層</h4>

- notebooks/**05_calculation**: クレンジングを施したデータをもとに各指標を計算する。ここで作成されるデータは他フォルダ行う分析の基盤となる。
- notebooks/**06_producer_long**: `05_calculation`で得た生産者(corporations or regions)側について、指標間の相関分析など長期間分析を行う。
- notebooks/**07_tech_long**: `06_producer_long`と同様に、技術分野側について指標間の相関分析など長期間分析を行う。
- notebooks/**08_producer_sep**: `06_producer_long`で行った長期間分析について、期間を区切って行い、変化や推移を見る。
- notebooks/**09_tech_sep**: `07_tech_long`で行った長期間分析を期間を区切って行い、変化や推移を見る。
- notebooks/**10_network_long**: 長期間分析で対象となったネットワークの構造について解析する。
- notebooks/**11_tech_r_national_comparison**: region level で得た技術分野側の指標について、国際比較を行う。

<h3>outputs</h3>
noteboooksで出力された図表ごとにフォルダが作成される。

<h3>src</h3>
notebooksにおいて、共通した変数や再利用可能な関数をモジュールとして管理する。
