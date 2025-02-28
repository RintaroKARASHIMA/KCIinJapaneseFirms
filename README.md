<h1>KCIinJapaneseFirms</h1>

This project is undergoing. Thus this README is also incompleted.
<br/>
日本語版（未完成）は[こちら](https://github.com/RintaroKARASHIMA/KCIinJapaneseFirms/blob/master/README_ja.md)から

<h2>About (TL;DR)</h2>

This project is a record for the research which applies [KCI (Knowledge Complexity Index)](https://) to Japanese corporations.
<br/>
**Rintaro Karashima** is the only person who is responsible for this project.

- The directory of this project is mainly composed of **three private** and **three public** folders as following.
  <br/>
  (1) archive (private):
  <br/>
  (2) data (private):
  <br/>
  (3) **notebooks** (public):tree
  <br/>
  (4) **outputs** (public):
  <br/>
  (5) **src** (public):
  <br/>
  (6) tests (private):

## Directory Structure

This document defines the overall structure of the project directory and its constituent folders/files. <br/>
Below, `Memory-efficient Processing` refers to processes prioritized for memory efficiency over readability, such as recursive queries/functions in pandas. <br/>
Conversely, `Readable Processing` refers to processes where readability is prioritized over memory efficiency, exemplified by the query method and method chaining in pandas. <br/>
Processes prone to unnecessary copies, low readability Boolean Indexing, and discouraged practices like chained indexing/assignment or hidden chaining, and mutating should be avoided whenever possible.

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

### Data

The data is organized into three layers: the Data Lake Layer, the Data Cleansing Layer, and the Data Mart Layer. Within each layer, internal folders store proprietary data, and external folders store data from external sources.

### Notebooks

The data layers correspond to 11 folders indexed with two-digit numbers, uniquely named in snake case with underscores, such as `00_template`. The minimal components within each folder, Python files, are indexed with single-digit numbers and named in camel case, such as `0_LoadLibraries.py`. The `00_template` folder contains a `0_LoadLibraries.py` for loading necessary libraries and a base Python file `1_Sample`, and all folders are created following this template.

#### External Data

Processes that involve heavy computations, such as merging master data of patent classifications, utilize memory-efficient processing. Other processes prioritize readable processing.

#### Data Lake Layer

- notebooks/**02_merging_raw**

#### Data Cleansing Layer

- notebooks/**03_cleansing_filtering**
- notebooks/**04_observation**

#### Data Mart Layer

- notebooks/**05_calculation**: Calculates various metrics based on the cleansed data, serving as the foundation for analysis in other folders.
- notebooks/**06_producer_long**: Conducts long-term analysis of correlations among metrics from the producers' (corporations or regions) side.
- notebooks/**07_tech_long**: Similar to `06_producer_long`, conducts long-term analysis on the technology side.
- notebooks/**08_producer_sep**: Analyzes changes and trends over separate periods based on the long-term analysis conducted in `06_producer_long`.
- notebooks/**09_tech_sep**: Analyzes changes and trends over separate periods based on the long-term analysis conducted in `07_tech_long`.
- notebooks/**10_network_long**: Analyzes the structure of networks targeted in the long-term analysis.
- notebooks/**11_tech_r_national_comparison**: Conducts international comparisons of technology-side metrics obtained at the regional level.

### Outputs

Folders are created for each diagram and table outputted from the notebooks.

### Src

Common variables and reusable functions used in notebooks are managed as modules.

[^1]: Here is My reference
