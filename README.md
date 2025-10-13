<h1>KCIinJapaneseFirms</h1>

This project is undergoing. Thus this README is also incompleted.
<br/>
日本語版（未完成）は[こちら](https://github.com/RintaroKARASHIMA/KCIinJapaneseFirms/blob/master/README_ja.md)から

<h2>About (TL;DR)</h2>

This project is a record for the research which applies [KCI (Knowledge Complexity Index)](https://www.tandfonline.com/doi/full/10.1080/00130095.2016.1205947) to Japanese corporations.
<br/>
**Rintaro Karashima** is the only person who is responsible for this project.

## Directory Structure

The directory of this project is mainly composed of **three private** and **three public** folders and the corresponding files as following.

<details><summary><h3>Entire Directory Structure</h3></summary>

<pre>
.
├───archive (private)
│   ├───JPWAP_files
│   ├───JPWIP_files
│   └───JPWRP_files
├───data (private)
│   ├───description
│   ├───interim
│   │   ├───external
│   │   └───internal
│   │       ├───bulk
│   │       ├───fixed
│   │       ├───jp_filter
│   │       ├───jp_filtered
│   │       ├───merged
│   │       ├───reg_num_filter
│   │       ├───reg_num_filtered
│   │       ├───stack
│   │       └───weighted
│   ├───processed
│   │   ├───external
│   │   │   ├───abroad
│   │   │   ├───ipc
│   │   │   ├───letter
│   │   │   ├───ministry
│   │   │   ├───nistep
│   │   │   └───schmoch
│   │   └───internal
│   │       ├───05_2_1_rta
│   │       ├───05_2_2_bipartite
│   │       ├───05_2_3_corporations
│   │       ├───05_2_3_prefectures
│   │       ├───05_2_4_tech
│   │       ├───05_2_5_product_space
│   │       ├───05_2_6_projection
│   │       └───05_2_7_tech_comparison
│   └───raw
│       ├───external
│       │   ├───city
│       │   ├───eutci
│       │   ├───ipc
│       │   ├───ministry
│       │   ├───nistep
│       │   └───schmoch
│       └───internal
│           ├───bulk
│           │   ├───JPWAP
│           │   │   ├───upd_pmac_g_app_case
│           │   │   └───upd_sinseinin
│           │   ├───JPWIP
│           │   │   └───upd_dsptch_fin_ipc
│           │   └───JPWRP
│           │       ├───upd_mgt_info_p
│           │       └───upd_right_person_art_p
│           ├───bulk_html
│           │   ├───JPWAP_files
│           │   ├───JPWIP_files
│           │   └───JPWRP_files
│           ├───bulk_targz
│           │   ├───JPWAP
│           │   ├───JPWIP
│           │   └───JPWRP
│           └───stack
│               ├───JPAP
│               ├───JPIP
│               └───JPRP
├───notebooks (public)
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
│   └───11_tech_r_national_complarison
├───outputs (public)
│   ├───external_figures
│   │   ├───gephi_product_space
│   │   ├───gephi_projection
│   │   └───tablaeu
│   ├───internal_figures
│   │   ├───07_tech_long_ctci_vs_rtci
│   │   ├───07_tech_long_finer_vs_coarse
│   │   ├───07_tech_long_fiver_vs_coarse_residual
│   │   ├───07_tech_long_tci_vs_avediversity
│   │   ├───07_tech_long_ubiquity_vs_avediversity
│   │   ├───07_tech_long_ubiquity_vs_tci
│   │   ├───09_tech_sep_ranking
│   │   └───10_network_long_adjacency_matrix
│   └───internal_tables
├───src (public)
│   ├───calculation
│   │   └───__pycache__
│   ├───cleansing_filtering
│   │   └───__pycache__
│   ├───initialize
│   │   └───__pycache__
│   ├───visualize
│   │   └───__pycache__
│   └───__pycache__
└───tests (private)
</pre>

</details>

Here, the role and the rule of each folder are defined as following.<br/>
Below, `Memory-efficient Processing` refers to processes prioritized for memory efficiency over readability, such as recursive queries/functions in pandas. <br/>
Conversely, `Readable Processing` refers to processes where readability is prioritized over memory efficiency, exemplified by the `query` method and `method chaining` in pandas. <br/>
Processes prone to unnecessary copies, low readability `Boolean Indexing`, and discouraged practices like `chained indexing/assignment` or `hidden chaining`, and `mutating` should be avoided whenever possible.


### Data

The data is organized into three layers: the `Data Lake Layer`, the `Data Cleansing Layer`, and the `Data Mart Layer`.  
- The `Data Lake Layer` corresponds to the `raw` folder inside the `data` directory, containing unprocessed and unfiltered raw data.  
- The `Data Cleansing Layer` is represented by the `interim` folder within the `data` directory, which houses the data undergoing cleansing and filtering processes.  
- The `Data Mart Layer` corresponds to the `processed` folder in the `data` directory, where the data has been fully processed and is ready for further analysis.

Within each layer, internal folders store proprietary data (or raw patent data), and external folders store data from external sources.


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
