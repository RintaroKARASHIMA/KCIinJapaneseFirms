from dataclasses import dataclass
from typing import Any, Dict
import yaml
from pathlib import Path

# Weight Config
@dataclass
class WeightConfig:
    ar: str
    year_style: str
    region_corporation: str
    classification: str
    in_dir: str
    out_dir: str
    out_file_name: str

def load_weight_config(path: str | Path) -> WeightConfig:
    '''YAMLからWeightConfigを読み込む'''
    with open(path, 'r', encoding='utf-8') as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    # out_file_name を展開
    raw['out_file_name'] = raw['out_file_name'].format(**raw)

    return WeightConfig(**raw)

# Reg Num Filter Config
@dataclass
class RegNumFilterConfig:
    ar: str
    year_style: str
    year_start: int
    year_end: int
    year_range: int
    
    region_corporation: str
    applicant_weight: str
    classification: str
    class_weight: str
    extract_population: str
    extract_span: str
    top_p_or_num: str
    top_p_or_num_value: int
    
    in_dir: str
    out_dir: str
    in_file_name: str
    out_file_name: str

def load_filter_config(path: Path) -> RegNumFilterConfig:
    with open(path, 'r', encoding='utf-8') as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    # 文字列展開
    raw['in_file_name'] = raw['in_file_name'].format(**raw)
    raw['out_file_name'] = raw['out_file_name'].format(**raw)

    return RegNumFilterConfig(**raw)


# Adjacency Matrix Config
@dataclass
class AdjConfig:
    ar: str
    year_style: str
    year_start: int
    year_end: int
    year_range: int
    
    region_corporation: str
    applicant_weight: str
    classification: str
    class_weight: str
    extract_population: str
    extract_span: str
    top_p_or_num: str
    top_p_or_num_value: int
    linkage: str
    threshold: int
    
    in_dir: str
    out_dir: str
    in_file_name: str
    out_file_name: str

def load_adj_config(path: Path) -> AdjConfig:
    with open(path, 'r', encoding='utf-8') as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    # 文字列展開
    raw['in_file_name'] = raw['in_file_name'].format(**raw)
    raw['out_file_name'] = raw['out_file_name'].format(**raw)

    return AdjConfig(**raw)

