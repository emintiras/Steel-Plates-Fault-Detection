"""Configuration loader — reads config.yaml into typed dataclasses."""

from __future__ import annotations
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    path: str
    index_col: int
    fault_cols: List[str]
    test_size: float
    random_state: int


@dataclass
class FeaturesConfig:
    continuous_features: List[str]
    boxplot_features: List[str]


@dataclass
class RFConfig:
    n_estimators: int
    random_state: int
    n_jobs: int


@dataclass
class GBConfig:
    n_estimators: int
    random_state: int


@dataclass
class LRConfig:
    max_iter: int
    random_state: int


@dataclass
class CatBoostConfig:
    iterations: int
    learning_rate: float
    depth: int
    random_seed: int
    verbose: int
    loss_function: str
    eval_metric: str


@dataclass
class ModelsConfig:
    random_forest: RFConfig
    gradient_boosting: GBConfig
    logistic_regression: LRConfig
    catboost: CatBoostConfig
    cv_folds: int
    permutation_repeats: int


@dataclass
class OutputConfig:
    plot_dir: str
    results_file: str
    plot_dpi: int


@dataclass
class StyleConfig:
    dark_bg: str
    card_bg: str
    text_col: str
    grid_col: str
    accent: str
    palette: List[str]


@dataclass
class Config:
    data: DataConfig
    features: FeaturesConfig
    models: ModelsConfig
    output: OutputConfig
    style: StyleConfig


def load_config(path: str | Path = "config.yaml") -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    m = raw["models"]
    return Config(
        data=DataConfig(**raw["data"]),
        features=FeaturesConfig(**raw["features"]),
        models=ModelsConfig(
            random_forest=RFConfig(**m["random_forest"]),
            gradient_boosting=GBConfig(**m["gradient_boosting"]),
            logistic_regression=LRConfig(**m["logistic_regression"]),
            catboost=CatBoostConfig(**m["catboost"]),
            cv_folds=m["cv_folds"],
            permutation_repeats=m["permutation_repeats"],
        ),
        output=OutputConfig(**raw["output"]),
        style=StyleConfig(**raw["style"]),
    )
