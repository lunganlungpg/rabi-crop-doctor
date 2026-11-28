"""
rabi_crop_doctor
================

Free, offline-capable Python toolkit for Indian Rabi crop foliar disease
classification (ML + ANN models) and plot-level disease incidence tracking.

Public API
----------
- predict_leaf_image(image_path, crop_filter=None)
- extract_features_and_diagnostics(image_input)
- FEATURE_NAMES
- compute_field_survey_metrics(observations, max_scale=5)
- severity_to_grade(severity_pct)
- summarize_india_rabi_incidence(csv_path=None)
- train_pipeline(image_dir=None)
- CLASSES
"""

from __future__ import annotations

from .disease_incidence import (
    compute_field_survey_metrics,
    severity_to_grade,
    summarize_india_rabi_incidence,
)
from .feature_extractor import FEATURE_NAMES, extract_features_and_diagnostics
from .generate_rabi_dataset import CLASSES
from .predict_image import predict_leaf_image
from .train_and_evaluate import train_pipeline

__version__ = "1.0.2"

__all__ = [
    "__version__",
    "CLASSES",
    "FEATURE_NAMES",
    "compute_field_survey_metrics",
    "extract_features_and_diagnostics",
    "predict_leaf_image",
    "severity_to_grade",
    "summarize_india_rabi_incidence",
    "train_pipeline",
]
