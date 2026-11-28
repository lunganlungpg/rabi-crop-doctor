"""
Train Rabi Crop Doctor's real-image benchmark from the Kaggle 20k+ dataset.

The dataset is downloaded locally (never committed to Git), can be filtered
selectively to the Rabi crops active in the project's timeline (ROADMAP.md,
data/rabi_crops_knowledge.json, data/rabi_field_incidence_india.csv), keeps
official Train vs Validation/Test directories separate, and extracts the
project's existing 28-dimensional feature vector so results remain comparable
with the offline stack.

This creates a separate research model bundle under models/kaggle_20k/; it does
not overwrite the production 12-class Android model.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from .feature_extractor import FEATURE_NAMES, extract_features_and_diagnostics
    from .rabi_timeline import ALL_RABI_CROPS, resolve_rabi_timeline
except ImportError:
    from feature_extractor import FEATURE_NAMES, extract_features_and_diagnostics
    from rabi_timeline import ALL_RABI_CROPS, resolve_rabi_timeline


KAGGLE_DATASET = "jawadali1045/20k-multi-class-crop-disease-images"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "training_set": "train",
    "test": "test",
    "testing": "test",
    "testing_set": "test",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
}
GENERIC_DIRS = {"image", "images", "img", "imgs", "data", "dataset", "raw", "raw_images"}

# Explicit non-Rabi crops in jawadali1045/20k-multi-class-crop-disease-images so
# classes like "Yellow Rust Sugarcane" or Cotton "Wilt" are never misclassified as Rabi.
NON_RABI_CROP_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sugarcane", ("sugarcane", "redrot", "redrust")),
    ("cotton", ("cotton", "bollworm", "bollrot", "leaf_curl")),
    ("maize", ("maize", "corn", "army_worm", "common_rust", "gray_leaf_spot")),
    ("rice", ("rice", "brownspot", "tungro", "leaf_smut")),
)

RABI_CROP_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("wheat", ("wheat", "flag_smut")),
    ("mustard", ("mustard", "brassica", "sarson")),
    ("chickpea", ("chickpea", "cicer", "chana", "bengal_gram")),
    ("potato", ("potato", "solanum", "aloo")),
    ("barley", ("barley", "hordeum")),
    ("lentil", ("lentil", "masoor")),
    ("field_pea", ("field_pea", "pea", "matar")),
)

KNOWN_BARE_CROPS: set[str] = {
    *{crop for crop, _ in RABI_CROP_RULES},
    *{crop for crop, _ in NON_RABI_CROP_RULES},
}


def _slug(value: str) -> str:
    value = value.lower().replace("___", " ").replace("__", " ")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def _matches_keyword(slug: str, tokens: set[str], keyword: str) -> bool:
    return keyword in tokens or f"_{keyword}_" in f"_{slug}_"


def _infer_crop(label: str) -> str:
    """Infer standardized crop identifier from a normalized class folder slug."""
    slug = _slug(label)
    tokens = set(slug.split("_"))

    for crop, keywords in NON_RABI_CROP_RULES:
        if any(_matches_keyword(slug, tokens, kw) for kw in keywords):
            return crop

    for crop, keywords in RABI_CROP_RULES:
        if any(_matches_keyword(slug, tokens, kw) for kw in keywords):
            return crop

    # Generic "wilt" in this dataset represents Cotton FOV4 wilt (never Chickpea).
    if slug == "wilt":
        return "cotton"

    return slug.split("_", 1)[0]


def _canonical_label(raw_label: str) -> str:
    """Normalize dataset-specific folder slugs into consistent <crop>_<condition> labels."""
    slug = _slug(raw_label)
    if slug == "flag_smut":
        return "wheat_flag_smut"
    if slug.startswith("healthy_"):
        crop = _infer_crop(slug)
        if crop in KNOWN_BARE_CROPS:
            return f"{crop}_healthy"
    return slug


def _record_from_path(path: Path, root: Path) -> dict[str, str] | None:
    parts = path.relative_to(root).parts
    dirs = list(parts[:-1])
    if not dirs:
        return None

    split_index = next(
        (i for i, part in enumerate(dirs) if part.lower() in SPLIT_ALIASES),
        None,
    )
    if split_index is None:
        split = "unsplit"
        class_parts = dirs
    else:
        split = SPLIT_ALIASES[dirs[split_index].lower()]
        class_parts = dirs[split_index + 1 :]

    class_parts = [
        p for p in class_parts
        if p.lower() not in GENERIC_DIRS and not p.startswith(".")
    ]
    if not class_parts:
        return None

    # Handle nested duplicates such as Train/Healthy Wheat/Wheat___Healthy/*.jpg
    # while keeping Train/<crop>/<disease>/*.jpg hierarchies intact.
    if len(class_parts) > 1 and _slug(class_parts[0]) not in KNOWN_BARE_CROPS:
        selected_parts = [class_parts[0]]
    else:
        selected_parts = class_parts

    label = _canonical_label("_".join(selected_parts))
    if not label:
        return None
    crop = _infer_crop(label)
    return {"path": str(path), "label": label, "crop": crop, "split": split}


def _collect_records(dataset_dir: Path, crops: list[str] | None) -> list[dict[str, str]]:
    crop_set = {_slug(c) for c in crops} if crops else None
    records: list[dict[str, str]] = []
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        record = _record_from_path(path, dataset_dir)
        if record is None:
            continue
        if crop_set and record["crop"] not in crop_set:
            continue
        records.append(record)

    if not records:
        raise RuntimeError(
            f"No labelled images found under {dataset_dir}"
            + (f" matching crops {sorted(crop_set)}." if crop_set else ".")
            + " Expected folders such as Train/<class_name>/*.jpg and Validation/<class_name>/*.jpg."
        )
    return records


def _deduplicate_records(
    records: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Remove byte-identical duplicates and prevent exact train/test leakage."""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        digest = hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
        groups[digest].append(record)

    kept: list[dict[str, str]] = []
    duplicate_count = 0
    conflicting_hashes = 0
    for group in groups.values():
        labels = {item["label"] for item in group}
        if len(labels) > 1:
            conflicting_hashes += 1
            duplicate_count += len(group)
            continue
        # Prefer the held-out evaluation copy so identical training duplicates are dropped.
        chosen = next(
            (x for x in group if x["split"] == "test"),
            next((x for x in group if x["split"] == "validation"), group[0]),
        )
        kept.append(chosen)
        duplicate_count += len(group) - 1

    return kept, {
        "duplicate_images_removed": duplicate_count,
        "conflicting_label_hash_groups_removed": conflicting_hashes,
    }


class _HttpRangeFile(io.RawIOBase):
    """Seekable HTTP Range reader with a bounded sliding read-ahead buffer for ZIP extraction."""

    def __init__(
        self,
        url_provider: Any,
        url: str,
        size: int,
        chunk_bytes: int = 8 * 1024 * 1024,
        max_retries: int = 4,
    ) -> None:
        import requests

        self._requests = requests
        self._session = requests.Session()
        self._url_provider = url_provider
        self._url = url
        self.size = size
        self.pos = 0
        self._chunk_bytes = chunk_bytes
        self._max_retries = max_retries
        self._buf = b""
        self._buf_start = 0
        self._buf_end = 0
        self._span_limit: int | None = None
        self.bytes_fetched = 0
        self.http_requests = 0

    def set_span_limit(self, limit: int | None) -> None:
        """Bound read-ahead fetches to the end of the active contiguous ZIP span."""
        self._span_limit = limit

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        self.pos = max(0, min(self.pos, self.size))
        return self.pos

    def tell(self) -> int:
        return self.pos

    def _fetch_range(self, start: int, end: int) -> bytes:
        for attempt in range(self._max_retries):
            try:
                resp = self._session.get(
                    self._url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=60,
                )
                if resp.status_code in (401, 403) and self._url_provider is not None and attempt == 0:
                    self._url = self._url_provider()
                    continue
                resp.raise_for_status()
                data = resp.content
                self.bytes_fetched += len(data)
                self.http_requests += 1
                return data
            except self._requests.RequestException:
                if attempt + 1 >= self._max_retries:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError("Exhausted HTTP range retries")

    def readinto(self, b: Any) -> int:
        if self.pos >= self.size:
            return 0
        want = len(b)
        if want == 0:
            return 0

        if not (self._buf_start <= self.pos < self._buf_end):
            cap = self._span_limit if (self._span_limit and self._span_limit > self.pos) else self.size
            fetch_end = min(cap - 1, self.pos + max(want, self._chunk_bytes) - 1)
            self._buf = self._fetch_range(self.pos, fetch_end)
            self._buf_start = self.pos
            self._buf_end = self._buf_start + len(self._buf)

        avail_off = self.pos - self._buf_start
        n = min(want, len(self._buf) - avail_off)
        if n <= 0:
            return 0
        b[:n] = self._buf[avail_off : avail_off + n]
        self.pos += n
        return n

    def close(self) -> None:
        self._session.close()
        super().close()


def _build_kagglesdk_client() -> Any:
    from kagglesdk import KaggleClient

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return KaggleClient(username=username, password=key)
    return KaggleClient()


def _download_dataset_selective(
    destination: Path,
    crops: list[str],
    dataset_slug: str = KAGGLE_DATASET,
) -> bool:
    """
    Download only ZIP entries matching `crops` using HTTP Range requests.

    Returns True when selective extraction succeeds, or False if the remote server
    does not support byte-range requests (caller falls back to full CLI download).
    """
    try:
        from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest
    except ImportError:
        return False

    owner_slug, ds_name = dataset_slug.split("/", 1)
    crop_set = {_slug(c) for c in crops}

    def _get_signed_url() -> tuple[str, int, bool]:
        with _build_kagglesdk_client() as client:
            req = ApiDownloadDatasetRequest()
            req.owner_slug = owner_slug
            req.dataset_slug = ds_name
            resp = client.datasets.dataset_api_client.download_dataset(req)
            resp.close()
            total = int(resp.headers.get("Content-Length", "0"))
            accept_ranges = "bytes" in resp.headers.get("Accept-Ranges", "").lower()
            return resp.url, total, accept_ranges

    url, total_size, accept_ranges = _get_signed_url()
    if not accept_ranges or total_size <= 0:
        return False

    print(
        f"Inspecting Kaggle archive index ({total_size / 1e9:.2f} GB) for crops: {sorted(crop_set)}",
        flush=True,
    )
    range_reader = _HttpRangeFile(
        url_provider=lambda: _get_signed_url()[0],
        url=url,
        size=total_size,
        chunk_bytes=2 * 1024 * 1024,
    )
    try:
        buffered = io.BufferedReader(range_reader, buffer_size=2 * 1024 * 1024)
        zf = zipfile.ZipFile(buffered)
        dest_root = destination.resolve()
        dest_root.mkdir(parents=True, exist_ok=True)

        selected: list[tuple[zipfile.ZipInfo, Path]] = []
        skipped_cached = 0
        for info in zf.infolist():
            if info.is_dir():
                continue
            rel = Path(info.filename)
            if rel.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            record = _record_from_path(rel, Path("."))
            if record is None or record["crop"] not in crop_set:
                continue
            out_path = (dest_root / rel).resolve()
            if not out_path.is_relative_to(dest_root):
                continue
            if out_path.exists() and out_path.stat().st_size == info.file_size:
                skipped_cached += 1
                continue
            selected.append((info, out_path))

        if not selected and skipped_cached == 0:
            raise RuntimeError(
                f"No files in {dataset_slug} matched selected Rabi crops: {sorted(crop_set)}"
            )

        selected.sort(key=lambda item: item[0].header_offset)
        print(
            f"Selective download plan: {len(selected):,} files to fetch "
            f"({skipped_cached:,} already cached) across {sorted(crop_set)}",
            flush=True,
        )

        if selected:
            # Coalesce adjacent entries (<= 512 KB gap) into bounded spans.
            spans: list[tuple[int, int]] = []
            for info, _ in selected:
                est_start = info.header_offset
                est_end = info.header_offset + 512 + len(info.filename.encode("utf-8")) + info.compress_size
                if not spans or est_start - spans[-1][1] > 524_288:
                    spans.append((est_start, min(total_size, est_end)))
                else:
                    spans[-1] = (spans[-1][0], min(total_size, max(spans[-1][1], est_end)))

            # Reconfigure range reader with larger streaming buffer for file extraction.
            range_reader._chunk_bytes = 8 * 1024 * 1024
            span_idx = 0
            for idx, (info, out_path) in enumerate(selected, start=1):
                while span_idx < len(spans) and info.header_offset > spans[span_idx][1]:
                    span_idx += 1
                if span_idx < len(spans):
                    range_reader.set_span_limit(spans[span_idx][1])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, out_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                if idx % 500 == 0 or idx == len(selected):
                    print(
                        f"Extracted {idx:,}/{len(selected):,} selective files "
                        f"({range_reader.bytes_fetched / 1e6:.1f} MB fetched in {range_reader.http_requests} HTTP requests)",
                        flush=True,
                    )
        return True
    finally:
        range_reader.close()


def _download_dataset(
    destination: Path,
    crops: list[str] | None = None,
    full_download: bool = False,
) -> str:
    destination.mkdir(parents=True, exist_ok=True)
    if crops and not full_download:
        try:
            if _download_dataset_selective(destination, crops=crops, dataset_slug=KAGGLE_DATASET):
                return "selective_http_range"
        except Exception as exc:
            print(
                f"Selective HTTP-range download fell back to full Kaggle CLI ({exc}).",
                flush=True,
            )

    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI is not installed. Run: pip install kaggle six"
        )
    has_legacy_credentials = bool(
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    )
    has_api_token = bool(os.environ.get("KAGGLE_API_TOKEN"))
    config_path = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    access_token_path = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "access_token"
    if not (has_api_token or has_legacy_credentials or config_path.exists() or access_token_path.exists()):
        print(
            "No explicit Kaggle credentials detected; attempting anonymous public dataset download.",
            flush=True,
        )
    command = [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", str(destination),
        "--unzip",
    ]
    print("Downloading full Kaggle dataset:", KAGGLE_DATASET, flush=True)
    subprocess.run(command, check=True)
    return "full_kaggle_cli"


def _make_split(
    records: list[dict[str, str]],
    test_size: float,
    random_state: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], str, list[str]]:
    official_train = [r for r in records if r["split"] == "train"]
    official_val = [r for r in records if r["split"] == "validation"]
    official_test = [r for r in records if r["split"] == "test"]

    if official_train and official_test:
        train_pool = official_train + official_val
        train_labels = {r["label"] for r in train_pool}
        test_labels = {r["label"] for r in official_test}
        common = sorted(train_labels & test_labels)
        excluded = sorted((train_labels | test_labels) - set(common))
        train = [r for r in train_pool if r["label"] in common]
        test = [r for r in official_test if r["label"] in common]
        source = "official train/validation and test folders; test held out unchanged"
    elif official_train and official_val:
        train_labels = {r["label"] for r in official_train}
        val_labels = {r["label"] for r in official_val}
        common = sorted(train_labels & val_labels)
        excluded = sorted((train_labels | val_labels) - set(common))
        train = [r for r in official_train if r["label"] in common]
        test = [r for r in official_val if r["label"] in common]
        source = "official Train and Validation folders; Validation held out unchanged as evaluation split"
    else:
        all_records = [r for r in records if r["split"] == "unsplit"] or list(records)
        if not all_records:
            raise RuntimeError(
                "Could not construct a train/test split. Check the dataset folder structure."
            )
        counts = Counter(r["label"] for r in all_records)
        eligible = {label for label, n in counts.items() if n >= 2}
        excluded = sorted(set(counts) - eligible)
        all_records = [r for r in all_records if r["label"] in eligible]
        train, test = train_test_split(
            all_records,
            test_size=test_size,
            random_state=random_state,
            stratify=[r["label"] for r in all_records],
        )
        source = f"stratified {1-test_size:.0%}/{test_size:.0%} split (no official evaluation folder)"

    train_counts = Counter(r["label"] for r in train)
    test_counts = Counter(r["label"] for r in test)
    valid = sorted(
        label for label in set(train_counts) & set(test_counts)
        if train_counts[label] >= 2 and test_counts[label] >= 1
    )
    dropped = sorted((set(train_counts) | set(test_counts)) - set(valid))
    excluded = sorted(set(excluded) | set(dropped))
    train = [r for r in train if r["label"] in valid]
    test = [r for r in test if r["label"] in valid]

    if len(valid) < 2:
        raise RuntimeError(
            "Fewer than two classes have usable examples in both train and test. "
            "Inspect the folder names and dataset structure before training."
        )
    return train, test, source, excluded


def _extract_matrix(
    records: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    feature_rows: list[list[float]] = []
    labels: list[str] = []
    skipped = 0
    for i, record in enumerate(records, start=1):
        try:
            # Verify that Pillow can decode the image before feature extraction.
            with Image.open(record["path"]) as im:
                im.verify()
            result = extract_features_and_diagnostics(record["path"])
            feature_rows.append([float(x) for x in result["features"]])
            labels.append(record["label"])
        except Exception as exc:
            skipped += 1
            print(f"[skip] {record['path']}: {exc}", flush=True)
        if i % 500 == 0:
            print(f"Extracted features from {i:,}/{len(records):,} images", flush=True)

    if not feature_rows:
        raise RuntimeError("All selected images failed decoding or feature extraction.")
    return np.asarray(feature_rows, dtype=np.float64), np.asarray(labels), FEATURE_NAMES, skipped


def _score(y_true: np.ndarray, probabilities: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    prediction = np.argmax(probabilities, axis=1)
    report = classification_report(
        y_true, prediction, labels=np.arange(len(class_names)),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_true, prediction, labels=np.arange(len(class_names))
        ).tolist(),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        },
    }
    if len(class_names) > 2:
        try:
            result["macro_roc_auc_ovr"] = float(
                roc_auc_score(
                    y_true, probabilities,
                    labels=np.arange(len(class_names)),
                    multi_class="ovr", average="macro",
                )
            )
        except ValueError as exc:
            result["macro_roc_auc_ovr"] = None
            result["roc_auc_note"] = str(exc)
    elif len(class_names) == 2:
        try:
            result["roc_auc_binary"] = float(roc_auc_score(y_true, probabilities[:, 1]))
        except ValueError as exc:
            result["roc_auc_binary"] = None
            result["roc_auc_note"] = str(exc)
    return result


def _render_training_summary(metrics: dict[str, Any]) -> str:
    ds = metrics["dataset"]
    lines = [
        "## Rabi Crop Doctor — Selective Kaggle Benchmark Summary",
        "",
        f"- **Dataset**: `{ds['kaggle_id']}`",
        f"- **Download mode**: `{ds.get('download_mode', 'local')}`",
        f"- **Selected crops**: `{', '.join(ds['selected_crops']) if isinstance(ds['selected_crops'], list) else ds['selected_crops']}`",
        f"- **Crops present in dataset**: `{', '.join(ds['crops_present_in_dataset'])}`",
    ]
    if ds.get("crops_missing_from_dataset"):
        lines.append(
            f"- **Timeline crops not present in this Kaggle source**: `{', '.join(ds['crops_missing_from_dataset'])}`"
        )
    if metrics.get("timeline"):
        tl = metrics["timeline"]
        lines.append(
            f"- **Project timeline**: `{tl['quarter']}` ({tl['milestone']}, mode=`{tl['mode']}`, as_of=`{tl['as_of']}`)"
        )
    lines.extend([
        f"- **Split protocol**: {ds['split_protocol']}",
        f"- **Images decoded**: {ds['training_images_after_decode']:,} train / {ds['held_out_test_images_after_decode']:,} held-out across {ds['num_classes']} classes",
        "",
        "### Model Performance",
        "",
        "| Model | CV Accuracy | Held-out Accuracy | Held-out Macro-F1 | Macro ROC-AUC |",
        "|:---|---:|---:|---:|---:|",
    ])
    for name, m in metrics["models"].items():
        cv = m.get("cross_validation") or {}
        cv_acc = f"{cv['accuracy_mean'] * 100:.2f}%" if "accuracy_mean" in cv else "—"
        auc_val = m.get("macro_roc_auc_ovr", m.get("roc_auc_binary"))
        auc_str = f"{auc_val:.4f}" if isinstance(auc_val, (int, float)) else "—"
        lines.append(
            f"| `{name}` | {cv_acc} | {m['accuracy'] * 100:.2f}% | {m['macro_f1']:.4f} | {auc_str} |"
        )
    lines.extend(["", "### Class Distribution", "", "| Class | Train | Held-out |", "|:---|---:|---:|"])
    for cls_name in ds["classes"]:
        lines.append(
            f"| `{cls_name}` | {ds['training_class_counts'][cls_name]:,} | {ds['test_class_counts'][cls_name]:,} |"
        )
    lines.append("")
    return "\n".join(lines)


def train(
    dataset_dir: Path,
    output_dir: Path,
    crops: list[str] | None,
    test_size: float,
    cv_folds: int,
    random_state: int,
    skip_hash_dedup: bool,
    timeline_context: dict[str, Any] | None = None,
    download_mode: str = "local",
) -> dict[str, Any]:
    records = _collect_records(dataset_dir, crops)
    raw_counts = Counter(r["label"] for r in records)
    print(f"Discovered {len(records):,} images across {len(raw_counts)} class folders.")

    dedup_report = {"duplicate_images_removed": 0, "conflicting_label_hash_groups_removed": 0}
    if not skip_hash_dedup:
        records, dedup_report = _deduplicate_records(records)
        print(
            f"Exact-duplicate audit: removed {dedup_report['duplicate_images_removed']:,}; "
            f"conflicting-label groups: {dedup_report['conflicting_label_hash_groups_removed']:,}"
        )

    train_records, test_records, split_description, excluded = _make_split(
        records, test_size, random_state
    )
    train_counts = Counter(r["label"] for r in train_records)
    test_counts = Counter(r["label"] for r in test_records)
    class_names = sorted(train_counts)
    print("Split:", split_description)
    print(f"Training images: {len(train_records):,}; held-out test images: {len(test_records):,}")
    print("Classes:", len(class_names))
    for name in class_names:
        print(f"  {name:<36} train={train_counts[name]:>5,}  test={test_counts[name]:>5,}")

    print("\nExtracting 28-dimensional image features...")
    X_train, y_train_names, feature_names, skipped_train = _extract_matrix(train_records)
    X_test, y_test_names, _, skipped_test = _extract_matrix(test_records)

    # Decoding failures can alter class support; retain only classes present in both matrices.
    common = sorted(set(y_train_names) & set(y_test_names))
    train_mask = np.isin(y_train_names, common)
    test_mask = np.isin(y_test_names, common)
    X_train, y_train_names = X_train[train_mask], y_train_names[train_mask]
    X_test, y_test_names = X_test[test_mask], y_test_names[test_mask]
    class_names = common
    if len(class_names) < 2:
        raise RuntimeError("Fewer than two classes remain after image decoding.")

    encoder = LabelEncoder().fit(class_names)
    y_train = encoder.transform(y_train_names)
    y_test = encoder.transform(y_test_names)
    min_train_class = min(Counter(y_train).values())
    if min_train_class < 2:
        raise RuntimeError("Every training class needs at least two decodable images.")

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    min_cv = min(Counter(y_train).values())
    folds = min(cv_folds, min_cv)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state) if folds >= 2 else None

    # Guard MLP internal stratified early-stopping split when per-class count is small.
    min_fold_train = min_train_class if not cv else max(1, (min_train_class * (folds - 1)) // folds)
    use_early_stopping = min_fold_train >= 16 and int(len(X_train) * 0.12) >= len(class_names) * 2
    mlp_batch_size = min(128, max(8, len(X_train) // 4))

    estimators = {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=4.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            cache_size=2048,
            random_state=random_state,
        ),
        "ann_mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            batch_size=mlp_batch_size,
            learning_rate_init=0.001,
            max_iter=400,
            early_stopping=use_early_stopping,
            n_iter_no_change=20,
            validation_fraction=0.12,
            random_state=random_state,
        ),
    }

    crops_present = sorted({_infer_crop(c) for c in class_names})
    crops_missing = (
        [c for c in crops if _slug(c) not in set(crops_present)]
        if crops
        else []
    )
    if crops_missing:
        print(
            f"Note: timeline crops with 0 matching images in {KAGGLE_DATASET}: {crops_missing}",
            flush=True,
        )

    metrics: dict[str, Any] = {
        "dataset": {
            "name": "20k+ Multi-Class Crop Disease Images",
            "kaggle_id": KAGGLE_DATASET,
            "dataset_root": str(dataset_dir.resolve()),
            "download_mode": download_mode,
            "selected_crops": crops or "all",
            "crops_present_in_dataset": crops_present,
            "crops_missing_from_dataset": crops_missing,
            "split_protocol": split_description,
            "raw_image_count": len(records),
            "training_images_after_decode": int(len(X_train)),
            "held_out_test_images_after_decode": int(len(X_test)),
            "num_classes": len(class_names),
            "classes": class_names,
            "training_class_counts": {
                name: int(Counter(y_train_names)[name]) for name in class_names
            },
            "test_class_counts": {
                name: int(Counter(y_test_names)[name]) for name in class_names
            },
            "excluded_or_unpaired_classes": excluded,
            "feature_names": feature_names,
            "num_features": len(feature_names),
            "exact_duplicate_audit": dedup_report,
            "images_skipped_train": skipped_train,
            "images_skipped_test": skipped_test,
            "cross_validation_folds": folds if cv else 0,
        },
        "models": {},
    }
    if timeline_context is not None:
        metrics["timeline"] = timeline_context

    probabilities: dict[str, np.ndarray] = {}
    fitted: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}

    for name, estimator in estimators.items():
        print(f"\nTraining {name}...", flush=True)
        cv_summary: dict[str, Any] = {}
        if cv:
            pipeline = Pipeline([("scale", StandardScaler()), ("model", estimator)])
            cv_result = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
                n_jobs=1,
                error_score="raise",
            )
            cv_summary = {
                "accuracy_mean": float(np.mean(cv_result["test_accuracy"])),
                "accuracy_sd": float(np.std(cv_result["test_accuracy"], ddof=1)),
                "macro_f1_mean": float(np.mean(cv_result["test_macro_f1"])),
                "macro_f1_sd": float(np.std(cv_result["test_macro_f1"], ddof=1)),
            }

        started = time.perf_counter()
        estimator.fit(X_train_scaled, y_train)
        fit_seconds = time.perf_counter() - started
        probs = estimator.predict_proba(X_test_scaled)
        probabilities[name] = probs
        fitted[name] = estimator
        predictions[name] = np.argmax(probs, axis=1)
        model_metrics = _score(y_test, probs, class_names)
        model_metrics.update({
            "cross_validation": cv_summary,
            "fit_seconds": float(fit_seconds),
        })
        if hasattr(estimator, "loss_curve_"):
            model_metrics["loss_curve"] = [float(x) for x in estimator.loss_curve_]
        metrics["models"][name] = model_metrics
        print(
            f"{name}: held-out accuracy={model_metrics['accuracy']:.4f}; "
            f"macro-F1={model_metrics['macro_f1']:.4f}; fit={fit_seconds:.1f}s",
            flush=True,
        )

    # Equal-weight probability average is reported as a benchmark, not assumed superior.
    ensemble_probs = np.mean(np.stack(list(probabilities.values()), axis=0), axis=0)
    ensemble_metrics = _score(y_test, ensemble_probs, class_names)
    metrics["models"]["soft_vote_ensemble"] = ensemble_metrics
    print(
        f"soft_vote_ensemble: held-out accuracy={ensemble_metrics['accuracy']:.4f}; "
        f"macro-F1={ensemble_metrics['macro_f1']:.4f}",
        flush=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "trained_models.pkl"
    with model_path.open("wb") as handle:
        pickle.dump({
            "scaler": scaler,
            "models": fitted,
            "class_names": class_names,
            "feature_names": feature_names,
            "source_dataset": KAGGLE_DATASET,
            "selected_crops": crops or "all",
            "timeline": timeline_context,
        }, handle)

    metrics["environment"] = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    metrics_path = output_dir / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary_path = output_dir / "training_summary.md"
    summary_path.write_text(_render_training_summary(metrics), encoding="utf-8")

    pred_path = output_dir / "held_out_predictions.csv"
    pred_frame = pd.DataFrame({
        "true_label": encoder.inverse_transform(y_test),
    })
    for name, pred in predictions.items():
        pred_frame[f"{name}_prediction"] = encoder.inverse_transform(pred)
    pred_frame.to_csv(pred_path, index=False)

    print("\nTraining complete.")
    print("Model bundle:", model_path)
    print("Metrics:", metrics_path)
    print("Summary:", summary_path)
    print("Held-out predictions:", pred_path)
    print(
        "\nIMPORTANT: This is a separate external-dataset research bundle. "
        "The production Android model was not overwritten."
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and train Rabi Crop Doctor on the Kaggle 20k+ crop disease image dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/kaggle_20k_crop_disease_images"),
        help="Directory containing the extracted Kaggle image folders.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and extract dataset images using selective HTTP Range (with Kaggle CLI fallback).",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download and extract the selected dataset images and exit without training.",
    )
    parser.add_argument(
        "--full-download",
        action="store_true",
        help="Force full Kaggle dataset download instead of selective Rabi crop extraction.",
    )
    parser.add_argument(
        "--crops",
        nargs="+",
        default=None,
        help="Optional crop prefixes to include, e.g. --crops wheat rice. Overrides --timeline.",
    )
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="Automatically select Rabi crops from ROADMAP.md and data/rabi_crops_knowledge.json.",
    )
    parser.add_argument(
        "--timeline-mode",
        choices=["roadmap", "peak-month", "all-rabi"],
        default=None,
        help="Timeline selection mode (implies --timeline): roadmap, peak-month, or all-rabi.",
    )
    parser.add_argument(
        "--quarter",
        default="auto",
        help="Roadmap quarter override when using --timeline: Q1, Q2, Q3, Q4, or auto.",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Reference ISO date YYYY-MM-DD for timeline resolution (default: today).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/kaggle_20k"))
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--skip-hash-dedup",
        action="store_true",
        help="Skip exact-file duplicate audit (faster for very large downloads).",
    )
    args = parser.parse_args()

    if not 0.05 <= args.test_size <= 0.5:
        parser.error("--test-size must be between 0.05 and 0.50")
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")

    timeline_context: dict[str, Any] | None = None
    if args.crops:
        crop_filter = [_slug(x) for x in args.crops]
    elif args.timeline or args.timeline_mode is not None:
        tl_mode = args.timeline_mode or "roadmap"
        timeline_context = resolve_rabi_timeline(
            as_of=args.as_of,
            mode=tl_mode,
            quarter=args.quarter,
        )
        crop_filter = list(timeline_context["crops"])
        print(
            f"Resolved project timeline ({timeline_context['quarter']} | {timeline_context['milestone']} | "
            f"mode={tl_mode}, as_of={timeline_context['as_of']}): crops={crop_filter}",
            flush=True,
        )
    else:
        crop_filter = None

    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    download_mode = "local"
    if args.download or args.download_only:
        download_mode = _download_dataset(
            args.dataset_dir,
            crops=crop_filter,
            full_download=args.full_download,
        )
        if args.download_only:
            print(f"Download complete ({download_mode}) -> {args.dataset_dir}", flush=True)
            return
    elif not any(args.dataset_dir.rglob("*")):
        parser.error(
            f"No data found in {args.dataset_dir}. Supply --download "
            "or point --dataset-dir to an extracted dataset."
        )

    train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        crops=crop_filter,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        skip_hash_dedup=args.skip_hash_dedup,
        timeline_context=timeline_context,
        download_mode=download_mode,
    )


if __name__ == "__main__":
    main()
