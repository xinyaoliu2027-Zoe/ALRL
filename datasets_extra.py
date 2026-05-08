"""
datasets_extra.py
-----------------
Optional extra benchmark datasets to mix into the ALRL task stream.

These are *real* regression datasets (continuous targets, not
classification-as-regression) which strengthen the empirical claims
of the paper for a small additional cost. They are NOT loaded by
default — if you want to add them to the stream, edit
``forgetting_prevention_comparison.py`` and append the entries
returned by ``add_extra_datasets()`` to the ``datasets`` dict and the
new task ids to ``task_sequence``.

Each loader returns a dict in the same shape as
``forgetting_prevention_comparison._make_entry`` produces:
    {
        'X_train', 'y_train', 'X_test', 'y_test',
        'description', 'feature_names'
    }
i.e., already split 80/20 and standardised on the train portion only.

USAGE
-----
>>> from datasets_extra import add_extra_datasets
>>> datasets = load_datasets()
>>> datasets.update(add_extra_datasets(which=("concrete", "energy")))
>>> task_sequence = ['california', 'wine', 'diabetes', 'cancer',
...                  'concrete',                     # <-- new
...                  'california_older', 'california_subset']
"""

from __future__ import annotations

import io
import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Same split conventions used by forgetting_prevention_comparison.py
SPLIT_SEED = 42
TEST_SIZE = 0.2

# Cache UCI data locally so repeated runs don't re-download.
CACHE_DIR = os.path.expanduser("~/Documents/fyp_code/.uci_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _split_and_scale(X, y, description, feature_names):
    """80/20 split + StandardScaler fit on train only. Matches
    forgetting_prevention_comparison._make_entry exactly so the
    results are interchangeable."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, shuffle=True
    )
    sx = StandardScaler()
    X_tr = sx.fit_transform(X_tr)
    X_te = sx.transform(X_te)
    sy = StandardScaler()
    y_tr = sy.fit_transform(y_tr.reshape(-1, 1)).ravel()
    y_te = sy.transform(y_te.reshape(-1, 1)).ravel()
    return {
        "X_train": X_tr.astype(np.float32),
        "y_train": y_tr.astype(np.float32),
        "X_test":  X_te.astype(np.float32),
        "y_test":  y_te.astype(np.float32),
        "description": description,
        "feature_names": list(feature_names),
    }


def _download_to_cache(url, filename):
    path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(path):
        return path
    print(f"Downloading {url} -> {path}")
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    with open(path, "wb") as f:
        f.write(data)
    return path


# ---------------------------------------------------------------------
# loaders — one per dataset
# ---------------------------------------------------------------------
def load_concrete():
    """UCI Concrete Compressive Strength.

    1030 samples, 8 features, target = compressive strength (MPa).
    A canonical real regression dataset for benchmarking.
    """
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "concrete/compressive/Concrete_Data.xls")
    path = _download_to_cache(url, "Concrete_Data.xls")
    df = pd.read_excel(path)
    feature_names = [c.strip().split('(')[0].strip() for c in df.columns[:-1]]
    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = df.iloc[:, -1].to_numpy(dtype=np.float64)
    return _split_and_scale(
        X, y,
        description="Concrete Compressive Strength (8 features)",
        feature_names=feature_names,
    )


def load_energy_efficiency():
    """UCI Energy Efficiency.

    768 samples, 8 features, two real-valued targets (heating load,
    cooling load). We use heating load as the regression target.
    """
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "00242/ENB2012_data.xlsx")
    path = _download_to_cache(url, "ENB2012_data.xlsx")
    df = pd.read_excel(path)
    feature_names = [f"X{i+1}" for i in range(8)]
    X = df.iloc[:, :8].to_numpy(dtype=np.float64)
    y = df.iloc[:, 8].to_numpy(dtype=np.float64)  # heating load
    return _split_and_scale(
        X, y,
        description="Energy Efficiency, heating load (8 features)",
        feature_names=feature_names,
    )


def load_air_quality():
    """UCI Air Quality (CO concentration).

    Subset: predict CO(GT) from the other on-board sensor readings.
    Drops rows with sentinel value -200.
    """
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "00360/AirQualityUCI.zip")
    path = _download_to_cache(url, "AirQualityUCI.zip")
    import zipfile
    with zipfile.ZipFile(path) as z:
        with z.open("AirQualityUCI.csv") as f:
            df = pd.read_csv(io.TextIOWrapper(f, encoding="latin-1"),
                              sep=";", decimal=",")
    # strip empty cols and rows
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = df.replace(-200, np.nan).dropna()
    target = "CO(GT)"
    if target not in df.columns:
        raise RuntimeError("Air-quality dataset format changed; check columns.")
    feature_cols = [c for c in df.columns
                    if c not in {"Date", "Time", target}
                    and df[c].dtype != object]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df[target].to_numpy(dtype=np.float64)
    return _split_and_scale(
        X, y,
        description=f"Air Quality (CO; {len(feature_cols)} features)",
        feature_names=feature_cols,
    )


# ---------------------------------------------------------------------
# convenience entry-point
# ---------------------------------------------------------------------
LOADERS = {
    "concrete": load_concrete,
    "energy":   load_energy_efficiency,
    "air":      load_air_quality,
}


def add_extra_datasets(which=("concrete",)):
    """Return a dict of extra datasets, keyed by short name.

    Parameters
    ----------
    which : iterable of {"concrete", "energy", "air"}
        Which loaders to call. Defaults to just "concrete" — the safest
        and most often-cited continuous-target real-regression
        benchmark.
    """
    out = {}
    for name in which:
        if name not in LOADERS:
            raise KeyError(f"Unknown extra dataset: {name}. "
                           f"Available: {list(LOADERS)}")
        try:
            out[name] = LOADERS[name]()
        except Exception as exc:
            print(f"[warn] could not load extra dataset '{name}': {exc}")
    return out


if __name__ == "__main__":
    # Quick sanity check.
    extras = add_extra_datasets(which=("concrete",))
    for k, v in extras.items():
        print(f"{k}: {v['description']}, "
              f"train={v['X_train'].shape}, test={v['X_test'].shape}")
