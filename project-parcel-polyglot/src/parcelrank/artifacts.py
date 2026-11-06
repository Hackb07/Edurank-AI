"""Shared artifact save/load — single source of truth for train/predict/
monitor/explain AND the Rust scorer (rust/parcel-scorer) which reads the
exact same JSON files.

Layout under models/:
  preprocessing.json   medians, means, stds, categories, carrierRates,
                       globalRate, featureNames (ordered)
  logreg.json          {weights, bias, ...}
  forest.json          recursive trees: {leaf, p} or {feature, threshold, left, right}
  threshold.json       {primary, logreg, forest, costFn, costFp}
  baseline_stats.json  frozen train-window distributions for the monitor
"""
import json
import os

ARTIFACTS = ("preprocessing.json", "logreg.json", "forest.json", "threshold.json", "baseline_stats.json")


def save_models(models_dir, artifacts: dict):
    os.makedirs(models_dir, exist_ok=True)
    for name, obj in zip(ARTIFACTS, artifacts):
        with open(os.path.join(models_dir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1)
            f.write("\n")


def load_models(models_dir: str) -> dict:
    out = []
    for name in ARTIFACTS:
        with open(os.path.join(models_dir, name), encoding="utf-8") as f:
            out.append(json.load(f))
    return {"preproc": out[0], "logreg": out[1], "forest": out[2], "thresholds": out[3], "baseline_stats": out[4]}
