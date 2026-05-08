# ALRL: Advanced Lifelong Regression Learning for Heterogeneous Tasks

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Reference implementation for the paper:

> **ALRL: Advanced Lifelong Regression Learning for Heterogeneous Tasks**
> Xinyao Liu, Boxuan Zhu, Steven Guan.
> *Engineering Applications of Artificial Intelligence* (under review).
> 📄 [Manuscript](manuscript.pdf) — 🌐 [Project page (anonymous, review-only)](#)

The framework targets continual regression learning over a stream of
tasks with **variable input dimensionality**, combining:

1. **Per-task input adapters** — map $d_t$-dimensional inputs into a
   shared 128-dimensional representation.
2. **Globally shared backbone** — a single `Linear → ReLU → LayerNorm`
   block (128 → 256), the locus of catastrophic forgetting.
3. **Per-task heads** — task-specific 256→128→1 MLPs warm-started from
   the most-similar past task.
4. **Three coordinated forgetting-prevention mechanisms** — selective
   Elastic Weight Consolidation (anchored only to similar past tasks),
   an importance- and recency-aware replay buffer, and a
   linearly-decayed replay-loss schedule.
5. **Hybrid task similarity** — 32-dimensional distributional signature
   plus a small learned embedding projection drives module
   inheritance, EWC anchoring, and replay re-weighting; **no task
   labels required**.

---

## ⚡ One-line reproduction

```bash
git clone <repo> && cd fyp_code && bash reproduce_all.sh
```

This single command installs dependencies, runs the full 5-seed × 8-config × 7-task experimental sweep, aggregates the results, and regenerates every figure / table in the paper. Total wall-clock time on a CPU-only Apple-M-class laptop: **≈ 4–5 hours**.

For a 17-minute smoke test (1 config × 1 seed) before committing to the full run:

```bash
bash reproduce_minimal.sh
```

---

## Headline results

| Method | Knowledge retention | Avg ΔR² |
|---|---|---|
| Fine-tuning (baseline) | 80.3% ± 33.6 | −0.0697 |
| EWC only | 98.4% ± 1.3 | −0.0068 |
| Replay only | 68.5% ± 49.0 | −0.1091 |
| **Full ALRL (this work)** | **97.7% ± 1.7** | **−0.0103** |

5 random seeds × 8 ablation configurations × 7 regression tasks (real-estate / medical / chemical / engineering domains, including UCI Concrete Compressive Strength as a natively continuous-target control). The full framework achieves **85% reduction in mean forgetting** and a **20× tightening of across-seed variance** relative to the fine-tuning baseline.

---

## Repository layout

```
fyp_code/
├── reproduce_all.sh                    # One-command end-to-end reproduction
├── reproduce_minimal.sh                # 17-min smoke test
├── improved_lifelong_regression.py     # Core model (SharedBackbone, DynamicNetwork,
│                                       #   InputAdapter, TaskEmbeddingNetwork,
│                                       #   AdvancedLifelongRegressionModel)
├── forgetting_prevention_comparison.py # Multi-seed training runner with --seeds CLI
├── datasets_extra.py                   # Optional UCI datasets (Concrete, Energy)
├── example_usage.py                    # Single-run demo + analysis utilities
├── visualization_script.py             # Stand-alone visualisation helpers
├── requirements.txt                    # Python dependencies
├── experiment_results/                 # Per-seed result pickles + per-config plots
└── visualizations/                     # Output of the demo script
```

Post-processing tools are in a separate analysis directory at
`~/Documents/Claude/Projects/ALML/`:

```
ALML/
├── aggregate_seeds.py                  # mean ± std across seeds → CSV / LaTeX tables
├── make_figures.py                     # 6 vector PDFs + Figure 1 TikZ source
└── rename_pickle_keys.py               # one-shot helper for legacy pickles
```

---

## Installation

Tested on Python 3.10 / 3.11 (macOS, Linux). A CPU is sufficient
(≈ 4 min per configuration on an M-class Mac); GPU is used
automatically if PyTorch detects one.

```bash
git clone <repo>
cd fyp_code

# Recommended: isolated env
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

The optional UCI Concrete Compressive Strength dataset is downloaded
on first use to `.uci_cache/`.

---

## Reproducing the paper (manual three steps)

If you prefer running the steps manually instead of `reproduce_all.sh`:

### 1. Multi-seed sweep

```bash
# 8 configurations × 5 seeds = 40 runs.
# caffeinate -i prevents macOS from sleeping during the long run.
caffeinate -i python -u forgetting_prevention_comparison.py \
        --seeds 0 1 2 3 4 --skip-analyze 2>&1 \
    | grep --line-buffered -v "Intel MKL WARNING\|Intel oneAPI" \
    | tee run_log.txt
```

Each `(config, seed)` writes to
`experiment_results/<config_name>/seed_<i>/experiment_results.pkl`.

To resume / re-run subsets:
```bash
python forgetting_prevention_comparison.py --seeds 2 3 4 --skip-analyze
python forgetting_prevention_comparison.py --seeds 0 --configs 7   # only Full ALRL
```

### 2. Aggregate across seeds

```bash
python ~/Documents/Claude/Projects/ALML/aggregate_seeds.py
```

Outputs to `~/Documents/Claude/Projects/ALML/aggregated/`:
- `aggregated_per_task.csv` — per-(config, task) mean ± std
- `aggregated_per_config.csv` — per-config aggregate
- `aggregated_table4.tex` / `aggregated_table5.tex` — drop-in LaTeX
- `aggregated_metrics.json` — machine-readable dump

### 3. Generate publication figures

```bash
python ~/Documents/Claude/Projects/ALML/make_figures.py
```

Outputs to `~/Documents/Claude/Projects/ALML/figures/`:

| File | Used as |
| --- | --- |
| `fig6_task_similarity.pdf` | Task-similarity heatmap |
| `fig7_final_performance.pdf` | Final R²/MSE/MAE per task (full ALRL) |
| `fig8_forgetting_compare.pdf` | Aggregate ΔR² across configurations |
| `fig9_task_level_forgetting.pdf` | Per-task ΔR² across configurations |
| `fig10a_r2_over_time.pdf` | R² trajectory per task as new tasks arrive |
| `fig10b_task_network.pdf` | Task-relationship network |
| `figure1_architecture.tex` | Drop-in TikZ for Figure 1 |

---

## Datasets

Seven regression tasks; the first six load automatically via scikit-learn (no manual download). Concrete Compressive Strength downloads on first use from the UCI ML Repository.

| # | Task | Source | Features | Samples | Target |
| - | --- | --- | --- | --- | --- |
| 1 | `california` | `fetch_california_housing` | 8 | 20,640 | Median house value |
| 2 | `wine` | `load_wine` | 13 | 178 | Cultivar class (ordinal) |
| 3 | `diabetes` | `load_diabetes` | 10 | 442 | Disease progression |
| 4 | `cancer` | `load_breast_cancer` | 30 | 569 | Diagnosis class (ordinal) |
| 5 | `concrete` | UCI Concrete Compressive Strength | 8 | 1,030 | Compressive strength (MPa) |
| 6 | `california_older` | derived from #1 | 8 | 10,320 | House value, high-HouseAge half |
| 7 | `california_subset` | derived from #1 | 5 | 20,640 | House value, first 5 features |

Each task uses a fixed 80/20 train/test split (`SPLIT_SEED = 42`); StandardScaler is fit on train only.

---

## Reproducibility checklist

- [x] All random sources seeded (`numpy`, `torch`, `random`).
- [x] Train / test split fixed (`SPLIT_SEED = 42`).
- [x] Per-seed result pickles preserved under `experiment_results/`.
- [x] Hyperparameters listed in `forgetting_prevention_comparison.py` and Table 2 of the paper.
- [x] Single shared backbone (no covert per-task copies); see `SharedBackbone` class in `improved_lifelong_regression.py`.
- [x] Evaluation uses held-out test split, not training-set subset.
- [x] EWC anchors only similar past tasks (`S_t = {k<t : sim(t,k) ≥ τ}`).
- [x] Replay weight follows linear decay `α(e) = 0.3·(1 − e/E)`.
- [x] One-command end-to-end reproduction (`reproduce_all.sh`).

---

## Citation

If you use this code or build on the framework, please cite:

```bibtex
@article{liu2026alrl,
  title  = {ALRL: Advanced Lifelong Regression Learning for Heterogeneous Tasks},
  author = {Liu, Xinyao and Zhu, Boxuan and Guan, Steven},
  journal= {Engineering Applications of Artificial Intelligence},
  year   = {2026},
  note   = {Under review}
}
```

---

## License

Released under the MIT License. See [LICENSE](LICENSE).

---

## Contact

- **Xinyao Liu** (corresponding code author): xinyaoliu2027@u.northwestern.edu
- **Steven Guan** (corresponding paper author): Steven.Guan@xjtlu.edu.cn

For substantive questions about the method, please open a GitHub issue or
contact the authors directly.
