# RTDL Quant

A modular research framework for cross-sectional stock prediction with
Alpha158, MLP, ResNet, FT-Transformer, IC/RankIC analysis, and grouped
portfolio backtests.

The neural architectures follow the design in *Revisiting Deep Learning Models
for Tabular Data* (NeurIPS 2021). The framework keeps data, estimators,
training, signal evaluation, and portfolio evaluation separate so that TabM,
TabR, LightGBM, and CatBoost can be added without changing experiment code.

## Project layout

```text
rtdl_quant/
├── backtest/       # Daily IC and grouped portfolio analysis
├── configs/        # YAML experiment definitions
├── datasets/       # Alpha158/Qlib adapters
├── experiments/    # Reproducible orchestration and artifact writing
├── metrics/        # NumPy regression and signal metrics
├── models/         # Local RTDL models and official-package wrappers
├── outputs/        # One directory per experiment
├── scripts/        # Data preparation/automation entry points
├── trainer/        # PyTorch training loop
├── utils/          # Configuration and reproducibility helpers
└── main.py
```

## Environment

Python 3.11 or 3.12 is the safest choice for the combined PyTorch and Qlib
dependency stack. Other Python 3 versions can be used when compatible wheels
are available for the target platform.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Data contract

The experiment runner accepts CSV or Parquet with one row per stock and date:

```text
date | code | feature_0 ... feature_157 | label | future_return
```

- `label`: model target, for example the within-date rank of the future
  20-trading-day return.
- `future_return`: the raw future return used by the grouped portfolio
  backtest. Keep this separate from a ranked/standardized label.
- Features must be cleaned using train-period statistics only to avoid
  look-ahead leakage.

For a fitted Qlib handler, use:

```python
from rtdl_quant.datasets import Alpha158Dataset

dataset = Alpha158Dataset.from_qlib(
    handler,
    segment=slice("2008-01-01", "2016-12-31"),
    rank_label=True,
)
```

The adapter understands Qlib's `(datetime, instrument)` MultiIndex and
`(feature, label)` column groups. It retains the raw Qlib label as
`future_return` and creates a within-date percentile rank as `label`.

### Using the local `prices/` market data

The default configuration now uses the 5,497 per-stock CSV files under
`prices/`. Raw market data is intentionally ignored by Git.

Build the reusable Alpha158 cache once:

```bash
python -m rtdl_quant.scripts.build_prices_dataset \
  --prices-dir prices \
  --output data/alpha158_prices.parquet \
  --start-date 2014-01-01 \
  --end-date 2026-06-18
```

For a quick pipeline check, limit the universe:

```bash
python -m rtdl_quant.scripts.build_prices_dataset \
  --max-instruments 100 \
  --start-date 2022-01-01
```

Limited universes are sampled evenly from the sorted SH/SZ file list rather
than taking only the first stock codes.

The builder reads the adjusted OHLCV fields, calculates the 158 Qlib-style
factors using only current and historical observations, and creates the raw
future 20-trading-day return. Cross-sectional rank labels are added only after
all instruments are combined. The Parquet cache is also ignored by Git.

## Run an experiment

The default configuration is a one-command pipeline. It checks for the
Alpha158 cache, builds it from `prices/` when missing, then trains, evaluates,
and runs the grouped backtest:

```bash
python main.py
```

The first run processes all configured stock CSVs and therefore takes much
longer than later runs. To perform a quick check first, set
`data.prices_build.max_instruments: 100` in the YAML file.

Artifacts are written to `rtdl_quant/outputs/<experiment_name>/`:

```text
config.yaml
train.log
metrics.csv
daily_ic.csv
group_returns.csv
predictions.parquet
training_history.csv
model.pt
tensorboard/
```

## Model APIs

`rtdl_quant.models.mlp` and `resnet` are readable local implementations for
research modifications. `rtdl_quant.models.wrappers` exposes `MLPModel`,
`ResNetModel`, and `FTTransformerModel` using the official
`rtdl_revisiting_models` package with a common `fit/predict/save/load` API.

Tree models should implement the same `BaseModel` interface. They do not belong
inside the PyTorch trainer; a backend-neutral experiment layer is the cleaner
extension point.

## Research cautions

- Split chronologically, never randomly across dates.
- Fit normalization, imputation, and feature selection on the training period.
- Use raw future returns for portfolio PnL and ranked returns for rank-style
  learning targets.
- Overlapping 20-day labels induce serial dependence; use purging/embargo for
  strict out-of-sample studies.
- This group backtest is a signal diagnostic, not an execution simulator. Add
  turnover, transaction costs, suspensions, limit-up/down rules, and tradable
  universe controls before interpreting it as investable performance.
