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

Qlib is optional because the default workflow reads `prices/*.csv` directly.
Install it separately only when using `Alpha158Dataset.from_qlib`:

```bash
pip install ".[qlib]"
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
./run.sh
```

The script creates `.venv` and installs missing dependencies automatically.

Run a specific model with a dedicated configuration:

```bash
./train_model.sh mlp
./train_model.sh resnet
./train_model.sh ftt
```

Train all models sequentially and create a comparison table:

```bash
./train_all.sh
```

Existing completed models are skipped. To retrain all three from scratch:

```bash
./train_all.sh --force
```

The summary is written to:

```text
rtdl_quant/outputs/model_comparison.csv
```

All three models reuse `data/alpha158_prices.parquet`; factors are not rebuilt
between model runs. Results are stored separately:

```text
rtdl_quant/outputs/alpha158_mlp/
rtdl_quant/outputs/alpha158_resnet/
rtdl_quant/outputs/alpha158_ft_transformer/
```

The FTT configuration uses two 64-dimensional Transformer blocks and a batch
size of 128 to reduce runtime. It is still substantially slower than MLP and
ResNet on CPU.

## Industry and market-cap neutralization

Each model configuration includes an optional post-processing step:

```yaml
evaluation:
  neutralization:
    enabled: true
    industry: true
    market_cap: true
    standardize: true
    industry_path: industry/industry.csv
    industry_code_column: symbol
    industry_column: industry
```

For each date, the model score is residualized against industry dummies and
log free-float market cap. Free-float market cap is estimated from the local
market data as:

```text
free-float shares = volume / (turnover_percent / 100)
free-float market cap = close * free-float shares
```

Set `enabled: false` to use the raw model score. `predictions.parquet` retains
both `raw_prediction` and the final `prediction`. When neutralization is
enabled, additional files are created:

```text
daily_ic_raw.csv
group_returns_raw.csv
neutralization_summary.csv
```

The supplied industry file is a static classification dated June 22, 2026.
It is useful for current-universe research, but it does not capture historical
industry changes and should not be treated as a point-in-time classification.

`data.prices_build.max_instruments` controls how many stocks are written when
the factor cache is created. Set it to `null` for a full-universe cache or to a
smaller number for a quick factor-building test. Changing it requires deleting
the existing `data/alpha158_prices.parquet` cache.

For full-universe caches, `data.load_max_instruments` controls how many stocks
are loaded into memory for one training run. This does not rebuild or shrink
the cache:

```yaml
data:
  load_max_instruments: 500  # null loads every stock into RAM
```

On a typical Mac, start with 500 for MLP/ResNet and 200 for FTT. Loading all
5,000+ stocks at once can exceed memory and cause macOS to print `zsh: killed`.

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
