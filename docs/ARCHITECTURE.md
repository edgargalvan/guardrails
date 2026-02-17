# ARCHITECTURE.md

## Overview

The system is a comparison engine. The unit of work is not "run one strategy" — it's "run N strategies and M benchmarks across K time windows and compare them." Everything is driven by `default.yaml`.

```
default.yaml → run.py → [data, filters, weights, backtest, metrics, compare, plots] → results/
```

## Directory Structure

```
project/
├── AGENT.md                # Bot routing (read first)
├── STRATEGY.md             # Research writeup
├── README.md               # Quick start
├── docs/
│   ├── ARCHITECTURE.md     # This file
│   ├── GOVERNANCE.md       # Research discipline
│   └── CODING.md           # Style and conventions
├── configs/
│   └── default.yaml        # Strategy definitions, benchmarks, settings
├── src/
│   ├── data.py             # Download, cache, align dates
│   ├── filters.py          # Signal functions (SMA, etc.)
│   ├── weights.py          # Weighting functions (equal, fixed, momentum, etc.)
│   ├── backtest.py         # Weights + returns + exit mode → equity curve
│   ├── metrics.py          # Sharpe, CAGR, MaxDD, capture ratios, etc.
│   ├── compare.py          # Orchestrator: run all strategies across all windows
│   └── plots.py            # All visualization
├── scripts/
│   └── run.py              # Entry point: load config → compare → save results
├── tests/                  # Unit and regression tests
├── results/                # Auto-generated output (JSON, CSV, PNG)
└── data/                   # Cached price data
```

## Data Flow

1. **run.py** loads `default.yaml`
2. **data.py** collects all unique tickers across all strategies and benchmarks, downloads daily prices via yfinance, caches locally, aligns to common trading calendar
3. For each strategy, for each window:
   - **filters.py** computes the in/out signal (boolean DataFrame)
   - **weights.py** computes target weights among survivors
   - **backtest.py** applies exit mode (cash or renormalize), applies costs, computes equity curve. Cash portion earns the cash vehicle's return.
4. For each benchmark, for each window:
   - **backtest.py** runs with fixed weights, quarterly rebalance, no filter
5. **compare.py** collects all equity curves, calls **metrics.py** on each, builds comparison tables
6. **plots.py** generates all configured plots
7. Results saved to `results/`: summary JSON, per-window CSVs, PNG plots

## Module Interfaces

### filters.py

Each filter is a function with this pattern:

```python
def sma(prices: pd.DataFrame, window: int = 200, frequency: str = "monthly", **kwargs) -> pd.DataFrame:
    """
    Args:
        prices: Daily close prices, columns = tickers
        window: Moving average lookback in trading days
        frequency: How often to evaluate ("monthly", "weekly", "daily")

    Returns:
        Boolean DataFrame, same shape as prices. True = asset is "in" (above trend).
        Signal is evaluated at the specified frequency and forward-filled to daily.
    """
```

To add a new filter: write a function in `filters.py` following this signature. Reference it in the config by function name:

```yaml
filter:
  type: my_new_filter     # Must match function name in filters.py
  my_param: 42            # Passed as kwargs
```

The runner looks up the function by name and passes all config params as kwargs.

### weights.py

Each weighting scheme is a function with this pattern:

```python
def equal(prices: pd.DataFrame, in_mask: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Args:
        prices: Daily close prices, columns = tickers
        in_mask: Boolean DataFrame from the filter. True = asset is "in".

    Returns:
        Weight DataFrame, same shape as prices. Weights are for "in" assets only.
        Weights do NOT need to sum to 1.0 — the exit mode handles that.
    """
```

For momentum weighting, additional kwargs come from the config:

```python
def momentum(prices: pd.DataFrame, in_mask: pd.DataFrame,
             lookback_days: int = 252, skip_days: int = 21,
             split: list = [0.70, 0.20, 0.10], **kwargs) -> pd.DataFrame:
```

To add a new weighting scheme (e.g., inverse volatility, risk parity, min variance):
1. Write a function in `weights.py` following the signature above
2. Reference it in the config: `weights: {type: my_scheme, my_param: value}`

### backtest.py

The backtest engine is not swappable — it's the core. It takes:

```python
def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    exit_mode: str,               # "cash" or "renormalize"
    cash_prices: pd.Series,       # Cash vehicle daily prices (e.g., SHY)
    initial_capital: float,
    commission_per_trade: float,
    slippage_bps: float,
) -> pd.Series:
    """Returns daily equity curve as a Series."""
```

Key behaviors:
- **Cash exit**: weights for "out" assets = 0, survivors keep computed weights, remainder invested in cash vehicle (SHY) via explicit `_CASH` column
- **Renormalize**: weights for "out" assets = 0, survivors' weights scaled to sum to 1.0
- **Costs**: slippage + commission applied per-order as a fraction of trade value via vectorbt's fee model
- **Execution timing**: vectorbt executes at same-bar close, equivalent to next-bar execution (no look-ahead bias)

### metrics.py

Each metric is a function:

```python
def sharpe(returns: pd.Series, risk_free: pd.Series = None) -> float:
    """Annualized Sharpe ratio."""

def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative percentage."""

def upside_capture(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Upside capture ratio vs benchmark."""
```

To add a new metric:
1. Write a function in `metrics.py`
2. Add its name to the `metrics` list in the config

### compare.py

The orchestrator. Not user-extensible — this is internal plumbing.

```python
def run_comparison(config: dict) -> dict:
    """
    Runs all strategies and benchmarks across all windows.
    Returns a nested dict:
    {
        window_label: {
            strategy_name: {
                "equity_curve": pd.Series,
                "metrics": {metric_name: value, ...}
            }
        }
    }
    """
```

### plots.py

Each plot type is a function:

```python
def equity_curves(results: dict, window: str, initial_capital: float, **kwargs):
    """Plots all strategies and benchmarks for a given window on one chart."""
```

To add a new plot type:
1. Write a function in `plots.py`
2. Add its name to the `plots` list in the config

## Config Schema

See `configs/default.yaml` for the complete, annotated config. The config is the user interface — all strategy definitions, benchmarks, and settings live there. Users should never need to edit Python files to try a new combination.

### Strategy block

```yaml
strategy_name:
  description: "Human-readable description"
  universe: [TICKER1, TICKER2, ...]
  filter:
    type: function_name        # Must exist in filters.py
    # All other keys passed as kwargs to the filter function
  weights:
    type: function_name        # Must exist in weights.py
    # All other keys passed as kwargs to the weighting function
  exit_mode: cash | renormalize
```

### Benchmark block

```yaml
benchmark_name:
  description: "Human-readable description"
  tickers: [TICKER1, TICKER2, ...]
  weights: {TICKER1: 0.6, TICKER2: 0.4}
```

Benchmarks are buy-and-hold with periodic rebalancing (frequency set in `settings.benchmark_rebalance`). No filter applied.

## Output Structure

```
results/
├── summary.json              # All metrics, all strategies, all windows
├── Full_Period/
│   ├── comparison_table.csv
│   ├── equity_curves.png
│   ├── drawdowns.png
│   ├── annual_returns.png
│   ├── risk_return_scatter.png
│   └── rolling_12m.png
├── Post_GFC/
│   ├── ...
└── ...
```

## Extending the System

| I want to... | Do this |
|---|---|
| Try a different moving average window | Change `filter.window` in the config |
| Try a completely different filter (RSI, dual MA) | Write a function in `filters.py`, reference by name in config |
| Try different weights (inverse vol, min variance) | Write a function in `weights.py`, reference by name in config |
| Add an asset to the universe | Add the ticker to `universe` in the config |
| Add a benchmark | Add a block under `benchmarks` in the config |
| Add a metric | Write a function in `metrics.py`, add to `metrics` list in config |
| Add a plot type | Write a function in `plots.py`, add to `plots` list in config |
| Change initial capital or costs | Edit `settings` in the config |

The pattern is always: write a function with the standard signature, reference it in the config by name. No registration, no classes, no boilerplate.
