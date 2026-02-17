# 200dma Three-Fund Strategy

A config-driven backtesting framework for trend-filtered multi-asset portfolios.

A backtesting framework built on the trend-following approach described in Meb Faber's ["A Quantitative Approach to Tactical Asset Allocation"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=962461) (2007, *The Journal of Wealth Management*). Faber's core idea: use a 10-month (roughly 200-day) moving average as a trend filter to reduce drawdowns in a diversified portfolio.

**The strategy:** Hold SPY, TLT, and GLD. Check each against its 200-day moving average on the last trading day of each month. If it's below, move that allocation to cash (or redistribute to survivors). That's it.

**The result:** 1.04 Sharpe, -14% max drawdown, 7.9% CAGR over 18 years (2007–2026). During the 2008 crisis, the strategy gained +13% while SPY lost -37%.

Read [STRATEGY.md](STRATEGY.md) for the full research behind why this works and what was tested.

## Quick Start

```bash
pip install pandas numpy yfinance matplotlib pyyaml quantstats vectorbt
python scripts/run.py                     # uses configs/default.yaml
python scripts/run.py configs/custom.yaml  # uses a custom config
```

This runs all four strategy variants and benchmarks across five time windows. Results are saved to `results/`.

## The 2x2 Matrix

|  | Cash Exit | Renormalize |
|---|---|---|
| **Equal Weight** | 1.04 Sharpe, 7.9% CAGR, -14% MaxDD | 0.98 Sharpe, 11.3% CAGR, -22% MaxDD |
| **Momentum Tilt** | 0.98 Sharpe, 9.4% CAGR, -17% MaxDD | 0.98 Sharpe, 13.1% CAGR, -21% MaxDD |

All four are defined in `configs/default.yaml`. Comment out what you don't want, or modify parameters to experiment.

## Customizing

**Try a different filter window:** Change `filter.window` in the config.

**Try different assets:** Change `universe` in the config.

**Try a new weighting scheme:** Write a function in `src/weights.py`, reference it in the config.

**Add a benchmark:** Add a block under `benchmarks` in the config.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full details on extending the system.

## Testing a New Strategy

Before experimenting, read [docs/GOVERNANCE.md](docs/GOVERNANCE.md). It contains the research discipline that prevented us from overfitting across 12+ tested variants. The short version: pre-register your parameters, run once, compare against EW-cash, report all windows not just the best one.

## Files

| File | What |
|---|---|
| `STRATEGY.md` | The research: why this works, what was tested, what failed |
| `configs/default.yaml` | All strategy definitions, benchmarks, and settings |
| `docs/ARCHITECTURE.md` | Module design, interfaces, how to extend |
| `docs/GOVERNANCE.md` | Research discipline and evaluation standards |
| `docs/CODING.md` | Python conventions and anti-patterns |
| `AGENT.md` | Instructions for AI coding assistants |
