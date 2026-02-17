# CODING.md

## Dependencies

Core (required):
- pandas
- numpy
- yfinance
- matplotlib
- pyyaml
- quantstats (metrics computation and HTML tearsheet reports)
- vectorbt (portfolio construction engine in backtest.py)

Do not add new dependencies without justification. Most things can be done with pandas and numpy. If you're tempted to add scipy, sklearn, or a new framework, check if the existing code already handles the use case.

## Style

- Python 3.10+
- Type hints on all function signatures
- Docstrings on all public functions (Google style)
- f-strings for formatting
- snake_case for functions and variables
- No classes unless genuinely needed (prefer functions — the module pattern uses functions as the unit of composition)
- Keep functions short. If a function exceeds 50 lines, it probably does too many things.
- Black formatting (default settings)

## The Config Is the Interface

Users interact with the system through `default.yaml`. Code should never require editing Python files to try a new strategy combination.

**This means:**
- Filter functions read their parameters from kwargs, not hardcoded values
- Weighting functions read their parameters from kwargs, not hardcoded values
- The runner resolves function names from the config dynamically
- If a new feature can't be expressed through the config, extend the config schema — don't create a one-off script

## Before You Write Anything

1. **Check if it already exists.** Read through `filters.py`, `weights.py`, `metrics.py` before writing a new function. The function you need may already be there, possibly under a different name.

2. **Check if it belongs in an existing module.** A new metric goes in `metrics.py`. A new filter goes in `filters.py`. A new plot goes in `plots.py`. Do not create new files for single functions.

3. **Check if the config already supports it.** Changing a moving average window from 200 to 100 is a config change, not a code change. Adding a ticker to the universe is a config change. Switching from equal weight to momentum is a config change.

4. **Follow the existing pattern.** Every filter function has the same signature. Every weighting function has the same signature. Every metric function has the same signature. New additions must follow these signatures exactly, or the runner won't be able to call them.

## Anti-Patterns to Avoid

**Do not bypass the backtest engine.** `backtest.py` uses vectorbt for portfolio construction (target-percent orders, fees, cash sharing) and a custom loop for benchmark weight-drift. If you think it needs modification, check if the issue is in your filter or weighting function instead.

**Do not create standalone scripts that bypass the config.** Every analysis should be reproducible by running `scripts/run.py` with a config file. If you need a one-off analysis, add it as a config option, not as a separate script.

**Do not hardcode parameters.** If a function uses a magic number (lookback window, threshold, etc.), it should come from kwargs which come from the config. The only exception is truly universal constants (252 trading days per year, etc.).

**Do not compute metrics manually.** Use `metrics.py`. If you need a metric that doesn't exist, add it there. Do not compute Sharpe ratios inline in a script.

**Do not install new packages to avoid writing 10 lines of code.** pandas and numpy can handle most financial computations. matplotlib handles all the plotting we need. Adding a dependency for one function is not worth the maintenance cost.

**Do not create new directories.** The directory structure is fixed. Results go in `results/`. Data goes in `data/`. Source goes in `src/`. Tests go in `tests/`. If you think you need a new directory, you're probably putting something in the wrong place.

## Testing

### Unit Tests

Every function in `filters.py`, `weights.py`, and `metrics.py` must have tests. Tests live in `tests/` and follow the naming convention `test_<module>.py`.

Test the basics:
- Does the filter return a boolean DataFrame of the correct shape?
- Are weights zero for filtered-out assets?
- Do weights sum to <= 1.0 (or exactly 1.0 for renormalized)?
- Does Sharpe return a float? Is it positive for an upward equity curve?
- Does max_drawdown return a negative number?

### Regression Tests

The default config should produce known results. After any code change, verify:
- EW-cash full-period Sharpe ≈ 1.04 (within 0.05)
- EW-cash full-period MaxDD ≈ -13.8% (within 2%)
- EW-cash full-period CAGR ≈ 7.9% (within 1%)

If these numbers change, something is broken. The most common causes:
- Look-ahead bias (using today's signal for today's weights instead of yesterday's)
- Cash yield not being applied (or being applied incorrectly)
- Cost model changed
- Date alignment issue (weekends, holidays creating gaps)

### Smoke Tests

Before committing any change, run the default config end-to-end and verify:
- All strategies produce equity curves
- All metrics compute without errors
- All plots generate without errors
- No NaN values in output tables

## Data Handling

- **Cache downloaded data locally** in `data/`. Don't re-download on every run.
- **Handle missing data gracefully.** Some tickers don't go back to 2007. The system should use the longest common date range for each window, not fail.
- **Align all price series to the same trading calendar** before any computation. Missing dates (holidays, delistings) cause subtle bugs in rolling calculations.
- **Use adjusted close prices** for total return calculations. yfinance provides these.

## Git Hygiene

- Commit configs alongside code changes. A code change without the config that exercises it is untestable.
- Don't commit `data/` (cached downloads) or `results/` (generated output) to git. Add them to `.gitignore`.
- Exception: a `results/reference/` directory with the default config output can be committed as a regression baseline.
