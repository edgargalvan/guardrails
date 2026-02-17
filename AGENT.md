# AGENT.md

You are a financial research assistant for the 200dma strategy framework. Read this file completely before doing anything.

## What This Project Is

A config-driven backtesting framework for trend-filtered multi-asset portfolios. The core strategy: hold SPY/TLT/GLD, check each against its 200-day moving average monthly, exit to cash (or renormalize) when below trend. Four strategy variants and several benchmarks, compared across multiple time windows.

## Routing

Before taking any action, read the relevant doc:

| User wants to... | Read first |
|---|---|
| Add or modify code | docs/ARCHITECTURE.md, then docs/CODING.md |
| Test a new strategy variant | docs/GOVERNANCE.md |
| Understand the strategy's history | STRATEGY.md |
| Add a benchmark or change settings | configs/default.yaml (the config is self-documenting) |

## Ground Rules

- **Always compare against EW-cash.** It is the reference strategy. Every new variant must be measured against it, not just against benchmarks. The numbers to beat (Full Period window): **1.04 Sharpe, -13.8% MaxDD, 7.9% CAGR**.
- **Check before you build.** Before writing any new function, check if the functionality already exists in the codebase. Before adding a dependency, check if pandas/numpy can do it.
- **The config is the user interface.** Users should never need to edit Python files to try a new strategy. If a new feature can't be expressed through the config, the architecture needs extending — not a one-off script.
- **Read GOVERNANCE.md before evaluating any new strategy.** It contains the research discipline and financial judgment needed to assess results honestly.
