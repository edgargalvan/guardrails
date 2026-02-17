#!/usr/bin/env python3
"""Entry point: load config, run comparison, save results, generate plots.

Usage:
    python scripts/run.py                     # uses configs/default.yaml
    python scripts/run.py configs/custom.yaml  # uses custom config
"""

import json
import logging
import os
import sys

import quantstats as qs
import yaml

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import compare, data, plots as plots_module

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main(config_path: str = None):
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")

    # Load config
    logger.info("Loading config from %s", config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    settings = config.get("settings", {})
    initial_capital = settings.get("initial_capital", 200_000)
    reference = settings.get("reference", "EW-cash")

    # Load prices
    logger.info("Loading price data...")
    prices = data.load_prices(config)
    logger.info("Loaded %d days x %d tickers (%s to %s)",
                len(prices), len(prices.columns),
                prices.index[0].date(), prices.index[-1].date())

    # Run comparison
    logger.info("\n" + "=" * 80)
    logger.info("  RUNNING COMPARISON")
    logger.info("=" * 80)
    results = compare.run_comparison(config, prices)

    # Output directory — configurable to avoid stomping previous runs
    output_dir = os.path.join(
        PROJECT_ROOT, settings.get("output_dir", "results")
    )
    os.makedirs(output_dir, exist_ok=True)

    # Print summary tables
    logger.info("\n" + "=" * 100)
    logger.info("  RESULTS SUMMARY")
    logger.info("=" * 100)

    summary = {}
    for window_label, window_data in results.items():
        logger.info(f"\n--- {window_label} ---")
        names = list(window_data.keys())
        if not names:
            continue

        # Header
        metric_names = settings.get("metrics", ["sharpe", "cagr", "max_drawdown"])
        header = f"  {'Portfolio':<20}"
        for m in metric_names:
            header += f" {m:>14}"
        logger.info(header)
        logger.info(f"  {'-' * (20 + 15 * len(metric_names))}")

        summary[window_label] = {}
        for name in names:
            m = window_data[name]["metrics"]
            row = f"  {name:<20}"
            for metric in metric_names:
                val = m.get(metric)
                if val is None:
                    row += f" {'N/A':>14}"
                elif metric in ("cagr", "annual_vol", "max_drawdown", "worst_month", "worst_year"):
                    row += f" {val:>13.1%}"
                elif metric in ("sharpe", "sortino", "calmar"):
                    row += f" {val:>14.3f}"
                elif metric in ("ulcer_index",):
                    row += f" {val:>14.4f}"
                elif metric in ("upside_capture", "downside_capture"):
                    row += f" {val:>14.2f}"
                else:
                    row += f" {val:>14.4f}"
            logger.info(row)
            summary[window_label][name] = m

    # Save summary JSON
    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("\nSaved summary to %s", json_path)

    # Generate plots
    plot_types = settings.get("plots", [])
    if plot_types:
        logger.info("\nGenerating plots...")
        for window_label in results:
            window_dir = os.path.join(output_dir, _slugify(window_label))
            os.makedirs(window_dir, exist_ok=True)
            for plot_type in plot_types:
                plot_fn = getattr(plots_module, plot_type, None)
                if plot_fn is None:
                    logger.warning("Unknown plot type: %s", plot_type)
                    continue
                path = plot_fn(results, window_label, settings, window_dir)
                if path:
                    logger.info("  Saved %s", path)

    # Generate per-strategy quantstats tearsheets
    generate_tearsheets = settings.get("generate_tearsheets", False)
    tearsheet_benchmark = settings.get("tearsheet_benchmark", "SPY-BH")
    if generate_tearsheets:
        logger.info("\nGenerating quantstats tearsheets...")
        for window_label, window_data in results.items():
            window_dir = os.path.join(output_dir, _slugify(window_label))
            os.makedirs(window_dir, exist_ok=True)
            # Use configured benchmark for tearsheet comparisons
            bench_ret = None
            if tearsheet_benchmark in window_data:
                bench_ret = window_data[tearsheet_benchmark]["returns"]
            for name, strat_data in window_data.items():
                # Skip if strategy IS the benchmark (identical comparison)
                if name == tearsheet_benchmark:
                    continue
                tearsheet_path = os.path.join(
                    window_dir, f"{_slugify(name)}_tearsheet.html"
                )
                try:
                    qs.reports.html(
                        strat_data["returns"],
                        benchmark=bench_ret,
                        output=tearsheet_path,
                        title=f"{name} — {window_label}",
                        strategy_title=name,
                        benchmark_title=tearsheet_benchmark,
                    )
                    logger.info("  Saved tearsheet: %s", tearsheet_path)
                except Exception as e:
                    logger.warning("  Tearsheet failed for %s: %s", name, e)

    # Regression check
    logger.info("\n" + "=" * 80)
    logger.info("  REGRESSION CHECK")
    logger.info("=" * 80)
    _regression_check(results, reference)

    logger.info(f"\n{'=' * 80}")
    logger.info("  DONE")
    logger.info(f"{'=' * 80}")


def _regression_check(results: dict, reference: str):
    """Check reference strategy metrics against known baselines."""
    # Prefer "Full Period" window; fall back to first window with reference
    label = None
    if "Full Period" in results and reference in results["Full Period"]:
        label = "Full Period"
    else:
        for lbl, window_data in results.items():
            if reference in window_data:
                label = lbl
                break

    if label is None:
        logger.info(f"\n  WARNING: {reference} not found in any window")
        return

    m = results[label][reference]["metrics"]
    s = m.get("sharpe", 0)
    dd = m.get("max_drawdown", 0)
    c = m.get("cagr", 0)
    logger.info(f"\n  {reference} in '{label}':")
    logger.info(f"    Sharpe:  {s:.3f}  (expected ~1.04)")
    logger.info(f"    MaxDD:   {dd:.1%}  (expected ~-13.8%)")
    logger.info(f"    CAGR:    {c:.1%}  (expected ~7.9%)")


def _slugify(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_")


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_file)
