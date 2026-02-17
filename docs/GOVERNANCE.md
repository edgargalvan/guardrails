# GOVERNANCE.md

## What This Document Is

This is the research discipline for the 200dma strategy framework. Read this before testing any new strategy variant. It contains the financial judgment and evaluation standards needed to assess results honestly.

If you are an AI assistant helping a user test strategies, this document defines how you think about results, what you recommend, and how you guide the user through the process.

---

## The Reference Baseline

Every new strategy is measured against **EW-cash** (equal weight, exit to cash):

| Metric | EW-cash |
|---|---|
| Sharpe | 1.04 |
| CAGR | 7.9% |
| Max Drawdown | -13.8% |
| Terminal value ($200K, 18yr) | $821K |

This is the bar. A new variant must meaningfully improve on these numbers to justify its added complexity.

**"Meaningful improvement" means:**
- Sharpe improvement > 0.03 (anything less is within noise on 18 years of monthly data)
- OR CAGR improvement > 1.5% without proportionally worse drawdowns
- OR MaxDD improvement > 3 percentage points without proportionally worse returns

If a variant improves Sharpe by 0.01 but adds a parameter, it's not worth it. If it improves CAGR by 2% but doubles the drawdown, it's just leveraging the same return stream. If it only looks good in one time window, it's overfit.

---

## What "Good" Looks Like in Finance

Calibrate your expectations:

| Sharpe | Interpretation |
|---|---|
| < 0.3 | Poor — barely compensating for risk |
| 0.3–0.5 | Mediocre — typical buy-and-hold equity |
| 0.5–0.7 | Decent — better than most retail portfolios |
| 0.7–1.0 | Good — institutional quality for a simple strategy |
| > 1.0 | Excellent — be skeptical, check for overfitting |

A Sharpe above 1.0 on a monthly-rebalanced ETF strategy should trigger suspicion, not celebration. Check: is the window cherry-picked? Is there look-ahead bias? Is it driven by one regime?

**Drawdowns matter more than CAGR for real investors.** An investor who experiences -40% will behave differently from one who experiences -12%, even if the long-run CAGR is the same. The strategy that keeps people invested beats the strategy that maximizes returns on paper.

---

## The Pre-Registration Workflow

When testing a new strategy variant, follow this process. If you are an AI assistant, walk the user through each step.

### Step 1: State the Hypothesis

Before touching any code, write down:
- What are you changing? (filter type, weighting scheme, universe, exit mode)
- Why do you think it will work? (economic reasoning, not "it might improve Sharpe")
- What do you expect to see? (directionally — better crisis performance? higher CAGR? lower drawdown?)

**AI guidance:** Help the user articulate the hypothesis. Ask "why would this work better than the simple baseline?" If the answer is "I don't know, I just want to try it," that's fine — but note that the prior probability of improvement is low. Most things we tested didn't beat the baseline.

### Step 2: Set Parameters Before Running

Every parameter must be fixed before seeing any results:
- Filter window (e.g., 200 days)
- Evaluation frequency (e.g., monthly)
- Weighting parameters (e.g., momentum lookback 252 days, skip 21 days, split 70/20/10)
- Exit mode (cash or renormalize)
- Universe (which tickers)

**AI guidance:** Suggest standard, well-established values rather than arbitrary ones. For moving averages: 50, 100, 200 are standard. For momentum lookback: 252 days (12 months) is the academic standard. If the user wants a non-standard value, ask for the economic justification.

### Step 3: One Run, All Windows

Run the strategy once with the pre-registered parameters across all configured time windows. Do not run it, look at results, tweak a parameter, and run again. That's curve-fitting.

If the user wants to test two parameter values (e.g., 100dma vs 200dma), that's fine — but both must be pre-registered, and the results should be evaluated for robustness (do they agree?) not cherry-picked (which is better?).

### Step 4: Evaluate Honestly

Compare against EW-cash across all windows. Report:
- Full-period metrics
- Per-window metrics (does it win everywhere, or just in one window?)
- Crisis behavior (2008, 2020, 2022)
- Where it's worse than EW-cash (every variant has a weak spot — find it)

**AI guidance:** Do not lead with the best result. Lead with the full picture. If the variant wins Sharpe in 4 of 5 windows but has 5% deeper drawdowns, say that upfront. If it only looks good post-2020, flag that as a short sample.

### Step 5: Make a Recommendation

Be direct. One of:
- **Adopt**: meaningfully better on risk-adjusted metrics, robust across windows, complexity is justified
- **Interesting but not worth it**: small improvement doesn't justify added complexity
- **Reject**: worse than baseline, or improvement is fragile/window-dependent
- **Investigate further**: promising signal but needs more analysis (specify what)

Most variants will be "interesting but not worth it" or "reject." That's normal. The baseline is surprisingly hard to beat.

---

## Lessons from Prior Testing

These findings are established. A new variant should not repeat these mistakes.

### Architecture > Signals

The biggest improvement in this project came not from a better signal but from a better capital allocation structure. Changing from "exit everything" to "exit per-asset to cash" dramatically improved Sharpe and MaxDD. Ten different signals (eigenvalues, credit spreads, AR residuals, etc.) were tested and none improved on a simple 200dma.

### Renormalization Concentrates Risk

When an asset drops below its 200dma and you redistribute its weight to survivors, you're concentrating into a shrinking opportunity set during a crisis. In 2022, this meant piling into assets that subsequently also fell. Exit to cash avoids this — cash is the only truly uncorrelated asset during a crisis.

Renormalization is a valid choice for growth-oriented investors, but understand the tradeoff: roughly double the CAGR for roughly double the drawdown.

### Continuous Taper Is Parameter-Fragile

We tested linear scaling near the 200dma (partial weight when close to the MA instead of binary 0/1). Two taper widths (5% and 10%) produced a 0.032 Sharpe divergence — the result depended heavily on the arbitrary choice of taper width. Binary exits (in or out) are simpler and don't require this choice.

If someone proposes a taper or gradual scaling, test two reasonable widths simultaneously. If they diverge by more than 0.03 Sharpe, the approach is fragile — keep the binary version.

### Momentum Adds Return, Not Sharpe

Momentum ranking among survivors (70/20/10 by trailing 12-month return) adds ~1.5% CAGR but costs ~0.06 Sharpe and ~3% more drawdown. It's a risk/return dial, not a free improvement. The momentum tilt has performed well in recent windows (2011+) but the full-period result is a slight Sharpe drag for more return.

### Managed Futures Are Architecturally Incompatible

DBMF (managed futures ETF) was below its 200dma 43% of all days. Managed futures strategies embed their own trend-following internally — applying a 200dma filter on top creates double-filtering that's too aggressive. The trend filter works for traditional trending assets (stocks, bonds, gold), not for strategies with fundamentally different return profiles.

### The 200dma Filter Is The Strategy

We tested inverse-volatility weighting vs equal weight with the same filter. The Sharpe difference was 0.028 — noise. The filter and exit mode determine performance. Everything else (weighting scheme, momentum ranking, vol targeting) is second-order.

---

## What to Be Skeptical About

When evaluating any new result, watch for:

**Window dependence.** Does it only look good post-2020? Post-2009? If the improvement disappears in the full-period comparison, it's likely driven by one regime.

**Parameter sensitivity.** Would a slightly different parameter give a meaningfully different result? If changing 200 to 180 swings the Sharpe by 0.05, the result is fragile.

**Complexity without commensurate gain.** A strategy that requires 3 extra parameters to improve Sharpe by 0.02 is worse than the simpler version. Complexity has a cost: more opportunities for overfitting, more things to go wrong, harder to explain, harder to stick with during drawdowns.

**Recency bias.** Gold has had a historic run since 2022. Any strategy that tilts toward gold will look brilliant in recent windows. That doesn't mean the tilt will persist.

**Survivorship in crisis.** A strategy that works in 2008 and 2022 but fails in 2020 (fast crash, fast recovery) has a blind spot. Test across all crisis types, not just the one that validates your hypothesis.

---

## Encouraging Exploration

Nothing in this document should prevent someone from testing a wild idea. RSI filter? Try it. 5-fund universe with crypto? Go for it. Inverse momentum? Why not.

The discipline is in the evaluation, not in the ideation. Test anything. But test it honestly: pre-register parameters, run once, compare against EW-cash across all windows, report the full picture, and be willing to conclude "the simple version is still better."

The most valuable finding in this project was not the winning strategy — it was the dozen variants that proved the simple strategy was already near-optimal. Each failed experiment narrowed the design space and increased confidence in the result.
