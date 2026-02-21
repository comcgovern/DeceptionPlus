# Predict+ — Mechanics Reference for Dynasty Index Integration

This document explains every aspect of how Predict+ works: what it measures, how it is calculated, what the output numbers mean, and how to correctly surface the metric to Dynasty Index users.

---

## Table of Contents

1. [What Predict+ Measures](#1-what-predict-measures)
2. [The Score Scale](#2-the-score-scale)
3. [Core Concept: Surprise](#3-core-concept-surprise)
4. [The Two-Model Architecture](#4-the-two-model-architecture)
5. [Feature Inputs](#5-feature-inputs)
6. [Unpredictability Ratio](#6-unpredictability-ratio)
7. [Standardization Formula](#7-standardization-formula)
8. [Alternative Metric: PPI](#8-alternative-metric-ppi)
9. [Data Pipeline Step by Step](#9-data-pipeline-step-by-step)
10. [Execution Modes](#10-execution-modes)
11. [Output Columns](#11-output-columns)
12. [Pitch Type Taxonomy](#12-pitch-type-taxonomy)
13. [Minimum Sample Requirements](#13-minimum-sample-requirements)
14. [Interpreting Extreme Scores](#14-interpreting-extreme-scores)
15. [Validated Correlations](#15-validated-correlations)
16. [Known Limitations](#16-known-limitations)
17. [Glossary](#17-glossary)

---

## 1. What Predict+ Measures

Predict+ quantifies **how difficult a pitcher's pitch selection is to predict** given full knowledge of game context, sequencing history, and batter tendencies.

It does **not** measure:
- Pitch quality (velocity, movement, spin)
- Pitch mix breadth (having many pitch types)
- Deception mechanics (arm angle, tunneling)

It **does** measure:
- Whether the pitcher's pitch choices follow learnable patterns
- Whether knowing the count, runners, batter, score, and previous pitch makes the next pitch predictable

A pitcher can have a two-pitch arsenal and still score extremely high if they use those two pitches without following recognizable situational rules. Conversely, a pitcher with six pitch types can score low if their choices are highly count-dependent and predictable.

---

## 2. The Score Scale

Predict+ is standardized to the same scale as ERA+ and wRC+:

| Score | Interpretation |
|-------|---------------|
| 115+ | Elite unpredictability — essentially impossible to anticipate |
| 110–114 | Highly unpredictable |
| 105–109 | Above average |
| 100 | League average |
| 95–99 | Slightly below average |
| 90–94 | Predictable |
| Below 90 | Highly predictable — follows recognizable patterns |

- **Mean = 100** (league average pitcher in the training population)
- **Standard deviation = 10** (one SD above average = 110)

The score is always relative to the training population and period. A score of 107 means the pitcher is 0.7 standard deviations more unpredictable than the average pitcher in that reference group.

---

## 3. Core Concept: Surprise

The mathematical engine behind Predict+ is **surprise** (also called negative log-likelihood):

```
Surprise(pitch | model) = -log( P_model(actual pitch) )
```

Where `P_model(actual pitch)` is the predicted probability the model assigned to the pitch that was actually thrown.

**Intuition:**
- Model says "I'm 80% sure a fastball is coming" → pitcher throws fastball → low surprise
- Model says "I'm 10% sure a curveball is coming" → pitcher throws curveball → high surprise
- Model says "I'm 50% sure a fastball is coming" → pitcher throws fastball → moderate surprise

**Mathematical properties that make this useful:**
- **Inversely proportional to probability:** rare choices produce higher values
- **Proper scoring rule:** the model is incentivized to give well-calibrated probabilities
- **Additive:** per-pitch surprises sum meaningfully across many pitches
- **Information-theoretically grounded:** the expected surprise over a distribution equals its entropy

We use natural logarithm (nats) rather than log base 2 (bits) for numerical stability.

---

## 4. The Two-Model Architecture

Predict+ does not simply measure raw surprise from one model. It compares two models to isolate **genuine** unpredictability from superficial effects like pitch mix diversity or count-driven tendencies.

### Full Model (Complex Predictor)

A multinomial logistic regression trained on 18 features representing full game context:
- Current count (balls, strikes, two-strike indicator, ahead-in-count indicator)
- Game situation (inning, outs, score differential, high-leverage indicator)
- Base-out state (8 configurations, runner-in-scoring-position flag)
- Batter profile (handedness, chase rate, in-zone contact, overall swing rate)
- Times through the order in this game
- Previous pitch type in this at-bat (or "NONE" if first pitch)

This model asks: *given everything I know, what pitch would a pattern-following pitcher throw here?*

### Baseline Model (Simple Predictor)

A simpler model using only basic features:
- Count (balls, strikes, two-strike indicator)
- Runner in scoring position (yes/no)
- Batter handedness and pitcher handedness

This model captures the most elementary situational patterns — count-based tendencies that all pitchers exhibit to some degree.

**Why two models?**

The ratio of their surprise values isolates what cannot be explained even with deep context:

| Scenario | Full Model | Baseline | Ratio | Meaning |
|----------|-----------|---------|-------|---------|
| Very predictable pitcher | Low surprise | Low surprise | < 1.0 | Full context helps prediction a lot |
| Average pitcher | Moderate surprise | Moderate surprise | ≈ 1.0 | Context helps, but not much more than basics |
| Very unpredictable pitcher | High surprise | High surprise | > 1.0 | Full context doesn't help; pitcher is genuinely random |

Pitchers with a ratio > 1.0 surprise the complex model *more* than the simple model — meaning knowing more about them actually doesn't help. They are situationally independent.

### Why Multinomial Logistic Regression?

The choice of algorithm for the full model is intentional:
- **Multi-class by design:** handles 14+ pitch types naturally, with probabilities summing to 1.0
- **Fast:** trains on 100k+ pitches in seconds, enabling daily runs
- **Interpretable:** coefficients can be sanity-checked
- **Conservative:** if a pitcher can fool even a simple logistic model, they are genuinely hard to predict

The model is an *intentional lower bound* — it deliberately doesn't use deep learning or tree ensembles, which would overfit and understate predictability. A pitcher who defies multinomial logistic regression is truly unpredictable.

---

## 5. Feature Inputs

### Categorical Features (treated as unordered factors)

| Feature | Description | Values |
|---------|-------------|--------|
| `last_pitch_type` | Previous pitch in this at-bat | FF, SL, CH, CU, SI, FC, FS, KC, ST, KN, FO, SV, CS, FT, OTHER, NONE |
| `base_state` | Base-out configuration | 0–7 (see encoding below) |
| `stand` | Batter handedness | L, R |
| `p_throws` | Pitcher handedness | L, R |
| `inning` | Inning number | Categorical integer |

**Base state encoding:** `base3 × 4 + base2 × 2 + base1`

| Value | Configuration |
|-------|--------------|
| 0 | Bases empty |
| 1 | Runner on 1st |
| 2 | Runner on 2nd |
| 3 | Runners on 1st and 2nd |
| 4 | Runner on 3rd |
| 5 | Runners on 1st and 3rd |
| 6 | Runners on 2nd and 3rd |
| 7 | Bases loaded |

### Continuous/Ordinal Features

| Feature | Description | Range |
|---------|-------------|-------|
| `balls` | Current ball count | 0–3 |
| `strikes` | Current strike count | 0–2 |
| `outs` | Current outs in the inning | 0–2 |
| `score_diff` | Home score minus away score | Typically -10 to +10 |
| `o_swing_pct` | Batter's chase rate (swings on pitches outside zone) | 0.0–1.0 |
| `z_contact_pct` | Batter's in-zone contact rate | 0.0–1.0 |
| `swing_pct` | Batter's overall swing rate | 0.0–1.0 |
| `chase_contact_pct` | Batter's contact rate on pitches outside zone | 0.0–1.0 |

### Binary Features

| Feature | Description | Condition |
|---------|-------------|-----------|
| `two_strikes` | 2-strike count indicator | `strikes == 2` |
| `ahead_in_count` | Hitter-favorable count | `balls > strikes` |
| `is_risp` | Runner in scoring position | Runner on 2nd or 3rd |
| `is_top` | Top of inning | `inning_topbot == "TOP"` |
| `high_leverage` | Late and close game | `inning >= 7 AND |score_diff| <= 3` |
| `n_thruorder_pitcher` | Times through order | 1, 2, or 3 (from Statcast) |

### Batter Metric Defaults

If insufficient history exists for a pitcher-batter pair, batter metrics default to `0.5`. This prevents NA-driven model failures while remaining neutral.

---

## 6. Unpredictability Ratio

After computing per-pitch surprise from both models, the pipeline aggregates to the pitcher level:

```
Mean_S_model    = mean( -log(P_model(actual_pitch))   ) over all test pitches
Mean_S_baseline = mean( -log(P_baseline(actual_pitch)) ) over all test pitches

Unpredictability_Ratio = Mean_S_model / Mean_S_baseline
```

**Interpretation guide:**

| Ratio | Meaning |
|-------|---------|
| > 1.0 | Complex model more surprised than simple model; pitcher resists pattern recognition |
| = 1.0 | Equal surprise; context neither helps nor hurts prediction |
| < 1.0 | Complex model less surprised; pitcher follows complex but learnable patterns |

A ratio of exactly 1.0 is rare in practice. Most pitchers fall in the 0.90–1.10 range.

**Why the ratio is robust:**
- Scale-invariant (both models see the same pitches)
- Unaffected by overall pitch difficulty or arsenal diversity
- Normalizes out count-level tendencies shared by both models

---

## 7. Standardization Formula

Raw ratios are transformed into the 100-point Predict+ scale using the training population's distribution:

```
μ    = mean(Unpredictability_Ratio across all pitchers in reference population)
σ    = standard_deviation(Unpredictability_Ratio across reference population)

Predict+ = 100 + 10 × ( (Unpredictability_Ratio - μ) / σ )
```

**Reference population for standardization:**

The reference `μ` and `σ` come from the **training period**, not the test period. This is deliberate:
- Anchors every analysis to a stable "league average" definition
- Scores from different test periods are comparable as long as they share the same training baseline
- Prevents the mean from drifting as sample sizes change between analyses

In daily mode, a pre-computed `baseline_params.rds` (generated from 100 random 50/50 splits over 2+ years of data) provides fixed `μ` and `σ` values, ensuring long-run comparability.

---

## 8. Alternative Metric: PPI

Alongside Predict+, the pipeline computes an alternative metric called the **Pitch Predictability Index (PPI)**:

```
PPI = 1 - (Mean_S_model / Mean_S_baseline)
    = 1 - Unpredictability_Ratio
```

| PPI Value | Meaning |
|-----------|---------|
| +1.0 | Model is always completely surprised (perfectly unpredictable) |
| 0.0 | Both models equally surprised (average) |
| -1.0 | Baseline always more surprised (complex model captures all patterns) |

PPI lives on the range [-1, 1] and is clamped to that range. It is an intuitive complement to Predict+ for users who prefer a bounded scale.

**Dynasty Index recommendation:** Surface Predict+ as the primary metric (familiar scale, comparable to ERA+) and PPI as a secondary detail for users who want the raw ratio information.

---

## 9. Data Pipeline Step by Step

### Step 1: Data Acquisition

Source: MLB Statcast via the [sabRmetrics](https://github.com/tbuffington7/sabRmetrics) R package (MLB) and direct Baseball Savant API (AAA).

Data is cached locally as `.Rds` files named:
```
cache/savant_raw_{start_date}_{end_date}_{game_type}_{level}.Rds
```

Supported game types: `R` (regular season), `P` (playoffs), `W` (World Series), `S` (spring training).

### Step 2: Feature Engineering

Raw Statcast pitch-level data is transformed:

1. **Pitch type canonicalization** — standardizes all pitch codes to the 14-type taxonomy (see Section 12)
2. **Count features** — `two_strikes`, `ahead_in_count` computed from `balls` and `strikes`
3. **Base state encoding** — 8-integer encoding from runner flags
4. **Batter metrics** — computed from rolling historical data for the pitcher-batter pair; defaults to 0.5 if insufficient history
5. **Sequence feature** — `last_pitch_type` is the previous pitch in the current at-bat, or "NONE" for the first pitch of each plate appearance
6. **Times through order** — pulled directly from Statcast `n_thruorder_pitcher`

### Step 3: Feature Validation

Before model training, the pipeline removes features that would cause fitting errors:
- Categorical features with only one level
- Numeric features with zero variance

Missing values are handled as follows:
- Categorical: `NA` → `"UNK"` factor level
- Numeric: imputed with median (or 0 if all values are `NA`)

### Step 4: Data Splitting

The pipeline supports three splitting strategies:

| Mode | Training Data | Test Data | Use Case |
|------|--------------|-----------|----------|
| Temporal (default) | `train_start` to `train_end` | `test_start` to `test_end` | Regular season → playoffs comparison |
| Random 50/50 | Random half of each pitcher's pitches | Opposite half | Stable baseline computation |
| Same-period | Full period | Full period | In-sample validation |

### Step 5: Model Training

**Full season mode** (`run.R`): one multinomial model trained on the entire league's training-period data.

**Daily mode** (`run_daily.R`): one individual model per pitcher, trained on that pitcher's last N pitches (default: 500) drawn from historical data.

Model configuration:
```r
nnet::multinom(
  pitch_class ~ [all features],
  data = training_data,
  maxit = 500,
  trace = FALSE
)
```

No regularization is applied — the model is meant to fully fit whatever patterns exist in training data.

Typical convergence: 50–100 iterations with 1,000+ pitches.

### Step 6: Prediction and Surprise Calculation

For every pitch in the test period, the pipeline:
1. Calls `predict(model, newdata = test_pitch, type = "probs")` to get a probability vector over all pitch types
2. Looks up the probability assigned to the actual pitch thrown
3. Computes `surprise_model = -log(max(p_actual, 1e-12))` (the floor prevents `log(0)`)
4. Repeats with the baseline model to get `surprise_baseline`

### Step 7: Pitcher-Level Aggregation

```
pitcher_level:
  n_pitches_test        = count of test pitches for this pitcher
  mean_surp_model       = mean(surprise_model across test pitches)
  mean_surp_base        = mean(surprise_baseline across test pitches)
  unpredictability_ratio = mean_surp_model / mean_surp_base
```

### Step 8: Standardization

Using the reference `μ` and `σ` from the training population:
```
predict_plus = 100 + 10 × ((unpredictability_ratio - μ) / σ)
```

### Step 9: Name Resolution

Pitcher names are resolved via the MLB Stats API in batches of 100 IDs:
```
GET https://statsapi.mlb.com/api/v1/people?personIds=id1,id2,...
```

Results are cached in `cache/mlbam_name_cache.csv`. If a name cannot be resolved, the pitcher is labeled `Pitcher_{mlbam_id}`.

---

## 10. Execution Modes

### Full Season Analysis (`run.R`)

Designed for end-of-season or custom-range analysis. Trains one league-wide model on the specified training period, then evaluates all pitchers against the test period.

Key configuration:
```r
MIN_TEST_PITCHES  <- 100   # Minimum test-period pitches to include a pitcher
MIN_TOTAL_PITCHES <- 250   # Minimum combined pitches
SPLIT_METHOD      <- "temporal"  # or "random"
BASELINE_TYPE     <- "conditional"  # or "marginal" or "hybrid"
TRAIN_LEVEL       <- "MLB"  # or "AAA"
```

### Daily Analysis (`run_daily.R`)

Designed for daily production runs. For each pitcher who threw pitches on the target date:
1. Pull their last 500 pitches as individual training data
2. Train a per-pitcher multinomial model
3. Evaluate today's pitches
4. Standardize using the fixed `baseline_params.rds`

Command-line usage:
```bash
Rscript run_daily.R [YYYY-MM-DD] --level [MLB|AAA] --n_history 500 --min_history 100
```

Output path: `output/{year}/{month}/{day}.csv`

The daily output also includes `role` (starter or reliever) and `status` columns (see Section 11).

### Baseline Computation (`compute_baseline.R`)

Run once (or periodically) to establish the reference distribution for standardization:
1. Load 2+ years of historical data
2. Run 100 independent random 50/50 splits
3. For each split, train per-pitcher models and evaluate on the held-out half
4. Compute `μ` and `σ` of the resulting unpredictability ratios
5. Save to `baseline_params.rds`

---

## 11. Output Columns

### Core Columns (all modes)

| Column | Type | Description |
|--------|------|-------------|
| `pitcher_id` | integer | MLB Advanced Media (MLBAM) player ID |
| `pitcher_name` | character | Full name resolved from MLB Stats API |
| `total_pitches` | integer | Pitches across training + test periods combined (may double-count overlapping periods) |
| `n_pitches_test` | integer | Pitches in the test period used for evaluation |
| `mean_surp_model` | numeric | Average per-pitch surprise from the full model (nats) |
| `mean_surp_base` | numeric | Average per-pitch surprise from the baseline model (nats) |
| `ppi` | numeric | Pitch Predictability Index: `1 - (mean_surp_model / mean_surp_base)`, clamped to [-1, 1] |
| `unpredictability_ratio` | numeric | `mean_surp_model / mean_surp_base` |
| `predict_plus` | numeric | Final standardized score (mean = 100, SD = 10) |

### Additional Columns (daily mode only)

| Column | Type | Description |
|--------|------|-------------|
| `role` | character | `"starter"` or `"reliever"` based on pitch count threshold |
| `status` | character | Why a pitcher may be excluded (see below) |

**Status values:**

| Status | Meaning |
|--------|---------|
| `"evaluated"` | Normal result; pitcher had enough history and test pitches |
| `"debut_no_history"` | Pitcher has no prior MLB Statcast data |
| `"insufficient_history"` | Fewer than `min_history` pitches in historical record |
| `"insufficient_test_pitches"` | Threw too few pitches today to produce a reliable estimate |

---

## 12. Pitch Type Taxonomy

All Statcast pitch type codes are canonicalized to the following 14 types before any modeling:

| Code | Pitch Name |
|------|-----------|
| `FF` | 4-seam fastball |
| `SI` | Sinker |
| `FT` | 2-seam fastball |
| `FC` | Cutter |
| `FS` | Splitter |
| `CH` | Changeup |
| `SL` | Slider |
| `CU` | Curveball |
| `KC` | Knuckle curve |
| `SV` | Slurve |
| `CS` | Slow curve / curveball-slider variant |
| `ST` | Sweeper |
| `KN` | Knuckleball |
| `FO` | Forkball |
| `OTHER` | Any unrecognized or unusual pitch code |

Multiple raw Statcast labels map to the same canonical code (e.g., sweeper maps to `SL` in some historical data). Canonicalization uses case-insensitive pattern matching.

---

## 13. Minimum Sample Requirements

| Context | Minimum | Rationale |
|---------|---------|-----------|
| Historical training data (per-pitcher daily mode) | 100 pitches | Below this, model coefficients are unreliable |
| Test period pitches | 100 pitches (full season), 30 pitches (daily) | Below this, mean surprise estimates are noisy |
| Total pitches to appear in output | 250 (full season) | Prevents outlier scores from one-pitch samples |
| Model convergence (reliable) | 1,000+ pitches | Stable coefficients across all features |

Pitchers below minimum thresholds are excluded from the output CSV entirely in full-season mode, or flagged with an appropriate `status` value in daily mode.

---

## 14. Interpreting Extreme Scores

### High Predict+ (115+): What's Happening

- **Situational independence**: Pitch selection does not shift based on count, runners, or score
- **Sequence independence**: Previous pitch does not predict the next
- **Balanced usage in "obvious" situations**: Doesn't default to fastball on 3-0, doesn't always throw offspeed with 2 strikes
- **Not necessarily a large arsenal**: Two-pitch pitchers can be elite if their two pitches appear without detectable rules

**Real example:** A reliever with only a fastball and slider who alternates them seemingly at random regardless of count, batter, or score will have a very high ratio — neither model can anticipate which he will throw.

### Low Predict+ (85 and below): What's Happening

- **Strong count patterns**: e.g., always throws fastball 0-0, always throws offspeed with 2 strikes
- **Strict sequencing**: e.g., always follows fastball with breaking ball
- **Situation-dependent patterns**: dramatically shifts pitch mix with runners on base
- **Expected edge cases**: position players pitching (tiny arsenal, no strategy), knuckleballers (throws one pitch), pitchers with depleted arsenals due to injury

**Real examples from 2025 data:** Matt Waldron (knuckleballer — essentially one pitch), Enrique Hernandez (position player — predictable by necessity).

---

## 15. Validated Correlations

The following relationships have been observed in 2025 Statcast data, holding for starters with 1,500+ pitches in a season:

| Outcome | Direction | Interpretation |
|---------|-----------|---------------|
| xFIP | Negative | Higher Predict+ → lower xFIP → better performance |
| SIERA | Negative | Higher Predict+ → lower SIERA → better performance |
| SwStr% | Positive | Higher Predict+ → more swinging strikes |
| K% | Positive | Higher Predict+ → more strikeouts |

Effect sizes are meaningful but not overwhelming — unpredictability is one factor among many. The correlations persist after controlling for raw pitch quality metrics.

**Role-specific findings:**
- **Starters**: Effect is strongest, likely because facing the same batter 2–3 times in a game amplifies the value of unpredictability
- **Relievers**: Effect is most pronounced in high-leverage appearances; smaller effect in blowouts

---

## 16. Known Limitations

| Limitation | Impact | Future Direction |
|-----------|--------|-----------------|
| No catcher game-calling effects | Catcher influence on pitch selection is unmeasured | Pitcher-catcher dyad analysis planned |
| Linear model only | Non-linear interaction effects not captured | XGBoost / random forest comparison in progress |
| Single previous pitch | Only one-pitch sequencing (not multi-pitch patterns) | Multi-pitch sequence features planned |
| No leverage weighting | A walk-off strikeout and a garbage-time pitch count equally | Leverage-weighted surprise being evaluated |
| No platoon splits | Score is averaged across vs. LHH and vs. RHH | Separate platoon scores planned |
| Training/test temporal assumption | Assumes patterns learned in training persist to test; may drift if pitcher makes in-season adjustments | Rolling window models being tested |

---

## 17. Glossary

| Term | Definition |
|------|-----------|
| **Surprise** | `-log(P)` where P is the predicted probability of the actual pitch. Higher = more unexpected |
| **Unpredictability Ratio** | Mean model surprise divided by mean baseline surprise for a given pitcher |
| **Full model** | Multinomial logistic regression using 18 context features |
| **Baseline model** | Simpler model using only count and handedness features, or frequency table lookup |
| **Marginal baseline** | Pitch type frequencies from the overall training set (no situational conditioning) |
| **Conditional baseline** | Pitch type frequencies within specific count/situation cells |
| **Hybrid baseline** | Conditional when cell has ≥5 observations, marginal otherwise |
| **PPI** | Pitch Predictability Index: `1 - Unpredictability_Ratio`, range [-1, 1] |
| **Predict+** | Standardized metric: `100 + 10 × z-score of Unpredictability_Ratio` |
| **MLBAM ID** | MLB Advanced Media player identifier used to join to other Statcast data |
| **Times through order** | How many times a pitcher has faced a particular batter in the current game |
| **Canonical pitch type** | Standardized pitch code from the 14-type taxonomy used internally |
| **Training period** | Date range used to fit the model (learn patterns) |
| **Test period** | Date range used to evaluate the model (measure surprise) |
| **Reference population** | Set of pitchers whose ratios define the `μ` and `σ` for standardization |

---

*Source: [Predict+ on GitHub](https://github.com/comcgovern/PredictPlus) — Conor McGovern, 2025*
*For commercial licensing inquiries: comcgovern@gmail.com*
