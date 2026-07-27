# ⚾ Deception+

If you’re anything like me – which, if you’re looking at a GitHub repository for a new sabermetric model, I have to assume you are – then you have a particular tic that comes out when you're watching a baseball game. It probably happens so often and so automatically that you fail to register it, thinking it’s something everyone does, but I'm told that it's not.

I’m talking, of course, about guessing the next pitch that gets thrown. Whether you’re doing it out loud (and, like me, annoying your kids, spouse, or friends) or just in your head, guessing that next pitch is as much a part of the routine as singing “Take Me Out to the Ballgame” in the middle of the seventh.

This sequence of events (and my temporary idleness in the wake of the October 2025 government shutdown that left me furloughed from my day job) led to an investigation of whether certain pitchers were more predictable in their selection in any given situation than others.

**Deception+** is an R script that quantifies how unpredictable a pitcher's pitch selection is by comparing machine learning predictions against actual pitch choices. Higher scores indicate pitchers who successfully defy pattern recognition — even when sophisticated models know their history, tendencies, and game situation.

## What It Measures

Traditional scouting tells us that unpredictability matters. Hitters and scouting departments study video, memorize tendencies, and look for patterns. But *how much* does unpredictability matter, and how do we measure it objectively?

Deception+ uses an information-theoretic approach: we train a multinomial logistic regression model on each pitcher's historical data, including game context (count, outs, runners, batter handedness, previous pitch), and then measure how "surprised" the model is by the pitcher's actual choices. We compare this surprise against a baseline model to isolate genuine unpredictability from simple pitch mix diversity.

### Two metrics, two questions

There are two honest ways to ask "is this pitcher unpredictable," and they don't
give the same answer. Both are reported, both scaled to **100 = league average,
SD = 10**.

| | **Surprise+** | **Deception+** |
|---|---|---|
| Asks | Of all the uncertainty this pitcher's arsenal *could* create, how much survives once you know the situation? | Does this pitcher defy prediction *beyond* what the count and handedness already give away? |
| Built from | model surprise ÷ log(arsenal size) | model surprise ÷ baseline surprise |
| 110+ | next pitch is near a coin flip among their offerings | breaks patterns a situational model can't learn |
| 90− | the situation gives the pitch away | textbook, situation-driven selection |
| Rewards big arsenals? | barely (R² ≈ 0.27 vs pitch-type count) | no (R² ≈ 0.03) |
| **Usable on one outing?** | **yes** (reliability ≈ 0.92 at 20 pitches) | **no** (≈ 0.16 at 20 pitches; ≈ 0.79 at 1500) |

The difference in that last row is the important one. Deception+ is the more
conceptually pure measure — it is almost perfectly independent of how many pitches
you throw, which is why a two-pitch reliever can top it. But it is a **season-scale
statistic**: on a single 20-pitch outing roughly five-sixths of its spread is
estimation noise, so ranking one day by it ranks noise.

So: **daily leaderboards and graphics sort on Surprise+; Deception+ is the
season-level number.** Both appear in every output file. Run
`Rscript scripts/compare_metrics.R` to reproduce these figures, or
`--real <season>` to check them against your own cached data.

A note on what Deception+ deliberately *excludes*: because its baseline already
knows the count, a pitcher whose only tell is count-based is scored as average —
that predictability is controlled away rather than counted. Surprise+ has no such
exclusion, which is part of why it reads more naturally.

🧢 [**See the 2025 regular season data here!**](https://public.tableau.com/app/profile/mcgov36/viz/Predict2025/SingleDash)

## Why It Matters

Initial validation shows meaningful correlations with pitcher performance. Higher Deception+ for starters (1500+ pitches in a season) is associated with a lower xFIP and SIERA and a higher swinging strike rate and strikeout rate.

This suggests there's strategic value in unpredictability, not just randomness, though there's also a lot of noise there. The effect exists even after controlling for pitch quality metrics.

Unpredictability appears to matter most when:
- Facing the same batter multiple times (starters)
- In high-leverage situations (relievers)

![Deception+ vs SwStr%](https://gcdnb.pbrd.co/images/eXQQXoRih26i.png)

## Quick Start

### Installation

```r
# Required packages
install.packages(c("dplyr", "tidyr", "purrr", "stringr", "lubridate",
                   "nnet", "readr", "tibble", "forcats", "jsonlite", "httr"))

# For data access
devtools::install_github("saberpowers/sabRmetrics")
```

### Basic Usage

```r
source("pitch_ppi.R")

# Analyze 2025 regular season
# Train on Mar-Aug, evaluate September
result <- train_and_save(
  train_start = "2025-03-01",
  train_end   = "2025-08-31",
  test_start  = "2025-09-01",
  test_end    = "2025-09-30",
  min_total_pitches = 50,
  out_model = "models/ppi_model.rds",
  out_ppi   = "output/pitcher_ppi.csv"
)

# View top unpredictable pitchers
head(result$pitcher_ppi, 10)
```

### Analyzing Specific Periods

You can train and test on any periods — same, overlapping, or separate:

```r
# Train on regular season, test on playoffs
result <- train_ppi(
  train_start = "2025-03-01",      # Training period
  train_end   = "2025-09-30",
  test_start  = "2025-10-01",      # Test period
  test_end    = "2025-11-05",
  test_game_type = "P"             # R = regular season, P = playoffs, W = World Series
)
```

Or split one period at random instead of by date:

```r
result <- train_ppi(
  train_start  = "2025-03-01",
  train_end    = "2025-09-30",
  split_method = "random",         # 50/50 per pitcher; test dates are ignored
  random_seed  = 42
)
```

### AAA Analysis

```r
# Analyze Triple-A data. Level (MLB/AAA) and game_type (R/P/S/W) are separate
# knobs — game_type selects regular season vs. playoffs, not the league.
result <- train_and_save(
  train_start = "2025-04-01",
  train_end   = "2025-08-15",
  test_start  = "2025-08-16",
  test_end    = "2025-09-15",
  train_level = "AAA",
  test_level  = "AAA",
  min_total_pitches = 50
)
```

## How It Works

### 1. Model Training

We train a multinomial logistic regression model to predict pitch type using:
- **Count state**: the joint count as a 12-level factor (`0-0` … `3-2`), not two
  numeric terms — 3-0 and 0-2 are not two steps along one axis
- **Game situation**: inning, outs, runners on base, score differential
- **Batter context**: handedness, chase rate, contact tendencies (computed from
  the *training* window only)
- **Sequence**: previous pitch type, what happened on it (ball / called strike /
  whiff / foul / in play), where it was located (in or out of the zone), and the
  pitch two back
- **Catcher**: who was calling the game
- **Workload**: pitch number within the appearance
- **Times through order**: how often batter has faced this pitcher today

Every predictor is something known *before* the pitch is released. Statcast
fields describing the pitch itself — velocity, movement, location, outcome — are
deliberately excluded: they would predict the pitch type nearly perfectly and
measure nothing.

### 2. Surprise Calculation

For each pitch in the test period, we calculate **surprise** = -log(predicted probability of actual pitch). This measures how unexpected each pitch choice was.

### 3. Baseline Comparison

We compare the full model's surprise against a simpler **baseline model** that uses only count and batter handedness. This isolates true unpredictability from simple pitch mix diversity.

**Unpredictability Ratio** = Model Surprise / Baseline Surprise

Ratios > 1 mean the pitcher remains unpredictable even when accounting for game context. Ratios < 1 mean situational patterns explain most pitch selection.

For that reading to hold, the full model has to be able to reproduce the baseline —
otherwise a ratio above 1 can just mean the full model is the weaker of the two.
The baseline is a saturated cross-tab over count and handedness, so those must
reach the full model as factors. See
[METHODOLOGY.md](METHODOLOGY.md#the-model-must-nest-the-baseline).

Note also what the baseline *removes*: because it already knows the count, a
pitch pattern driven by the count is controlled away rather than counted as
predictability. Deception+ measures unpredictability **beyond count and
handedness**, not in absolute terms. A pitcher whose tells live in outs, base
state, or sequencing moves the score sharply; one whose tells are purely
count-based moves it much less.

### 4. Standardization

Following the Pitching+ standard, we convert the ratio to **Deception+** with mean = 100, SD = 10 for easy interpretation.

## Features

- **Flexible period selection**: Train and test on any date ranges
- **MLB and AAA support**: Analyze both major and minor league data
- **Cached downloads**: Baseball Savant data cached locally to avoid re-downloads
- **Multiple baseline models**: Choose between marginal, conditional, or hybrid baselines

## Output

The main output (`pitcher_ppi.csv`) includes:

| Column | Description |
|--------|-------------|
| `pitcher_id` | MLB player ID |
| `pitcher_name` | Full name from StatsAPI |
| `total_pitches` | Total pitches thrown in training and testing windows (note that if there's overlaps, this will duplicate) |
| `n_pitches_test` | Pitches in test period used for evaluation |
| `mean_surp_model` | Average surprise from full model |
| `mean_surp_base` | Average surprise from baseline model |
| `ppi` | Pitch Predictability Index (1 - ratio, range: -1 to 1) |
| `unpredictability_ratio` | Model surprise / baseline surprise |
| `surp_excess` | Model surprise **minus** baseline surprise, in nats. Same comparison as the ratio on a difference scale. Prefer it when the baseline surprise is small: for a pitcher who throws one pitch 99% of the time, both surprises are near zero and their *ratio* swings wildly on rounding-level differences, while the difference correctly reports "no meaningful gap." |
| `n_classes` | Size of that pitcher's pitch vocabulary |
| `normed_surprise` | Model surprise ÷ log(`n_classes`) — the raw input to Surprise+. ~1.0 means the next pitch is close to a coin flip among their own offerings. |
| `deception_plus` | **Deception+**: scaled `unpredictability_ratio` (mean=100, SD=10). Season-scale — see the two-metric table above. |
| `surprise_plus` | **Surprise+**: scaled `normed_surprise` (mean=100, SD=10). Reliable on a single outing; this is what the daily rankings sort on. |

The daily output additionally carries `role` and `status`. Rows with a `status`
other than `evaluated` are pitchers who appeared but could not be scored — a debut
with no history, too little history, or too few pitches on the day — and their
metric columns are intentionally blank.

## Advanced Usage

### Custom Features

```r
# Specify which features to include.
# Note: the engineered feature is `times_through_order` — `n_thruorder_pitcher`
# is the raw Statcast column it is derived from and is not a model feature.
result <- train_and_save(
  train_start = "2025-03-01",
  train_end   = "2025-08-31",
  test_start  = "2025-09-01",
  test_end    = "2025-09-30",
  feature_names = c("count", "high_leverage", "times_through_order", "outs",
                    "score_diff", "base_state", "is_risp",
                    "stand", "p_throws", "last_pitch_type",
                    "o_swing_pct", "z_contact_pct", "swing_pct"),
  baseline_keys = c("count", "is_risp", "stand", "p_throws")
)
```

`count` is the joint 12-level count factor (`0-0` … `3-2`). Prefer it over
separate numeric `balls` and `strikes`: the conditional baseline is a saturated
cross-tab over its keys, so a model that is merely *linear* in balls and strikes
is less expressive than the baseline it is scored against, and the ratio then
rises with predictability instead of falling. `check_baseline_nesting()` warns if
your feature set and baseline keys fall into that trap.

```r
# This configuration warns — 'balls'/'strikes' cell the baseline but reach the
# model as numerics, so the comparison is not apples to apples.
train_and_save(..., feature_names = c("balls","strikes",...),
                    baseline_keys = c("balls","strikes","stand"))
```

### Scoring Controls

Three parameters govern how probabilities become surprise. The defaults are
sane; change them only if you know why.

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `decay` | `1e-4` seasonal, `0.01` per-pitcher | Weight decay (L2 penalty) on the multinomial fit. Small per-pitcher samples separate easily; without a penalty the fit emits probabilities of 0 and 1 and `-log(p)` stops measuring surprise. |
| `prob_shrinkage` | `0.02` | Prior mass mixed into predicted probabilities before `-log()`. Keeps model and baseline surprise on the same floor so their ratio stays meaningful. |
| `standardize` | `"test"` | Population whose μ/σ anchor the Deception+ scale. `"test"` makes the output actually have mean 100 / SD 10. `"train"` gives a test-period-independent anchor but is measured in-sample and therefore optimistically biased. |
| `baseline_alpha` | `5` | Pseudo-count mass smoothing the baseline's conditional cells toward the marginal pitch mix. It sits in the denominator of every score; picked by held-out likelihood the optimum is ~2 and the curve is flat from 1 to 5. |

### Command Line

```bash
Rscript pitch_ppi.R \
  --train_start 2025-03-01 \
  --train_end   2025-08-31 \
  --test_start  2025-09-01 \
  --test_end    2025-09-30 \
  --min_total_pitches 50 \
  --train_game_type R \
  --test_game_type R \
  --out_model models/ppi_model.rds \
  --out_ppi   output/pitcher_ppi.csv
```

Run with no arguments to see the full option list, including `--split_method`,
`--baseline_type`, `--standardize`, `--decay` and `--prob_shrinkage`.

## Data Sources

- **MLB Statcast data**: Via [sabRmetrics](https://github.com/saberpowers/sabRmetrics) package
- **AAA Statcast data**: Direct Baseball Savant API integration using the sabRmetrics source code
- **Pitcher names**: MLB Stats API with local caching

## Technical Notes

### Standardization Baseline (`baseline_params.rds`)

The daily pipeline standardizes against fixed μ/σ stored in `baseline_params.rds`,
so that a score means the same thing on Tuesday as it did in April. That file is
produced by `compute_baseline.R` (or the `compute-baseline` workflow) and is
stamped with a `method_version`.

> **Re-run `compute_baseline.R` after any change to the scoring math.** μ and σ
> define the entire Deception+ scale, so standardizing new ratios against an old
> file shifts every published score. `run_daily.R` checks the stamp and warns
> loudly rather than failing silently, but it cannot fix the scale for you.

### Baseline Selection

Three baseline options available:

- **Marginal**: Simple pitch frequencies (fastest, good for small samples)
- **Conditional**: Frequencies by count/situation (more accurate, requires more data)  
- **Hybrid**: Uses conditional when possible, falls back to marginal (recommended default)

### Model Validation

Higher Deception+ correlates with lower xFIP and higher swinging strike rate for starters.
- **Effect size is sensible**: Unpredictability matters but isn't everything
- **Direction is correct**: More unpredictable = better performance and more whiffs
- **Role-specific**: Effect differs between starters and relievers (as expected)

In addition, at the low-end, the model produces the results you'd expect to see. Position players like Enrique Hernandez and Eric Yang and knuckleballers like Matt Waldron are among the most predictable pitchers.

## Limitations

- **Sample size**: Requires substantial pitch data (50+ pitches recommended minimum)
- **Context effects**: Doesn't yet account for catcher influence or hitter-specific adjustments
- **Linear model**: Uses logistic regression; may miss non-linear patterns
- **Visualization**: R-generated visualizations still a work in progress; recommend using the output

## Future Directions

- **Catcher game-calling**: Extend to pitcher-catcher dyad analysis
- **Leverage weighting**: Weight unpredictability by situation importance
- **Sequential patterns**: Capture multi-pitch sequences beyond just previous pitch
- **Outcome validation**: Correlate with swing-and-miss rates, called strikes, wOBA
- **Platoon effects**: Analyze unpredictability separately vs. same/opposite-handed batters

## Contributing

Contributions welcome! Areas of particular interest:
- Alternative baseline models
- Visualization improvements  
- Validation against additional performance metrics
- Extensions to catcher analysis

## Citation

If you use Deception+ in your research or analysis, please cite (APA):

```
McGovern, C. (2025). Deception+: Measuring MLB and AAA pitcher unpredictability in R through machine learning analysis of pitch selection (Version 1.0.0)
[Computer software]. https://doi.org/10.5281/zenodo.17553074
```

## License

### For Researchers, Journalists, and Hobbyists
GNU GPL-3 (free) - see LICENSE

Non-commercial use is encouraged! This means:
- Academic research and publications
- Journalism and media coverage
- Personal projects and blogs
- Fantasy sports and hobbyist analysis

### For Professional Organizations
Including MLB/MiLB teams, international professional leagues, sports betting companies, and commercial scouting services who want to modify the source code without making it open source as required under GPL-3.
Contact [Conor McGovern](mailto:comcgovern@gmail.com) for commercial licensing.

Commercial use requires licensing. This means:
- Professional team scouting departments
- Player evaluation for contracts/trades
- Commercial gambling/betting operations
- Paid consulting services
- Integration into commercial products

## Acknowledgments

- Baseball Savant for Statcast data
- sabRmetrics package for data access infrastructure
- The baseball analytics community for inspiration, particularly the developers of the various Stuff and Pitching models
- The Dynasty Dugout discord and Chris Clegg for putting together a tremendous community
- Anthropic's Claude AI for checking and correcting my code and extremely helpful commenting and organizing of the code, which I am far too lazy to do well on my own

---

**Questions?** Open an issue or reach out on [Bluesky](https://bsky.app/profile/conormcgovern.bsky.social).
