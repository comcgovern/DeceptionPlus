# ⚾ Deception+

If you’re anything like me – which, if you’re looking at a GitHub repository for a new sabermetric model, I have to assume you are – then you have a particular tic that comes out when you're watching a baseball game. It probably happens so often and so automatically that you fail to register it, thinking it’s something everyone does, but I'm told that it's not.

I’m talking, of course, about guessing the next pitch that gets thrown. Whether you’re doing it out loud (and, like me, annoying your kids, spouse, or friends) or just in your head, guessing that next pitch is as much a part of the routine as singing “Take Me Out to the Ballgame” in the middle of the seventh.

This sequence of events (and my temporary idleness in the wake of the October 2025 government shutdown that left me furloughed from my day job) led to an investigation of whether certain pitchers were more predictable in their selection in any given situation than others.

**Deception+** is an R script that quantifies how unpredictable a pitcher's pitch selection is by comparing machine learning predictions against actual pitch choices. Higher scores indicate pitchers who successfully defy pattern recognition — even when sophisticated models know their history, tendencies, and game situation.

## What It Measures

Traditional scouting tells us that unpredictability matters. Hitters and scouting departments study video, memorize tendencies, and look for patterns. But *how much* does unpredictability matter, and how do we measure it objectively?

Deception+ uses an information-theoretic approach: we train a multinomial logistic regression model on each pitcher's historical data, including game context (count, outs, runners, batter handedness, previous pitch), and then measure how "surprised" the model is by the pitcher's actual choices. We compare this surprise against a baseline model to isolate genuine unpredictability from simple pitch mix diversity.

The metric is **scaled to 100 (league average) with a standard deviation of 10**:
- **110 or higher: Highly unpredictable** — consistently defies pattern recognition
- **100: League average** — predictable in typical ways  
- **90 or lower: Highly predictable** — follows recognizable patterns

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
- **Count state**: balls, strikes, ahead/behind in count
- **Game situation**: inning, outs, runners on base, score differential
- **Batter context**: handedness, chase rate, contact tendencies
- **Sequence**: previous pitch thrown
- **Times through order**: how often batter has faced this pitcher today

### 2. Surprise Calculation

For each pitch in the test period, we calculate **surprise** = -log(predicted probability of actual pitch). This measures how unexpected each pitch choice was.

### 3. Baseline Comparison

We compare the full model's surprise against a simpler **baseline model** that uses only count and batter handedness. This isolates true unpredictability from simple pitch mix diversity.

**Unpredictability Ratio** = Model Surprise / Baseline Surprise

Ratios > 1 mean the pitcher remains unpredictable even when accounting for game context. Ratios < 1 mean situational patterns explain most pitch selection.

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
| `deception_plus` | Scaled metric (mean=100, SD=10 over the evaluated population) |

The daily output additionally carries `n_classes` (the size of that pitcher's
pitch vocabulary), `role`, and `status`. Rows with a `status` other than
`evaluated` are pitchers who appeared but could not be scored — a debut with no
history, too little history, or too few pitches on the day — and their metric
columns are intentionally blank.

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
  feature_names = c("balls", "strikes", "two_strikes", "ahead_in_count",
                    "high_leverage", "times_through_order", "outs",
                    "score_diff", "base_state", "is_risp",
                    "stand", "p_throws", "last_pitch_type",
                    "o_swing_pct", "z_contact_pct", "swing_pct"),
  baseline_keys = c("balls", "strikes", "is_risp", "stand", "p_throws")
)
```

### Scoring Controls

Three parameters govern how probabilities become surprise. The defaults are
sane; change them only if you know why.

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `decay` | `1e-4` seasonal, `0.01` per-pitcher | Weight decay (L2 penalty) on the multinomial fit. Small per-pitcher samples separate easily; without a penalty the fit emits probabilities of 0 and 1 and `-log(p)` stops measuring surprise. |
| `prob_shrinkage` | `0.02` | Prior mass mixed into predicted probabilities before `-log()`. Keeps model and baseline surprise on the same floor so their ratio stays meaningful. |
| `standardize` | `"test"` | Population whose μ/σ anchor the Deception+ scale. `"test"` makes the output actually have mean 100 / SD 10. `"train"` gives a test-period-independent anchor but is measured in-sample and therefore optimistically biased. |

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
