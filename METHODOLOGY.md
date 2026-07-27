# Methodology: How Deception+ Works

This document explains the technical approach behind Deception+ for those interested in the statistical and information-theoretic foundations.

## Overview

Deception+ measures pitcher unpredictability by quantifying how much a pitcher's actual pitch choices "surprise" a machine learning model that has learned their patterns. If a sophisticated model that knows everything about game context, pitcher history, and sequencing patterns still can't predict what you'll throw, you're genuinely unpredictable.

## The Predictability Challenge

### What Makes a Pitcher Unpredictable?

Consider two pitchers with identical 60/40 fastball/slider arsenals:

**Pitcher A** (Predictable):
- Always throws fastball on 0-0
- Always throws slider with 2 strikes
- Fastball when behind, slider when ahead
- Clear patterns by count and situation

**Pitcher B** (Unpredictable):
- 60/40 fastball/slider mix overall
- Similar usage in most counts
- No consistent pattern by situation
- Situationally independent

Both have the same pitch mix, but Pitcher A follows predictable rules while Pitcher B doesn't. Traditional pitch mix metrics can't distinguish them. **Deception+ can.**

## Information-Theoretic Foundation

### Surprise (Negative Log-Likelihood)

For a predicted probability distribution P over pitch types and an actual pitch y, **surprise** is defined as:

```
S(y | P) = -log(P(y))
```

Properties of surprise:
- **Inversely related to probability**: Rare events are more surprising
- **Proper scoring rule**: Encourages well-calibrated predictions
- **Additive**: Surprise over multiple pitches sums naturally
- **Information-theoretic**: Measured in bits (log base 2) or nats (natural log)

We use natural log (nats) for numerical stability.

### Expected Surprise = Entropy

The *expected* surprise over a distribution is entropy:

```
H(Y) = -∑ P(y) log P(y)
```

This connects our pitch-level surprise metric to classical information theory.

### Why Not Just Entropy?

We could calculate Shannon entropy on pitch type frequencies:

```r
freqs <- c(0.6, 0.4)  # 60% fastball, 40% slider
entropy <- -sum(freqs * log(freqs))
```

**Problems:**
1. **Ignores context**: Same entropy whether you always throw fastball on 0-0 or randomize
2. **Marginal only**: Doesn't capture conditional patterns
3. **No validation**: Can't test predictions against reality

Deception+ uses *conditional* surprise from an actual predictive model, then compares against a baseline.

## The Two-Model Approach

### Full Model (Complex)

A multinomial logistic regression predicting pitch type from:

#### Game Context
- `balls`, `strikes`: Current count
- `two_strikes`: Binary indicator for 2-strike counts
- `ahead_in_count`: Binary indicator for **pitcher**-favorable counts (`strikes > balls`)
- `outs`: Current outs (0, 1, 2)
- `inning`: Which inning (treated as categorical, not continuous)
- `score_diff`: Home score - away score
- `high_leverage`: Late inning + close game indicator
- `n_thruorder_pitcher`: Times through the order (from Statcast)

#### Base-Out State
- `base_state`: 8 possible configurations (empty, runner on 1st, 2nd, 3rd, 1st+2nd, 1st+3rd, 2nd+3rd, loaded)
- `is_risp`: Binary indicator for runner in scoring position

#### Batter Information
- `stand`: Batter handedness (L/R)
- `p_throws`: Pitcher handedness (L/R)
- `o_swing_pct`: Batter's chase rate (swing at pitches outside zone)
- `z_contact_pct`: Batter's in-zone contact rate
- `swing_pct`: Overall swing rate
- `chase_contact_pct`: Contact rate on chases

#### Sequence
- `last_pitch_type`: Previous pitch thrown in this at-bat

**Why Multinomial Logistic Regression?**
- Handles multiple pitch types naturally
- Interpretable coefficients
- Well-behaved probability predictions
- Fast to train even with 100k+ pitches
- Baseline established in prediction literature

Could we use fancier models (random forests, neural networks)?
1. MLR is our *minimum viable unpredictability* test — if you're unpredictable to MLR, you're unpredictable
2. Interpretable coefficients help verify model sanity
3. Faster training enables flexible period analysis

### Baseline Model (Simple)

The baseline model uses only:
- `balls`, `strikes`: Count state
- `is_risp`: Runner in scoring position
- `stand`, `p_throws`: Handedness matchup
- `two_strikes`: Two-strike indicator

This captures basic situational tendencies without deep context or sequencing.

**Three Baseline Options:**

1. **Marginal**: Simple pitch frequencies (ignores all context)
   - Fastest, works with small samples
   - Baseline for measuring pure contextual predictability

2. **Conditional**: Frequencies within count/situation cells
   - More accurate baseline
   - Requires adequate sample per cell
   - Default approach

3. **Hybrid**: Conditional when possible, marginal fallback
   - Robust to sparse data
   - Recommended for most uses

**Smoothing.** Conditional cells are smoothed by backing off toward the pitcher's
marginal mix rather than toward a uniform distribution:

```
P(class | cell) = (n_cell,class + α · P_marginal(class)) / (n_cell + α)
```

The distinction is not cosmetic. Plain add-one smoothing puts a pseudo-count on
*every* class, so a cell holding four real pitches from an eight-pitch vocabulary
is two-thirds prior — and that prior asserts the pitcher is equally likely to
throw all eight, which is never true. It inflates baseline surprise in proportion
to arsenal size, and since Deception+ divides by baseline surprise, it quietly
rewarded deep arsenals. Backing off to the marginal mix removes that dependence
and converges on the raw cell frequencies as cells fill.

Cells absent from the training window fall back to the marginal mix as well.
(A uniform `1/K` fallback charges ~log K nats of baseline surprise for a context
the baseline simply never encountered — arbitrary, and it deflates the ratio.)

### The Comparison

For each pitch in the test period:

```
S_model = -log(P_model(actual pitch))
S_baseline = -log(P_baseline(actual pitch))
```

Aggregate to pitcher level:

```
Mean_S_model = mean(S_model across test pitches)
Mean_S_baseline = mean(S_baseline across test pitches)

Unpredictability_Ratio = Mean_S_model / Mean_S_baseline
```

**Interpretation:**

- **Ratio > 1**: Complex model is *more* surprised than baseline
  - Pitcher doesn't follow predictable patterns
  - Context/sequencing doesn't help prediction
  - **High unpredictability**

- **Ratio ≈ 1**: Both models equally surprised
  - Simple count-based rules explain pitch selection
  - **Average unpredictability**

- **Ratio < 1**: Complex model is *less* surprised than baseline
  - Pitcher follows complex but predictable patterns
  - Context/sequencing *does* help prediction
  - **Low unpredictability** (high predictability)

### Why This Works

The ratio isolates genuine unpredictability from:
- **Arsenal diversity**: Controlled by baseline model seeing pitch frequencies
- **Count effects**: Both models include count
- **Sample size**: Ratio is scale-invariant (both models see same pitches)

What remains is **situational independence** — pitchers who don't follow learnable patterns even when we account for context.

## Standardization: Deception+

Raw ratios are hard to interpret. We standardize to a scaled metric:

```
μ = mean(Unpredictability_Ratio across all pitchers)
σ = std(Unpredictability_Ratio across all pitchers)

Deception+ = 100 + 10 × ((Unpredictability_Ratio - μ) / σ)
```

This gives us:
- **Mean = 100** (league average)
- **SD = 10** (one standard deviation = 10 points)
- **Intuitive scale**: Similar to ERA+, wRC+, etc.

### Which Population Sets μ and σ?

`train_ppi(standardize = ...)` chooses the reference population:

- **`"test"`** (default) — μ and σ from the pitchers actually evaluated. The
  output then genuinely has mean 100 and SD 10, as described above.
- **`"train"`** — μ and σ from the training period, giving an anchor that does not
  move when you change the test window. Useful for comparing several test windows
  fit from a single training window, but note it is measured **in sample**: model
  surprise on training data is optimistically low, so this μ sits below the
  out-of-sample μ and every score shifts upward. It is a stable anchor, not an
  unbiased one.

The daily pipeline is different again: it standardizes against fixed μ/σ stored in
`baseline_params.rds` by `compute_baseline.R`, so that scores are comparable from
day to day. Those parameters are estimated from repeated random splits with the
top and bottom 1% of ratios trimmed — μ and σ define the whole scale, so a handful
of extreme values would otherwise drag it for everyone.

**`baseline_params.rds` is versioned.** Any change to the scoring math invalidates
previously saved μ/σ, and `run_daily.R` warns loudly rather than silently
publishing scores on a stale scale. Re-run `compute_baseline.R` after such a
change.

### A Caveat on the Ratio

Dividing two mean surprises is only well-conditioned when the denominator is
comfortably above zero. For a pitcher who throws one pitch ~99% of the time, both
the model and the baseline predict nearly perfectly, both mean surprises are close
to zero, and their ratio is dominated by the rounding-level gap between them — such
a pitcher can post a higher ratio than a genuinely coin-flip pitcher while the
actual information gap is ~0.01 nats.

The `surp_excess` column (model surprise **minus** baseline surprise, in nats)
measures the same thing on a difference scale and does not have this failure mode.
It is reported alongside the ratio, and is the sounder basis for any future
respecification of Deception+.

## Training and Testing Periods

### Why Separate Train/Test?

We train on one period and evaluate on another to:
1. **Avoid overfitting**: Model can't memorize test period
2. **Measure stability**: Does unpredictability persist over time?
3. **Enable temporal analysis**: Train on regular season, test on playoffs
4. **Validate predictions**: True test of predictive power

### Flexible Period Design

The system supports three modes:

1. **Same period** (`test_days` parameter):
   ```r
   start_date = "2025-03-01"
   end_date = "2025-09-30"
   test_days = 30
   # Train: Mar 1 - Aug 31
   # Test: Sep 1 - Sep 30
   ```

2. **Explicit periods** (can overlap):
   ```r
   train_start_date = "2025-03-01"
   train_end_date = "2025-09-30"
   test_start_date = "2025-08-01"  # Overlaps!
   test_end_date = "2025-09-30"
   ```

3. **Separate periods** (most common):
   ```r
   train_start_date = "2025-03-01"
   train_end_date = "2025-09-30"
   test_start_date = "2025-10-01"  # Playoffs
   test_end_date = "2025-11-05"
   ```

### Handling Test Period Pitchers Not in Training

If a pitcher appears in test but not training:
- **Excluded from results** (can't measure unpredictability without learning patterns)
- Not an error — common for rookies or call-ups
- Requires sufficient training data (50+ pitches) to learn patterns

## Feature Engineering Details

### Categorical Variables

Several features are categorical despite numeric appearance:

- `last_pitch_type`: Factor with levels = all pitch types seen in training
  - Includes "NONE" for first pitch of at-bat
  - Prevents continuous treatment of "FF" = 1, "SL" = 2, etc.

- `base_state`: Factor for 8 possible configurations
  - Runner on 1st ≠ 2 × runner on 2nd
  - Non-linear importance by configuration

### Continuous Variables

Some features remain continuous:

- `balls`, `strikes`, `outs`: Natural ordinal scale
- `score_diff`: Linear relationship (ahead by 5 ≈ 2.5 × ahead by 2)
- Batter metrics (`o_swing_pct`, etc.): Continuous percentages (WIP)

### Times Through Order

Critical variable: `n_thruorder_pitcher` from Statcast measures how many times the pitcher has faced this batter in *this game*:

- 1st time: Fresh look
- 2nd time: Batter saw pitcher once
- 3rd time: Batter has adjusted twice

### Constant Feature Dropping

Before training, we remove features with:
- Only 1 level (categorical)
- Only 1 unique value (numeric)

This prevents model fitting errors on constant predictors.

## Model Training Details

### Multinomial Setup

```r
model <- nnet::multinom(
  pitch_class ~ balls + strikes + two_strikes + ... ,
  data  = training_data,
  trace = FALSE,
  maxit = 500,     # 200 for per-pitcher models
  decay = 1e-4     # 0.01 for per-pitcher models
)
```

- `pitch_class`: Response variable (pitch type factor)
- Formula: All features included
- `maxit = 500`: Usually converges in 50-100 iterations
- `decay`: light L2 penalty. The seasonal model is fit on hundreds of thousands
  of pitches and barely notices it; the per-pitcher models are fit on a few
  hundred pitches with a wide `last_pitch_type` factor, where separation is the
  norm rather than the exception, and need the stronger setting.

### Response Vocabulary Is Per-Pitcher

Each per-pitcher model ranges over **that pitcher's own** pitch types — the classes
in their history, plus any class they actually threw in the test window — not the
league-wide vocabulary. Carrying league-wide factor levels into an individual
model means most outcome classes have zero observations, and `nnet::multinom`
responds by dropping them, leaving the fitted model narrower than the class set the
caller believes it is working with.

Test-window classes are deliberately kept in the support even when they are absent
from the pitcher's history. A pitcher unveiling a new pitch *is* being
unpredictable; that is the signal, not an inconvenience. Those pitches are scored
against the smoothed prior, earning high but finite surprise.

### Convergence

Model typically converges with:
- **1,000+ pitches**: Very reliable
- **500-1,000 pitches**: Usually fine
- **< 500 pitches**: May struggle with complex features

If model fails to converge:
1. Try reducing features
2. Use marginal baseline instead of conditional
3. Expand training period
4. Accept that pitcher may not have learnable patterns (high unpredictability!)

## Calculating Surprise

### From Model Predictions

```r
# Get probability matrix: rows = pitches, columns = pitch types.
# safe_predict_probs() aligns the model's output to `classes` BY NAME.
# This matters: nnet::multinom silently drops response levels it never saw, and
# with exactly two levels predict() returns a bare vector instead of a matrix,
# so the number and order of returned columns cannot be assumed.
classes <- levels(training_data$pitch_class)
P_model <- safe_predict_probs(model, test_data, classes)

# Mix in a little of the training pitch mix before taking logs (see below)
P_model <- shrink_to_prior(P_model, class_prior(training_data$pitch_class, classes))

# For each pitch, extract probability of the actual pitch thrown
idx_actual <- match(as.character(test_data$pitch_class), classes)
p_actual_model <- P_model[cbind(1:nrow(test_data), idx_actual)]

# Calculate surprise
surprise_model <- -log(pmax(p_actual_model, 1e-9))
```

### Bounding the Surprise

`-log(p)` is unbounded as `p → 0`, and an unpenalized multinomial fit on a few
hundred pitches reaches complete separation routinely — it will happily report
`p = 1e-15`. Clamping at some tiny epsilon does not fix this; it just converts an
arbitrary probability into an arbitrary constant (a clamp at `1e-9` yields 20.7
nats, roughly fifteen times a typical pitch's surprise).

That matters more than it looks, because Deception+ is a *ratio*. The baseline is
a smoothed frequency table and so has a natural probability floor, while the model
did not. An unbounded numerator over a bounded denominator manufactures extreme
scores out of numerical noise.

Two mitigations, applied together:

1. **Weight decay** on the multinomial fit (`decay`), which keeps coefficients —
   and therefore fitted probabilities — away from the 0/1 boundary.
2. **Shrinkage toward a prior** (`prob_shrinkage`): each probability row is mixed
   with the training pitch mix, `p' = (1-λ)p + λ·prior`, before `-log()`. Applied
   identically to model and baseline, so both sit on the same floor. Mixing with a
   fixed distribution preserves the proper-scoring-rule property.

### From Baseline Model

Similar process with simpler model or frequency table:

```r
# For conditional baseline
P_baseline <- conditional_freq_table[count_situation_cells, pitch_types]
p_actual_baseline <- P_baseline[cbind(1:nrow(test_data), idx_actual)]
surprise_baseline <- -log(pmax(p_actual_baseline, 1e-12))
```

### Aggregation to Pitcher Level

```r
pitcher_stats <- test_data %>%
  mutate(
    surp_model = surprise_model,
    surp_baseline = surprise_baseline
  ) %>%
  group_by(pitcher_id) %>%
  summarise(
    n_pitches = n(),
    mean_surp_model = mean(surp_model),
    mean_surp_baseline = mean(surp_baseline)
  ) %>%
  mutate(
    unpredictability_ratio = mean_surp_model / mean_surp_baseline
  )
```

## Validation and Interpretation

### Correlations with Performance

Negative correlation with xFIP means: higher Deception+ → lower xFIP → better performance.
Positive correlation with SwStr% means: higher Deception+ → more swinging strikes.

### Two-Pitch Pitcher Case Study

Trevor Megill (2-pitch reliever): **Very high Deception+ score in 2025**

**Interpretation:**
- Only throws fastball and slider
- But doesn't follow predictable count-based patterns
- Simple arsenal, but situationally independent usage
- High unpredictability from *independence*, not diversity

**Key insight**: Unpredictability ≠ large arsenal. It's about breaking patterns.

### What Makes Scores Extreme?

**High Deception+ (115+):**
- Situational independence (no count/runner patterns)
- Balanced usage in traditionally "obvious" situations
- Sequence unpredictability (previous pitch doesn't matter)
- Two-pitch pitchers who don't follow rules

**Low Deception+ (85-):**
- Strong count-based patterns
- Clear sequencing rules
- Runner-dependent strategies
- "Textbook" pitch selection
- Position player or very limited arsenal

## Limitations and Future Work

### Current Limitations

1. **No catcher effects**: Doesn't account for game-calling differences
2. **Linear model**: May miss non-linear patterns
3. **Sample size**: Needs 50+ test pitches for stable estimates
4. **Temporal stability**: Assumes patterns learned in training apply to test

### Ongoing Development

**Catcher Integration**
Extend to pitcher-catcher dyads:
```r
deception_plus_with_catcher_A - deception_plus_with_other_catchers
→ Catcher A's game-calling effect
```

**Leverage Weighting**
Weight surprise by situation importance:
```r
leveraged_surprise = surprise × leverage_index
```

**Sequential Patterns**
Capture multi-pitch patterns:
```r
# Instead of just last_pitch_type
pitch_sequence = "FF-SL-FF" → predict next pitch
```

**Non-Linear Models**
Test whether random forests, XGBoost improve predictions:
- May capture interaction effects
- Risk: harder to interpret, may overfit

**Platoon-Specific**
Separate scores vs. LHH and RHH:
```r
deception_plus_vs_LHH
deception_plus_vs_RHH
```

---

**Questions about methodology?** Open an issue on GitHub or reach out on Twitter/Bluesky.
