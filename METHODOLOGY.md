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
- `ahead_in_count`: Binary indicator for hitter-favorable counts
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
  data = training_data,
  trace = FALSE,
  maxit = 500
)
```

- `pitch_class`: Response variable (pitch type factor)
- Formula: All features included
- `maxit = 500`: Usually converges in 50-100 iterations
- No regularization: Want to fully fit training patterns

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
# Get probability matrix: rows = pitches, columns = pitch types
P_model <- predict(model, newdata = test_data, type = "probs")

# For each pitch, extract probability of the actual pitch thrown
classes <- levels(training_data$pitch_class)
idx_actual <- match(test_data$pitch_class, classes)
p_actual_model <- P_model[cbind(1:nrow(test_data), idx_actual)]

# Calculate surprise
surprise_model <- -log(pmax(p_actual_model, 1e-12))
```

The `pmax(..., 1e-12)` ensures we never take log(0).

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
