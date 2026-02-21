# ============================================================================
# run.R — Predict+ Training and Analysis Runner
# ----------------------------------------------------------------------------
# This script runs the Predict+ training workflow with configurable parameters.
# Train on one period, test on another (they can overlap or be identical).
# ============================================================================

# Source the main Predict+ functions
source("pitch_ppi.R")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Training period - data used to train the multinomial model
TRAIN_START <- "2025-03-01"
TRAIN_END   <- "2025-09-30"

# Test period - data used to evaluate the model
# Can be:
#   - Same as training (e.g., to see how predictable pitchers were in-sample)
#   - Overlapping with training
#   - Completely separate (e.g., train on regular season, test on playoffs)
TEST_START <- "2025-03-01"
TEST_END   <- "2025-09-30"

# Examples of different configurations:
# 
# 1. SAME PERIOD (in-sample):
#    TRAIN_START <- "2024-04-01"
#    TRAIN_END   <- "2024-09-30"
#    TEST_START  <- "2024-04-01"
#    TEST_END    <- "2024-09-30"
#
# 2. OVERLAPPING (test includes part of training):
#    TRAIN_START <- "2024-04-01"
#    TRAIN_END   <- "2024-08-31"
#    TEST_START  <- "2024-08-01"
#    TEST_END    <- "2024-09-30"
#
# 3. SEPARATE (no overlap):
#    TRAIN_START <- "2024-04-01"
#    TRAIN_END   <- "2024-07-31"
#    TEST_START  <- "2024-08-01"
#    TEST_END    <- "2024-09-30"
#
# 4. PLAYOFFS (train on regular season, test on playoffs):
#    TRAIN_START     <- "2024-03-20"
#    TRAIN_END       <- "2024-09-28"
#    TEST_START      <- "2024-10-01"
#    TEST_END        <- "2024-10-31"
#    TEST_GAME_TYPE  <- "P"  # or "W" for World Series

# Analysis parameters
MIN_TEST_PITCHES  <- 100    # Minimum pitches in test period to be included
MIN_TOTAL_PITCHES <- 250    # Minimum pitches overall to be included
TRAIN_GAME_TYPE   <- "R"   # "R" = Regular season, "P" = Playoffs, "S" = Spring
TEST_GAME_TYPE    <- "R"   # "R" = Regular season, "P" = Playoffs, "W" = World Series

# Level selection (MLB or AAA)
TRAIN_LEVEL <- "MLB"  # "MLB" = Major League Baseball, "AAA" = Triple-A
TEST_LEVEL  <- "MLB"  # "MLB" = Major League Baseball, "AAA" = Triple-A

# Split method selection
# Options:
#   "temporal" - Use specific date ranges for train and test (default)
#                Train on one period, test on another (can overlap or be separate)
#   "random"   - Randomly assign 50% of each pitcher's pitches to train, 50% to test
#                Ignores TEST_START/TEST_END; uses TRAIN dates for entire period
#                Useful for understanding model fit without temporal confounds
SPLIT_METHOD <- "temporal"

# Random seed for reproducibility (only used when SPLIT_METHOD = "random")
# Set to NULL for different results each run, or a number for reproducibility
RANDOM_SEED <- NULL

# Baseline model selection
# Options:
#   "marginal"    - Simple overall pitch type distribution (weakest baseline)
#   "conditional" - Conditional on game state features (standard baseline)
#   "hybrid"      - Conditional when sufficient data, marginal fallback (recommended)
BASELINE_TYPE <- "conditional"

# Features to use in the multinomial model
FEATURE_NAMES <- c(
  "balls", "strikes", "two_strikes", "ahead_in_count",
  "is_top", "outs", "score_diff", 
  "base_state", "is_risp",
  "high_leverage",        # Late inning + close game
  "times_through_order",  # How many times pitcher has faced batter this game
  "stand", "p_throws", "last_pitch_type",
  "o_swing_pct", "z_contact_pct", "swing_pct", "chase_contact_pct"
)

# Features to use for conditional baseline
BASELINE_KEYS <- c(
  "balls", "strikes",
  "stand", "p_throws", "two_strikes"
)

# Output paths (will be created if they don't exist)
OUT_MODEL <- "models/ppi_model_mlb_2025.rds"
OUT_CSV   <- "output/pitcher_ppi_mlb_2025.csv"

# ============================================================================
# EXECUTION
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  Predict+ Training Pipeline\n")
cat("============================================================\n")
cat("Split Method:    ", SPLIT_METHOD, "\n")
if (SPLIT_METHOD == "temporal") {
  cat("Training Period: ", TRAIN_START, "to", TRAIN_END, "(", TRAIN_GAME_TYPE, ")\n")
  cat("Test Period:     ", TEST_START, "to", TEST_END, "(", TEST_GAME_TYPE, ")\n")
} else {
  cat("Data Period:     ", TRAIN_START, "to", TRAIN_END, "(", TRAIN_GAME_TYPE, ")\n")
  cat("Test Period:      Random 50/50 split per pitcher\n")
  if (!is.null(RANDOM_SEED)) cat("Random Seed:     ", RANDOM_SEED, "\n")
}
cat("Min Test Pitches:", MIN_TEST_PITCHES, "\n")
cat("Min Total:       ", MIN_TOTAL_PITCHES, "\n")
cat("Baseline Type:   ", BASELINE_TYPE, "\n")
cat("Output Model:    ", OUT_MODEL, "\n")
cat("Output CSV:      ", OUT_CSV, "\n")
cat("============================================================\n\n")

# Run training
res <- train_and_save(
  train_start       = TRAIN_START,
  train_end         = TRAIN_END,
  test_start        = if (SPLIT_METHOD == "random") NULL else TEST_START,
  test_end          = if (SPLIT_METHOD == "random") NULL else TEST_END,
  min_test_pitches  = MIN_TEST_PITCHES,
  min_total_pitches = MIN_TOTAL_PITCHES,
  feature_names     = FEATURE_NAMES,
  baseline_keys     = BASELINE_KEYS,
  baseline_type     = BASELINE_TYPE,
  train_game_type   = TRAIN_GAME_TYPE,
  test_game_type    = TEST_GAME_TYPE,
  train_level       = TRAIN_LEVEL,
  test_level        = TEST_LEVEL,
  split_method      = SPLIT_METHOD,
  random_seed       = RANDOM_SEED,
  out_model         = OUT_MODEL,
  out_ppi           = OUT_CSV,
  verbose           = TRUE
)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  Training Complete - Summary Statistics\n")
cat("============================================================\n")
cat("Total pitchers:  ", nrow(res$pitcher_ppi), "\n")
cat("Split method:    ", res$split_method, "\n")
cat("Training period: ", res$train_period, "\n")
cat("Test period:     ", res$test_period, "\n")
cat("Training pitches:", nrow(res$train), "\n")
cat("Test pitches:    ", nrow(res$test), "\n")
cat("Features used:   ", length(res$features_used), "(", paste(res$features_used, collapse = ", "), ")\n")
cat("Baseline keys:   ", length(res$baseline_keys), "(", paste(res$baseline_keys, collapse = ", "), ")\n")
cat("Baseline type:   ", res$baseline_type, "\n")
cat("============================================================\n\n")

# ============================================================================
# TOP/BOTTOM PERFORMERS
# ============================================================================

cat("============================================================\n")
cat("  Top 10 Most Predictable Pitchers (Lowest Predict+)\n")
cat("============================================================\n")
top_pred <- res$pitcher_ppi %>%
  arrange(predict_plus) %>%
  head(10) %>%
  mutate(
    rank = row_number(),
    pitcher_name = str_trunc(pitcher_name, 25)
  ) %>%
  select(rank, pitcher_name, n_pitches_test, predict_plus, ppi)
print(top_pred, n = 10)

cat("\n")
cat("============================================================\n")
cat("  Top 10 Least Predictable Pitchers (Highest Predict+)\n")
cat("============================================================\n")
top_unpred <- res$pitcher_ppi %>%
  arrange(desc(predict_plus)) %>%
  head(10) %>%
  mutate(
    rank = row_number(),
    pitcher_name = str_trunc(pitcher_name, 25)
  ) %>%
  select(rank, pitcher_name, n_pitches_test, predict_plus, ppi)
print(top_unpred, n = 10)

# ============================================================================
# VISUALIZATIONS
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  Generating Visualizations\n")
cat("============================================================\n")

create_visualizations(res, output_dir = "output/visualizations")

# ============================================================================
# COMPLETION
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  Pipeline Complete!\n")
cat("============================================================\n")
cat("✅ Model saved to:         ", OUT_MODEL, "\n")
cat("✅ Results CSV saved to:   ", OUT_CSV, "\n")
cat("✅ Visualizations saved to: output/visualizations/\n")
cat("✅ Cached data in:         cache/\n")
cat("\nTo view results:\n")
cat("  - Read CSV: read.csv('", OUT_CSV, "')\n", sep = "")
cat("  - Load model: readRDS('", OUT_MODEL, "')\n", sep = "")
cat("  - View visualizations: output/visualizations/*.png\n")
cat("============================================================\n\n")

# Return the results object for interactive use
res
