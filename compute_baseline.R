# ============================================================================
# compute_baseline.R — Compute Fixed Baseline Parameters for Deception+
# ----------------------------------------------------------------------------
# Samples N game-days from the historical window (2023-2025) and scores each one
# exactly the way run_daily.R does — per-pitcher models trained on that pitcher's
# preceding N_HISTORY_PITCHES — to establish stable standardization parameters
# (μ and σ).
#
# Mirroring the production setup is the point. The unpredictability ratio depends
# on how much data each model is fit on and on how many pitches it is scored over,
# so calibrating under a different regime puts the published scale off-centre.
#
# This creates a fixed reference point for Deception+ that:
#   - Defines what "100" means (average MLB pitcher unpredictability)
#   - Is stable across all future runs
#   - Should be recomputed periodically (e.g., annually)
#
# Output: baseline_params.rds containing:
#   - mu: mean unpredictability_ratio across all pitchers and runs
#   - sd: standard deviation of unpredictability_ratio
#   - n_runs: number of game-days sampled
#   - n_pitchers: total unique pitchers across all runs
#   - date_computed: when baseline was computed
#   - data_period: date range used for computation
# ============================================================================

source("pitch_ppi.R")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Date range for baseline computation
BASELINE_START <- "2023-03-30"  # 2023 Opening Day
BASELINE_END   <- "2025-09-28"  # End of 2025 regular season (adjust as needed)

# Number of sampled game-days to average over
N_RUNS <- 30

# These MUST mirror run_daily.R. The unpredictability ratio is not invariant to
# how much data the per-pitcher model is fit on: a multinomial with more
# parameters than it can support pays an out-of-sample overfitting penalty that
# lands in the numerator, so the ratio falls as training data grows. Measured on
# synthetic pitchers with identical true unpredictability:
#
#     training pitches   150     300     500    1000    3000
#     ratio (2 pitches) 1.066   1.014   1.001   0.998   0.997
#     ratio (4 pitches) 1.138   1.037   1.012   0.999   0.999
#
# The old calibration used a random 50/50 split of each pitcher's FULL 2023-2025
# history — often thousands of training pitches, and a test window of similar
# size — while production trains on at most 500 and scores a single day. mu was
# therefore measured at the bottom of that curve and production sat above it,
# inflating every score, most of all for short-history pitchers with deep
# arsenals. Sampling actual game-days reproduces the production setup exactly,
# including the test-window size that sets sigma.
N_HISTORY_PITCHES   <- 500
MIN_HISTORY_PITCHES <- 100
MIN_TEST_PITCHES    <- 5

# Output file
OUTPUT_FILE <- "baseline_params.rds"

# Features for per-pitcher models
FEATURE_NAMES <- c(
 "count", "outs", "is_risp", "stand", "last_pitch_type"
)

# Baseline keys for simple baseline comparison
BASELINE_KEYS <- c("count", "stand")

# ============================================================================
# PER-PITCHER MODEL FUNCTION
# ============================================================================

# Uses evaluate_per_pitcher() from pitch_ppi.R (shared implementation).
# Thin wrapper to match the interface expected by the baseline computation loop.
compute_per_pitcher_unpredictability <- function(df_train,
                                                   df_test,
                                                   min_train_pitches = 50,
                                                   min_test_pitches = 10,
                                                   feature_names = FEATURE_NAMES,
                                                   baseline_keys = BASELINE_KEYS,
                                                   verbose = TRUE) {

  result <- evaluate_per_pitcher(
    df_history = df_train,
    df_test = df_test,
    min_train_pitches = min_train_pitches,
    min_test_pitches = min_test_pitches,
    feature_names = feature_names,
    baseline_keys = baseline_keys,
    verbose = verbose
  )

  # Rename n_history -> n_train to match the baseline computation expectation
  if (nrow(result$results) > 0) {
    result$results %>%
      dplyr::rename(n_train = n_history)
  } else {
    tibble()
  }
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  Deception+ Baseline Computation\n")
cat("============================================================\n")
cat("Data Period:     ", BASELINE_START, "to", BASELINE_END, "\n")
cat("Sampled Days:    ", N_RUNS, "\n")
cat("History Cap:     ", N_HISTORY_PITCHES, " pitches\n")
cat("Min History:     ", MIN_HISTORY_PITCHES, "\n")
cat("Min Test:        ", MIN_TEST_PITCHES, "\n")
cat("Output File:     ", OUTPUT_FILE, "\n")
cat("============================================================\n\n")

# Ensure directories exist
ensure_directories()

# Load data for the full baseline period
message("Loading baseline data...")

TRAIN_LEVEL <- "MLB"
cachefile <- sprintf("cache/savant_raw_%s_%s_R_MLB.Rds", BASELINE_START, BASELINE_END)

if (file.exists(cachefile)) {
 message("Using cached data: ", cachefile)
 raw_data <- readRDS(cachefile)
} else {
 message("Downloading data (this may take a while)...")
 raw_data <- load_statcast_range(BASELINE_START, BASELINE_END,
                                  game_type = "R", level = "MLB", verbose = TRUE)
 if (nrow(raw_data) > 0) {
   saveRDS(raw_data, cachefile)
   message("Cached to: ", cachefile)
 } else {
   stop("No data found for baseline period")
 }
}

message("Engineering features...")
batter_metric_cols <- c("o_swing_pct", "z_contact_pct", "swing_pct", "chase_contact_pct")
df_all <- engineer_features(raw_data, include_batter_metrics = any(batter_metric_cols %in% FEATURE_NAMES))
df_all <- df_all %>% filter(!is.na(pitcher_id))

message("Total: ", nrow(df_all), " pitches from ",
       length(unique(df_all$pitcher_id)), " pitchers\n")

# Sample N game-days and score each one exactly the way run_daily.R would:
# history = that pitcher's last N_HISTORY_PITCHES before the date, test = the date.
df_all$game_day <- as.Date(df_all$game_date)
all_days <- sort(unique(df_all$game_day))

# Skip the opening stretch of the window — pitchers have no history to train on yet.
eligible_days <- all_days[all_days >= (min(all_days) + 45)]
if (length(eligible_days) < N_RUNS) {
  stop("Only ", length(eligible_days), " eligible game-days for ", N_RUNS, " runs")
}

set.seed(20260101)
sampled_days <- sort(sample(eligible_days, N_RUNS))

message("Scoring ", N_RUNS, " sampled game-days with per-pitcher models...")
message("  (history capped at ", N_HISTORY_PITCHES,
        " pitches, matching run_daily.R)\n")

all_ratios <- vector("list", N_RUNS)

for (run in seq_len(N_RUNS)) {
 target_day <- sampled_days[run]
 if (run %% 5 == 1 || run == N_RUNS) {
   message(sprintf("Run %d/%d (%s)...", run, N_RUNS, target_day))
 }

 df_test <- df_all %>% filter(game_day == target_day)
 if (nrow(df_test) == 0) next

 day_pitchers <- unique(df_test$pitcher_id)

 # Strictly before the target day — no leakage from the day being scored.
 df_history <- get_pitcher_history(
   df_all %>% filter(game_day < target_day),
   day_pitchers,
   n_pitches = N_HISTORY_PITCHES
 )
 if (nrow(df_history) == 0) next

 run_results <- compute_per_pitcher_unpredictability(
   df_history, df_test,
   min_train_pitches = MIN_HISTORY_PITCHES,
   min_test_pitches  = MIN_TEST_PITCHES,
   feature_names = FEATURE_NAMES,
   verbose = FALSE
 )

 if (nrow(run_results) > 0) {
   run_results$run <- run
   run_results$game_day <- target_day
   all_ratios[[run]] <- run_results
 }
}

# Combine all results
all_results <- bind_rows(all_ratios)

if (nrow(all_results) == 0) {
 stop("No valid results from any run")
}

message("\nComputing baseline parameters...")

# Compute stable μ and σ across all runs.
#
# Trim the tails first. μ and σ define the entire Deception+ scale, so a handful
# of extreme ratios drags the scale for everyone: the previous baseline_params.rds
# recorded μ = 2.466, σ = 2.355 — a coefficient of variation near 1, which is the
# signature of a heavy-tailed contaminant, not of a population of pitchers.
# (Most of that tail came from the prediction-alignment bug now fixed in
# safe_predict_probs(); trimming keeps the estimator robust regardless.)
TRIM <- 0.01  # drop the top and bottom 1% of ratios

ratios_all <- all_results$unpredictability_ratio
ratios_all <- ratios_all[is.finite(ratios_all)]
if (length(ratios_all) == 0) stop("No finite unpredictability ratios to summarise")

lim <- quantile(ratios_all, c(TRIM, 1 - TRIM), na.rm = TRUE)
ratios <- ratios_all[ratios_all >= lim[1] & ratios_all <= lim[2]]

n_trimmed <- length(ratios_all) - length(ratios)
message(sprintf("  Trimmed %d of %d observations (%.1f%%) outside [%.4f, %.4f]",
                n_trimmed, length(ratios_all),
                100 * n_trimmed / length(ratios_all), lim[1], lim[2]))

baseline_mu <- mean(ratios)
baseline_sd <- sd(ratios)

if (!is.finite(baseline_sd) || baseline_sd <= 0) {
  stop("Baseline σ is not usable (", baseline_sd, "); check the input data.")
}

# Sanity check: a healthy ratio distribution sits near 1 — the full model, which
# strictly dominates the baseline in information, should not be routinely more
# surprised than it.
if (baseline_mu > 1.5) {
  warning("Baseline μ = ", round(baseline_mu, 3), " is far above 1. The full model ",
          "is on average much more surprised than the simple baseline, which usually ",
          "means the per-pitcher models are overfitting rather than that pitchers ",
          "are unpredictable. Check `decay` and `prob_shrinkage`.")
}

# Summary stats
n_observations <- nrow(all_results)
n_unique_pitchers <- length(unique(all_results$pitcher_id))
observations_per_run <- all_results %>%
 group_by(run) %>%
 summarise(n = n(), .groups = "drop")

cat("\n")
cat("============================================================\n")
cat("  Baseline Parameters Computed\n")
cat("============================================================\n")
cat("μ (mean):              ", round(baseline_mu, 6), "\n")
cat("σ (std dev):           ", round(baseline_sd, 6), "\n")
cat("Total observations:    ", n_observations, "\n")
cat("Unique pitchers:       ", n_unique_pitchers, "\n")
cat("Avg pitchers per run:  ", round(mean(observations_per_run$n), 1), "\n")
cat("============================================================\n\n")

# Save baseline parameters
baseline_params <- list(
 # Bumped whenever a change to the scoring math makes previously saved μ/σ
 # incomparable. run_daily.R refuses to trust a baseline stamped with an older
 # version, because standardising new ratios against an old scale silently
 # shifts every published score.
 #   2 — probability alignment / smoothing overhaul (surprise is now bounded and
 #       label-correct, so ratios centre near 1 instead of near 2.5)
 method_version = 2L,
 mu = baseline_mu,
 sd = baseline_sd,
 n_runs = N_RUNS,
 n_observations = n_observations,
 n_unique_pitchers = n_unique_pitchers,
 avg_pitchers_per_run = mean(observations_per_run$n),
 date_computed = Sys.time(),
 data_period = paste(BASELINE_START, "to", BASELINE_END),
 n_history_pitches = N_HISTORY_PITCHES,
 min_history_pitches = MIN_HISTORY_PITCHES,
 min_test_pitches = MIN_TEST_PITCHES,
 sampled_days = sampled_days,
 trim = TRIM,
 n_trimmed = n_trimmed,
 feature_names = FEATURE_NAMES,
 baseline_keys = BASELINE_KEYS
)

saveRDS(baseline_params, OUTPUT_FILE)
message("Saved baseline parameters to: ", OUTPUT_FILE)

# Also save the full results for analysis
saveRDS(all_results, "baseline_full_results.rds")
message("Saved full results to: baseline_full_results.rds")

cat("\n")
cat("============================================================\n")
cat("  Done! Use these values for Deception+ standardization:\n")
cat("============================================================\n")
cat("  Deception+ = 100 + 10 * ((ratio - ", round(baseline_mu, 4), ") / ", round(baseline_sd, 4), ")\n", sep = "")
cat("============================================================\n\n")
