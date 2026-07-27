# ============================================================================
# compute_baseline.R — Compute Fixed Baseline Parameters for Deception+
# ----------------------------------------------------------------------------
# Runs N random 50/50 splits on historical data (2023-2025) using per-pitcher
# models to establish stable standardization parameters (μ and σ).
#
# This creates a fixed reference point for Deception+ that:
#   - Defines what "100" means (average MLB pitcher unpredictability)
#   - Is stable across all future runs
#   - Should be recomputed periodically (e.g., annually)
#
# Output: baseline_params.rds containing:
#   - mu: mean unpredictability_ratio across all pitchers and runs
#   - sd: standard deviation of unpredictability_ratio
#   - n_runs: number of random splits performed
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

# Number of random splits to average
N_RUNS <- 30

# Minimum pitches per pitcher to include in baseline
MIN_PITCHES <- 100

# Output file
OUTPUT_FILE <- "baseline_params.rds"

# Features for per-pitcher models
FEATURE_NAMES <- c(
 "balls", "strikes", "two_strikes", "ahead_in_count",
 "outs", "is_risp", "stand", "last_pitch_type"
)

# Baseline keys for simple baseline comparison
BASELINE_KEYS <- c("balls", "strikes", "stand", "two_strikes")

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
cat("Number of Runs:  ", N_RUNS, "\n")
cat("Min Pitches:     ", MIN_PITCHES, "\n")
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

# Run N random splits
message("Running ", N_RUNS, " random splits with per-pitcher models...\n")

all_ratios <- vector("list", N_RUNS)

for (run in seq_len(N_RUNS)) {
 if (run %% 10 == 1 || run == N_RUNS) {
   message(sprintf("Run %d/%d...", run, N_RUNS))
 }

 # Set seed for reproducibility
 set.seed(run)

 # Random 50/50 split per pitcher
 df_split <- df_all %>%
   group_by(pitcher_id) %>%
   mutate(
     random_order = sample(n()),
     is_train = random_order <= ceiling(n() / 2)
   ) %>%
   ungroup()

 df_train <- df_split %>% filter(is_train) %>% select(-random_order, -is_train)
 df_test <- df_split %>% filter(!is_train) %>% select(-random_order, -is_train)

 # Compute per-pitcher unpredictability
 run_results <- compute_per_pitcher_unpredictability(
   df_train, df_test,
   min_train_pitches = MIN_PITCHES / 2,  # Half goes to train
   min_test_pitches = MIN_PITCHES / 2,   # Half goes to test
   feature_names = FEATURE_NAMES,
   verbose = FALSE
 )

 if (nrow(run_results) > 0) {
   run_results$run <- run
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
 min_pitches = MIN_PITCHES,
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
