# ============================================================================
# compute_baseline.R — Compute Fixed Baseline Parameters for Predict+
# ----------------------------------------------------------------------------
# Runs N random 50/50 splits on historical data (2023-2025) using per-pitcher
# models to establish stable standardization parameters (μ and σ).
#
# This creates a fixed reference point for Predict+ that:
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
N_RUNS <- 100

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

#' Train per-pitcher models and compute unpredictability ratios
#'
#' For each pitcher, trains a multinomial model on their training pitches
#' and evaluates surprise on their test pitches.
#'
#' @param df_train Training data (all pitchers)
#' @param df_test Test data (all pitchers)
#' @param min_train_pitches Minimum training pitches per pitcher
#' @param min_test_pitches Minimum test pitches per pitcher
#' @param feature_names Features to use in model
#' @param baseline_keys Features for baseline model
#' @param verbose Print progress
#' @return Data frame with per-pitcher unpredictability ratios
compute_per_pitcher_unpredictability <- function(df_train,
                                                   df_test,
                                                   min_train_pitches = 50,
                                                   min_test_pitches = 10,
                                                   feature_names = FEATURE_NAMES,
                                                   baseline_keys = BASELINE_KEYS,
                                                   verbose = TRUE) {

 # Prepare factor columns
 df_train <- df_train %>%
   mutate(
     pitch_class = factor(pitch_class),
     stand = na_safe_factor(stand),
     p_throws = na_safe_factor(p_throws),
     last_pitch_type = na_safe_factor(last_pitch_type)
   )

 df_test <- df_test %>%
   mutate(
     pitch_class = factor(pitch_class),
     stand = na_safe_factor(stand),
     p_throws = na_safe_factor(p_throws),
     last_pitch_type = na_safe_factor(last_pitch_type)
   )

 # Get pitchers with enough data in both train and test
 train_counts <- df_train %>%
   group_by(pitcher_id) %>%
   summarise(n_train = n(), .groups = "drop") %>%
   filter(n_train >= min_train_pitches)

 test_counts <- df_test %>%
   group_by(pitcher_id) %>%
   summarise(n_test = n(), .groups = "drop") %>%
   filter(n_test >= min_test_pitches)

 valid_pitchers <- intersect(train_counts$pitcher_id, test_counts$pitcher_id)

 if (length(valid_pitchers) == 0) {
   warning("No pitchers with sufficient data in both train and test")
   return(tibble())
 }

 if (verbose) message("  Processing ", length(valid_pitchers), " pitchers...")

 eps <- 1e-12
 results <- vector("list", length(valid_pitchers))

 for (i in seq_along(valid_pitchers)) {
   pid <- valid_pitchers[i]

   # Get this pitcher's data
   ptr_train <- df_train %>% filter(pitcher_id == pid)
   ptr_test <- df_test %>% filter(pitcher_id == pid)

   # Prepare features for this pitcher
   pf_tr <- prepare_features(ptr_train, feature_names)
   tr2 <- pf_tr$data
   feats <- pf_tr$features

   # Align test data with training levels
   ptr_test <- ptr_test %>%
     mutate(pitch_class = factor(pitch_class, levels = levels(tr2$pitch_class)))

   # Drop test rows with unseen pitch classes
   ptr_test <- ptr_test %>% filter(!is.na(pitch_class))
   if (nrow(ptr_test) == 0) next

   te2 <- ptr_test
   if (length(feats) > 0) {
     for (nm in feats) {
       if (nm %in% names(te2)) {
         res <- clean_one_feature(te2[[nm]])
         te2[[nm]] <- res$v
       }
     }
   }

   # Check we have enough pitch classes
   if (nlevels(droplevels(tr2$pitch_class)) < 2) next

   # Fit per-pitcher model
   form <- if (length(feats) == 0) as.formula("pitch_class ~ 1")
           else as.formula(paste("pitch_class ~", paste(feats, collapse = " + ")))

   mod <- try(nnet::multinom(form, data = tr2, trace = FALSE, maxit = 200), silent = TRUE)
   if (inherits(mod, "try-error")) next

   classes <- levels(tr2$pitch_class)

   # Model predictions on test data
   P <- as.matrix(predict(mod, newdata = te2, type = "probs"))
   if (!is.matrix(P)) {
     P <- matrix(P, nrow = nrow(te2), ncol = length(classes), dimnames = list(NULL, classes))
   }

   idx_true <- match(as.character(te2$pitch_class), classes)
   # Skip if any NA matches (shouldn't happen after filtering)
   if (any(is.na(idx_true))) next

   p_true <- P[cbind(seq_len(nrow(te2)), idx_true)]
   surp_model <- -log(pmax(p_true, eps))

   # Baseline: pitcher's own marginal pitch frequencies from training
   pitch_freqs <- table(tr2$pitch_class)
   pitch_probs <- (pitch_freqs + 1) / sum(pitch_freqs + 1)  # Add-1 smoothing

   p_base <- pitch_probs[as.character(te2$pitch_class)]
   surp_base <- -log(pmax(as.numeric(p_base), eps))

   # Aggregate for this pitcher
   results[[i]] <- tibble(
     pitcher_id = pid,
     n_train = nrow(ptr_train),
     n_test = nrow(ptr_test),
     mean_surp_model = mean(surp_model),
     mean_surp_base = mean(surp_base),
     unpredictability_ratio = mean(surp_model) / pmax(mean(surp_base), eps)
   )
 }

 bind_rows(results)
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  Predict+ Baseline Computation\n")
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
df_all <- engineer_features(raw_data)
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
   baseline_keys = BASELINE_KEYS,
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

# Compute stable μ and σ across all runs
baseline_mu <- mean(all_results$unpredictability_ratio, na.rm = TRUE)
baseline_sd <- sd(all_results$unpredictability_ratio, na.rm = TRUE)

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
 mu = baseline_mu,
 sd = baseline_sd,
 n_runs = N_RUNS,
 n_observations = n_observations,
 n_unique_pitchers = n_unique_pitchers,
 avg_pitchers_per_run = mean(observations_per_run$n),
 date_computed = Sys.time(),
 data_period = paste(BASELINE_START, "to", BASELINE_END),
 min_pitches = MIN_PITCHES,
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
cat("  Done! Use these values for Predict+ standardization:\n")
cat("============================================================\n")
cat("  Predict+ = 100 + 10 * ((ratio - ", round(baseline_mu, 4), ") / ", round(baseline_sd, 4), ")\n", sep = "")
cat("============================================================\n\n")
