# ============================================================================
# run_daily.R — Daily Predict+ Analysis for Previous Day's Games
# ----------------------------------------------------------------------------
# This script runs the Predict+ analysis for the previous day's MLB games.
# Designed to run automatically at 1am UTC via GitHub Actions.
#
# Training data logic:
#   - Before May 1: Previous season's data through end of April +
#                   all previous days in current season
#   - May 1 onwards: Current season data only
#
# Output:
#   - CSV: output/{season}/{month}/{day}.csv
#   - Social media graphics: output/{season}/{month}/social_*.png
# ============================================================================

# Source the main Predict+ functions
source("pitch_ppi.R")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Get the date to analyze (default: yesterday)
# Can be overridden by command line argument: Rscript run_daily.R 2025-06-15
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0 && !grepl("^--", args[1])) {
  TARGET_DATE <- as.Date(args[1])
} else {
  TARGET_DATE <- Sys.Date() - 1
}

# Parse optional arguments
get_arg <- function(flag, default = NULL) {
  hit <- which(args == flag)
  if (length(hit) == 1 && hit < length(args)) args[hit + 1] else default
}

# Level selection (MLB or AAA) - can be overridden by --level
LEVEL <- get_arg("--level", "MLB")

# Minimum pitches for social media graphics
MIN_PITCHES_SOCIAL <- 25

# Analysis parameters
MIN_TEST_PITCHES  <- 1     # Include all pitchers who threw that day
MIN_TOTAL_PITCHES <- 1     # Minimal filter for daily analysis
BASELINE_TYPE     <- "hybrid"  # Robust for daily data

# Features to use in the multinomial model
FEATURE_NAMES <- c(
  "balls", "strikes", "two_strikes", "ahead_in_count",
  "is_top", "outs", "score_diff",
  "base_state", "is_risp",
  "high_leverage",
  "times_through_order",
  "stand", "p_throws", "last_pitch_type",
  "o_swing_pct", "z_contact_pct", "swing_pct", "chase_contact_pct"
)

# Features to use for conditional baseline
BASELINE_KEYS <- c(
  "balls", "strikes",
  "stand", "p_throws", "two_strikes"
)

# ============================================================================
# DATE CALCULATIONS
# ============================================================================

target_year <- as.integer(format(TARGET_DATE, "%Y"))
target_month <- as.integer(format(TARGET_DATE, "%m"))
target_day <- format(TARGET_DATE, "%d")
target_month_name <- tolower(format(TARGET_DATE, "%B"))

# Determine MLB season dates
# Regular season typically starts late March and ends late September
# For simplicity, we use March 20 as season start
current_season_start <- as.Date(sprintf("%d-03-20", target_year))
previous_season_start <- as.Date(sprintf("%d-03-20", target_year - 1))
previous_season_end <- as.Date(sprintf("%d-09-30", target_year - 1))
april_end_current <- as.Date(sprintf("%d-04-30", target_year))
may_first_current <- as.Date(sprintf("%d-05-01", target_year))

# Calculate training window based on date
# Before May 1: Use previous season (through end) + current season through day before target
# May 1 onwards: Use current season only (from start through day before target)

if (TARGET_DATE < may_first_current) {
  # Before May 1: Previous season + current season through yesterday
  TRAIN_RANGES <- list(
    # Previous full season
    list(
      start = as.character(previous_season_start),
      end = as.character(previous_season_end)
    )
  )

  # Add current season data if we're past opening day
  if (TARGET_DATE > current_season_start) {
    day_before <- as.character(TARGET_DATE - 1)
    if (as.Date(day_before) >= current_season_start) {
      TRAIN_RANGES[[length(TRAIN_RANGES) + 1]] <- list(
        start = as.character(current_season_start),
        end = day_before
      )
    }
  }

  TRAINING_DESCRIPTION <- sprintf(
    "Previous season (%d) + Current season through %s",
    target_year - 1,
    format(TARGET_DATE - 1, "%b %d")
  )
} else {
  # May 1 onwards: Current season only
  TRAIN_RANGES <- list(
    list(
      start = as.character(current_season_start),
      end = as.character(TARGET_DATE - 1)
    )
  )

  TRAINING_DESCRIPTION <- sprintf(
    "Current season (%d) through %s",
    target_year,
    format(TARGET_DATE - 1, "%b %d")
  )
}

# Test period: just the target date
TEST_START <- as.character(TARGET_DATE)
TEST_END <- as.character(TARGET_DATE)

# ============================================================================
# OUTPUT PATHS
# ============================================================================

# Create output directory structure: output/{season}/{month}/
output_base <- file.path("output", target_year, target_month_name)
viz_output <- file.path(output_base, "visualizations")

if (!dir.exists(output_base)) {
  dir.create(output_base, recursive = TRUE)
  message("Created output directory: ", output_base)
}

if (!dir.exists(viz_output)) {
  dir.create(viz_output, recursive = TRUE)
}

# Output file: output/{season}/{month}/{day}.csv
OUT_CSV <- file.path(output_base, paste0(target_day, ".csv"))
OUT_MODEL <- file.path("models", sprintf("daily_%s_%s.rds", LEVEL, TARGET_DATE))

# ============================================================================
# EXECUTION
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  Predict+ Daily Analysis Pipeline\n")
cat("============================================================\n")
cat("Target Date:     ", as.character(TARGET_DATE), "\n")
cat("Level:           ", LEVEL, "\n")
cat("Training:        ", TRAINING_DESCRIPTION, "\n")
cat("Test Period:     ", TEST_START, "\n")
cat("Min Pitches:     ", MIN_PITCHES_SOCIAL, " (for social graphics)\n")
cat("Output CSV:      ", OUT_CSV, "\n")
cat("Output Visuals:  ", viz_output, "\n")
cat("============================================================\n\n")

# Set global level variables
TRAIN_LEVEL <- LEVEL
TEST_LEVEL <- LEVEL

# Load and combine training data from all ranges
message("Loading training data...")
combined_train_raw <- NULL

for (i in seq_along(TRAIN_RANGES)) {
  range_info <- TRAIN_RANGES[[i]]
  message(sprintf("  Range %d: %s to %s", i, range_info$start, range_info$end))

  cachefile <- sprintf("cache/savant_raw_%s_%s_R_%s.Rds",
                       range_info$start, range_info$end, LEVEL)

  if (file.exists(cachefile)) {
    message("    Using cached data: ", cachefile)
    range_data <- readRDS(cachefile)
  } else {
    message("    Downloading...")
    range_data <- load_statcast_range(
      range_info$start, range_info$end,
      game_type = "R", level = LEVEL, verbose = TRUE
    )
    if (nrow(range_data) > 0) {
      saveRDS(range_data, cachefile)
      message("    Cached to: ", cachefile)
    }
  }

  if (!is.null(range_data) && nrow(range_data) > 0) {
    if (is.null(combined_train_raw)) {
      combined_train_raw <- range_data
    } else {
      combined_train_raw <- dplyr::bind_rows(combined_train_raw, range_data)
    }
  }
}

if (is.null(combined_train_raw) || nrow(combined_train_raw) == 0) {
  stop("No training data found for the specified ranges")
}

message("Total training rows: ", nrow(combined_train_raw))

# Save combined training data to a temporary cache for the model
train_combined_cache <- sprintf("cache/savant_combined_train_%s_%s.Rds",
                                TARGET_DATE, LEVEL)
saveRDS(combined_train_raw, train_combined_cache)

# Load test data (single day)
message("\nLoading test data for ", TARGET_DATE, "...")
test_cachefile <- sprintf("cache/savant_raw_%s_%s_R_%s.Rds",
                          TEST_START, TEST_END, LEVEL)

if (file.exists(test_cachefile)) {
  message("Using cached test data: ", test_cachefile)
  raw_test <- readRDS(test_cachefile)
} else {
  message("Downloading test data...")
  raw_test <- load_statcast_range(TEST_START, TEST_END,
                                   game_type = "R", level = LEVEL, verbose = TRUE)
  if (nrow(raw_test) > 0) {
    saveRDS(raw_test, test_cachefile)
    message("Cached test data to: ", test_cachefile)
  }
}

if (is.null(raw_test) || nrow(raw_test) == 0) {
  message("No games found for ", TARGET_DATE, ". This may be an off-day.")
  quit(save = "no", status = 0)
}

message("Test data: ", nrow(raw_test), " pitches")

# Engineer features
message("\nEngineering features...")
df_train <- engineer_features(combined_train_raw)
df_train <- df_train %>% dplyr::filter(!is.na(pitcher_id))

df_test <- engineer_features(raw_test)
df_test <- df_test %>% dplyr::filter(!is.na(pitcher_id))

message("Training features: ", nrow(df_train), " pitches from ",
        length(unique(df_train$pitcher_id)), " pitchers")
message("Test features: ", nrow(df_test), " pitches from ",
        length(unique(df_test$pitcher_id)), " pitchers")

# Prepare training data
df_train <- df_train %>% dplyr::mutate(
  pitch_class     = factor(pitch_class),
  stand           = na_safe_factor(stand),
  p_throws        = na_safe_factor(p_throws),
  last_pitch_type = na_safe_factor(last_pitch_type)
)

pf_tr <- prepare_features(df_train, FEATURE_NAMES)
tr2   <- pf_tr$data
feats <- pf_tr$features

# Fit model
message("\nFitting multinomial model...")
form <- if (length(feats) == 0) as.formula("pitch_class ~ 1")
        else as.formula(paste("pitch_class ~", paste(feats, collapse = " + ")))

mod <- nnet::multinom(form, data = tr2, trace = FALSE, maxit = 500)
classes <- levels(tr2$pitch_class)
message("Model trained with ", length(classes), " pitch classes")

# Prepare test data
df_test <- df_test %>% dplyr::mutate(
  pitch_class     = factor(pitch_class, levels = levels(df_train$pitch_class)),
  stand           = na_safe_factor(stand),
  p_throws        = na_safe_factor(p_throws),
  last_pitch_type = na_safe_factor(last_pitch_type)
)

te2 <- df_test
if (length(feats) > 0) {
  for (nm in feats) {
    res <- clean_one_feature(te2[[nm]])
    te2[[nm]] <- res$v
  }
}

# Model predictions
message("Evaluating test pitches...")
P <- as.matrix(predict(mod, newdata = te2, type = "probs"))
if (!is.matrix(P)) {
  P <- matrix(P, nrow = nrow(te2), ncol = length(classes), dimnames = list(NULL, classes))
}

idx_true <- match(as.character(te2$pitch_class), classes)
eps <- 1e-12
p_true <- P[cbind(seq_len(nrow(te2)), idx_true)]
surp_model <- -log(pmax(p_true, eps))

# Baseline predictions
message("Computing baseline...")
P_base <- compute_baseline_probs(tr2, te2, baseline_type = BASELINE_TYPE, baseline_keys = BASELINE_KEYS)
p_true_base <- P_base[cbind(seq_len(nrow(te2)), idx_true)]
surp_base <- -log(pmax(p_true_base, eps))

# Per-pitcher aggregation
message("Aggregating by pitcher...")
per_pitcher_test <- te2 %>%
  dplyr::select(pitcher_id) %>%
  dplyr::mutate(surp_model = surp_model, surp_base = surp_base) %>%
  dplyr::group_by(pitcher_id) %>%
  dplyr::summarise(
    n_pitches_test  = dplyr::n(),
    mean_surp_model = mean(surp_model),
    mean_surp_base  = mean(surp_base),
    .groups = "drop"
  ) %>%
  dplyr::filter(n_pitches_test >= MIN_TEST_PITCHES) %>%
  dplyr::mutate(
    ppi = 1 - (mean_surp_model / pmax(mean_surp_base, 1e-9)),
    ppi = pmin(pmax(ppi, -1), 1),
    unpredictability_ratio = mean_surp_model / pmax(mean_surp_base, 1e-9)
  )

# Total pitches for the day
all_pitchers <- df_test %>%
  dplyr::filter(!is.na(pitcher_id)) %>%
  dplyr::group_by(pitcher_id) %>%
  dplyr::summarise(total_pitches = dplyr::n(), .groups = "drop")

# Resolve names
message("Resolving pitcher names...")
all_pitcher_ids <- df_test %>%
  dplyr::filter(!is.na(pitcher_id)) %>%
  dplyr::distinct(pitcher_id)
name_map <- resolve_pitcher_names_with_fallback(all_pitcher_ids,
                                                 cache_file = "cache/mlbam_name_cache.csv",
                                                 verbose = TRUE)

# Calculate Predict+
u_mu <- mean(per_pitcher_test$unpredictability_ratio, na.rm = TRUE)
u_sd <- sd(per_pitcher_test$unpredictability_ratio, na.rm = TRUE)

pitcher_ppi <- all_pitchers %>%
  dplyr::left_join(per_pitcher_test, by = "pitcher_id") %>%
  dplyr::left_join(name_map, by = "pitcher_id") %>%
  dplyr::filter(!is.na(ppi)) %>%
  dplyr::mutate(
    pitcher_name = dplyr::if_else(is.na(pitcher_name), paste0("Pitcher_", pitcher_id), pitcher_name),
    predict_plus = 100 + 10 * ((unpredictability_ratio - u_mu) / pmax(u_sd, 1e-12))
  ) %>%
  dplyr::select(
    pitcher_id, pitcher_name, total_pitches, n_pitches_test,
    mean_surp_model, mean_surp_base, ppi,
    unpredictability_ratio, predict_plus
  ) %>%
  dplyr::arrange(dplyr::desc(predict_plus))

# Save results
message("\nSaving results...")
readr::write_csv(pitcher_ppi, OUT_CSV)
message("CSV saved: ", OUT_CSV)

# Create result object for visualization functions
res <- list(
  model = mod,
  train = df_train,
  test = df_test,
  classes = classes,
  pitcher_ppi = pitcher_ppi,
  features_used = feats,
  baseline_keys = BASELINE_KEYS,
  baseline_type = BASELINE_TYPE,
  split_method = "temporal",
  train_period = TRAINING_DESCRIPTION,
  test_period = as.character(TARGET_DATE)
)

# Save model metadata
saveRDS(list(
  classes = res$classes,
  features_used = res$features_used,
  baseline_keys = res$baseline_keys,
  baseline_type = res$baseline_type,
  split_method = "temporal",
  train_description = TRAINING_DESCRIPTION,
  test_date = as.character(TARGET_DATE),
  level = LEVEL
), OUT_MODEL)
message("Model saved: ", OUT_MODEL)

# ============================================================================
# SOCIAL MEDIA GRAPHICS
# ============================================================================

message("\nGenerating social media graphics...")
create_social_media_graphics(
  res = res,
  game_date = as.character(TARGET_DATE),
  min_pitches = MIN_PITCHES_SOCIAL,
  output_dir = viz_output,
  top_n = 5
)

# ============================================================================
# SUMMARY
# ============================================================================

cat("\n")
cat("============================================================\n")
cat("  Daily Analysis Complete!\n")
cat("============================================================\n")
cat("Date:            ", as.character(TARGET_DATE), "\n")
cat("Level:           ", LEVEL, "\n")
cat("Pitchers:        ", nrow(pitcher_ppi), "\n")
cat("Total pitches:   ", sum(pitcher_ppi$n_pitches_test), "\n")
cat("\n")

# Show top 5 and bottom 5 with >= MIN_PITCHES_SOCIAL
qualified <- pitcher_ppi %>%
  dplyr::filter(n_pitches_test >= MIN_PITCHES_SOCIAL)

if (nrow(qualified) > 0) {
  cat("Top 5 Least Predictable (>= ", MIN_PITCHES_SOCIAL, " pitches):\n", sep = "")
  top5 <- qualified %>%
    dplyr::arrange(dplyr::desc(predict_plus)) %>%
    head(5) %>%
    dplyr::mutate(rank = dplyr::row_number()) %>%
    dplyr::select(rank, pitcher_name, n_pitches_test, predict_plus)
  print(as.data.frame(top5), row.names = FALSE)

  cat("\nTop 5 Most Predictable (>= ", MIN_PITCHES_SOCIAL, " pitches):\n", sep = "")
  bottom5 <- qualified %>%
    dplyr::arrange(predict_plus) %>%
    head(5) %>%
    dplyr::mutate(rank = dplyr::row_number()) %>%
    dplyr::select(rank, pitcher_name, n_pitches_test, predict_plus)
  print(as.data.frame(bottom5), row.names = FALSE)
} else {
  cat("No pitchers with >= ", MIN_PITCHES_SOCIAL, " pitches today.\n")
}

cat("\n")
cat("============================================================\n")
cat("Output files:\n")
cat("  CSV:     ", OUT_CSV, "\n")
cat("  Model:   ", OUT_MODEL, "\n")
cat("  Visuals: ", viz_output, "/\n", sep = "")
cat("============================================================\n\n")
