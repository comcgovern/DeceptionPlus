# ============================================================================
# pitch_ppi.R — Pitch Type Predictability (PPI) with Flexible Period Selection
# ----------------------------------------------------------------------------
# - Downloads (and caches) Statcast pitches via sabRmetrics::download_baseballsavant()
# - Resolves pitcher names from MLB StatsAPI with on-disk cache; baseballr fallback
# - Trains multinomial model on one period; evaluates on another (can be same/overlapping)
# - Outputs per-pitcher table with PPI + Deception+ (avg=100; 10 pts = 1 SD)
# - Supports multiple baseline models: "marginal", "conditional", "hybrid"
# - Organizes outputs into subfolders: cache/, models/, output/
# ============================================================================

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(purrr); library(stringr); library(lubridate)
  library(nnet); library(readr); library(tibble); library(forcats)
})

`%||%` <- function(a, b) if (!is.null(a)) a else b
safe_div <- function(a, b) ifelse(b > 0, a / b, 0)

# Safe wrapper for multinom predict.
#
# predict.multinom() is deceptively shape-shifting and the naive handling of it
# silently corrupts the metric.  Two traps:
#
#   1. nnet::multinom() DROPS response levels with zero observations (it warns
#      "groups ... are empty").  A pitcher's pitch_class factor usually carries
#      the league-wide level set, so the fitted model returns FEWER columns than
#      `classes` for essentially every pitcher.
#   2. With exactly two fitted levels, predict(type="probs") returns a bare
#      vector of P(second level); with a single test row it returns a bare
#      *named* vector over levels instead.
#
# The previous fallback reshaped the prediction with matrix(P, ncol=length(classes)),
# which recycles the flattened probabilities across columns — attaching the wrong
# class label to every probability and producing rows that do not sum to 1.
# Everything downstream (-log p) then measured noise rather than surprise.
#
# This version aligns strictly BY CLASS NAME and leaves classes the model never
# saw at zero, for the caller to smooth against a prior.
safe_predict_probs <- function(mod, newdata, classes) {
  n <- nrow(newdata)
  # Levels the model was actually fitted on — never assume these match `classes`.
  fit_lev <- mod$lev %||% classes
  P <- predict(mod, newdata = newdata, type = "probs")

  if (!is.matrix(P)) {
    if (n == 1L && length(P) == length(fit_lev) && identical(names(P), fit_lev)) {
      # Single test row, >2 fitted levels: named vector over levels.
      P <- matrix(as.numeric(P), nrow = 1L, dimnames = list(NULL, fit_lev))
    } else if (length(fit_lev) == 2L && length(P) == n) {
      # Two fitted levels: vector of P(second level).
      p2 <- as.numeric(P)
      P <- cbind(1 - p2, p2)
      colnames(P) <- fit_lev
    } else {
      stop("Unrecognised predict.multinom() shape: length ", length(P),
           " for ", n, " rows and ", length(fit_lev), " fitted levels.")
    }
  }

  src <- colnames(P)
  if (is.null(src)) {
    if (ncol(P) != length(fit_lev)) {
      stop("Unlabelled prediction matrix with ", ncol(P), " columns but ",
           length(fit_lev), " fitted levels; cannot align safely.")
    }
    src <- fit_lev
  }

  out <- matrix(0, nrow = n, ncol = length(classes), dimnames = list(NULL, classes))
  shared <- intersect(src, classes)
  if (length(shared) == 0L) {
    stop("Fitted classes (", paste(src, collapse = ", "),
         ") share no levels with the requested classes (",
         paste(classes, collapse = ", "), ").")
  }
  out[, shared] <- P[, shared, drop = FALSE]
  out
}

# Laplace-smoothed marginal distribution of `y` over `classes`.
# Used both as the fallback for unseen contexts and as the shrinkage target when
# turning model probabilities into surprise.
class_prior <- function(y, classes, alpha = 1) {
  cnt <- as.numeric(table(factor(as.character(y), levels = classes)))
  (cnt + alpha) / sum(cnt + alpha)
}

# Shrink a probability matrix toward `prior` before it is fed to -log().
#
# Two jobs:
#   • Repair rows the model could not score (NA from predict()'s na.omit, or all
#     zeros because the row's class was never fitted) by falling back to `prior`.
#   • Bound the surprise.  An unpenalised multinomial fit on a few hundred
#     pitches routinely hits complete separation and emits p ≈ 1e-15, which the
#     old eps=1e-9 clamp turned into a flat 20.7 nats — an order of magnitude
#     above anything the Laplace-smoothed baseline can produce.  Because the
#     metric is a RATIO of the two surprises, that asymmetry alone manufactured
#     Deception+ scores in the hundreds.  Mixing in a little prior mass floors
#     both numerator and denominator on the same scale and keeps the score a
#     proper scoring rule.
shrink_to_prior <- function(P, prior, lambda = 0.02) {
  stopifnot(ncol(P) == length(prior))
  P[!is.finite(P)] <- 0
  P[P < 0] <- 0
  prior_mat <- matrix(prior, nrow = nrow(P), ncol = length(prior), byrow = TRUE)
  rs <- rowSums(P)
  bad <- !is.finite(rs) | rs <= 0
  if (any(bad)) {
    P[bad, ] <- prior_mat[bad, , drop = FALSE]
    rs[bad] <- 1
  }
  P <- P / rs
  if (lambda > 0) {
    P <- (1 - lambda) * P + lambda * prior_mat
    P <- P / rowSums(P)
  }
  P
}

# Scoring-method version. Bump whenever a change makes previously saved
# baseline_params.rds μ/σ incomparable, so run_daily.R can refuse to publish
# against a stale scale and the compute-baseline workflow knows to regenerate.
#   1 — original
#   2 — probability alignment / smoothing overhaul (surprise bounded and
#       label-correct, so ratios centre near 1 instead of near 2.5)
#   3 — count nests the baseline; calibration mirrors production; Surprise+ added
#       (baseline_params.rds now needs surprise_mu / surprise_sd as well)
BASELINE_METHOD_VERSION <- 3L

# ---------------------- The two unpredictability scales ----------------------
#
# Deception+ and Surprise+ answer different questions and need different amounts
# of data. Both are reported; neither replaces the other.
#
#   Deception+   standardises `unpredictability_ratio` = S_model / S_baseline.
#                "Does this pitcher defy prediction BEYOND what the count and
#                handedness already give away?" Almost perfectly independent of
#                arsenal size (R² vs pitch-type count ≈ 0.03), which is what makes
#                a two-pitch reliever able to top the board. But it is a
#                season-scale statistic: reliability (between-pitcher variance
#                over total variance) is ~0.16 on a 20-pitch outing, ~0.27 at 100
#                pitches, ~0.79 at 1500. Do not rank a single day by it.
#
#   Surprise+    standardises normalised surprise = S_model / log(n_classes).
#                "Of all the uncertainty this pitcher's arsenal could create, how
#                much survives once you know the situation?" ~1.0 means their next
#                pitch is close to a coin flip among their own offerings; 0.3 means
#                three-quarters of the uncertainty is gone. Reliability ~0.92 on a
#                20-pitch outing, so it can carry a daily leaderboard. Dividing by
#                log(n_classes) rather than reporting raw nats is what keeps a
#                two-pitch pitcher comparable to a five-pitch one; raw surprise is
#                ~64% arsenal size, normalised surprise ~27%.
#
# Both are scaled to mean 100 / SD 10 over their reference population.

# Normalised surprise: nats of surprise as a share of the most a given arsenal
# could deliver. log(1) = 0, so a single-pitch pitcher would divide by zero —
# floored at 2 classes, which reports them as near-zero surprise (correct: if
# there is only one pitch, there is nothing to guess).
normalized_surprise <- function(mean_surp_model, n_classes) {
  mean_surp_model / log(pmax(n_classes, 2))
}

# 100 + 10 * z, guarding a degenerate or missing spread.
scale_plus <- function(x, mu, sd) {
  100 + 10 * ((x - mu) / pmax(sd, 1e-9))
}

# ---------------------- Directory Setup --------------------------------------
ensure_directories <- function() {
  dirs <- c("cache", "models", "output", "output/visualizations")
  for (d in dirs) {
    if (!dir.exists(d)) {
      dir.create(d, recursive = TRUE)
      message("✓ Created directory: ", d)
    }
  }
}

# ---------------------- Statcast Loader (MLB + AAA with proper chunking) -----
load_statcast_range <- function(start_date, end_date, game_type = "R", level = "MLB", verbose = TRUE) {
  
  # For MLB, use sabRmetrics (it works perfectly)
  if (level == "MLB") {
    if (!requireNamespace("sabRmetrics", quietly = TRUE)) {
      stop("Please install 'sabRmetrics': devtools::install_github('saberpowers/sabRmetrics')")
    }
    if (verbose) message("Downloading Savant (MLB): ", start_date, " -> ", end_date, " | game_type=", game_type)

    # sabRmetrics::download_baseballsavant() does not support cross-year queries.
    # Split into per-year chunks and combine.
    start_d <- as.Date(start_date)
    end_d   <- as.Date(end_date)
    years   <- seq(as.integer(format(start_d, "%Y")), as.integer(format(end_d, "%Y")))

    year_chunks <- lapply(years, function(yr) {
      yr_start <- max(start_d, as.Date(sprintf("%d-01-01", yr)))
      yr_end   <- min(end_d,   as.Date(sprintf("%d-12-31", yr)))
      if (verbose) message("  Fetching year ", yr, ": ", yr_start, " -> ", yr_end)
      chunk <- try(sabRmetrics::download_baseballsavant(
        start_date = as.character(yr_start),
        end_date   = as.character(yr_end),
        game_type  = game_type,
        cl         = NULL,
        verbose    = verbose
      ), silent = TRUE)
      if (inherits(chunk, "try-error") || is.null(chunk) || nrow(chunk) == 0) {
        if (verbose) message("    No data for ", yr)
        return(NULL)
      }
      tibble::as_tibble(chunk)
    })

    year_chunks <- Filter(Negate(is.null), year_chunks)

    if (length(year_chunks) == 0) {
      warning("No Savant rows returned for this window.")
      return(tibble())
    }
    return(dplyr::bind_rows(year_chunks))
  }
  
  # For AAA, use minors endpoint with sabRmetrics-style chunking
  if (verbose) message("Downloading Savant (AAA): ", start_date, " -> ", end_date, " | game_type=", game_type)
  
  if (!requireNamespace("httr", quietly = TRUE)) {
    stop("Please install 'httr': install.packages('httr')")
  }
  if (!requireNamespace("readr", quietly = TRUE)) {
    stop("Please install 'readr': install.packages('readr')")
  }
  
  # Split into 5-day chunks (same as sabRmetrics strategy)
  start <- as.Date(start_date)
  end <- as.Date(end_date)
  days <- as.numeric(end - start)
  
  # Build payload of URLs for each chunk
  payload <- tibble::tibble(
    start_chunk = seq(start, by = 5, length.out = ceiling((days + 1) / 5))
  ) %>%
    dplyr::mutate(
      end_chunk = pmin(.data$start_chunk + 4, end),
      chunk_id = dplyr::row_number()
    )
  
  # Build URLs for each chunk
  base_url <- "https://baseballsavant.mlb.com/statcast-search-minors/csv"
  
  payload <- payload %>%
    dplyr::mutate(
      game_type_filter = paste0("hfGT=", game_type, "%7C"),
      date_filter = sprintf("game_date_gt=%s&game_date_lt=%s", .data$start_chunk, .data$end_chunk),
      level_filter = "hfLevel=AAA%7C",
      season_filter = paste0("hfSea=", format(start, "%Y"), "%7C"),
      url = paste0(
        base_url,
        "?all=true",
        "&type=details",
        "&minors=true",
        "&player_type=pitcher",
        "&", .data$game_type_filter,
        "&", .data$date_filter,
        "&", .data$level_filter,
        "&", .data$season_filter,
        "&group_by=name",
        "&min_pitches=0",
        "&min_results=0"
      )
    )
  
  n_chunks <- nrow(payload)
  if (verbose) message("Downloading ", n_chunks, " chunk(s) (5-day periods)...")
  
  # Step 1: Submit initial requests (like sabRmetrics does)
  if (verbose) message("Submitting initial API requests...")
  initial_requests <- lapply(payload$url, function(url) {
    try(httr::GET(url, httr::timeout(1)), silent = TRUE)
  })
  
  # Step 2: Download with proper timeout, retrying as needed
  if (verbose) message("Downloading data chunks...")
  
  data_list <- vector("list", n_chunks)
  names(data_list) <- paste0("chunk_", payload$chunk_id)
  is_error <- rep(TRUE, n_chunks)
  
  max_retries <- 3
  retry_count <- 0
  
  while (any(is_error) && retry_count < max_retries) {
    retry_count <- retry_count + 1
    
    if (retry_count > 1 && verbose) {
      message("Retry attempt ", retry_count, " for ", sum(is_error), " chunk(s)...")
    }
    
    for (i in which(is_error)) {
      if (verbose) message(sprintf("  Chunk %d/%d: %s to %s", 
                                    i, n_chunks, 
                                    payload$start_chunk[i], 
                                    payload$end_chunk[i]))
      
      response <- try(httr::GET(payload$url[i], httr::timeout(120)), silent = TRUE)
      
      if (inherits(response, "try-error")) {
        if (verbose) message("    ✗ Connection error")
        next
      }
      
      if (httr::http_error(response)) {
        if (verbose) message("    ✗ HTTP ", httr::status_code(response))
        next
      }
      
      content <- httr::content(response, as = "text", encoding = "UTF-8")
      
      if (nchar(content) < 100) {
        if (verbose) message("    • No data (likely no games)")
        data_list[[i]] <- NULL
        is_error[i] <- FALSE
        next
      }
      
      if (grepl("<html", content, ignore.case = TRUE)) {
        if (verbose) message("    ✗ Got HTML error page")
        next
      }
      
      chunk_data <- try(readr::read_csv(content, show_col_types = FALSE), silent = TRUE)
      
      if (inherits(chunk_data, "try-error")) {
        if (verbose) message("    ✗ CSV parse error")
        next
      }
      
      if (nrow(chunk_data) == 0) {
        if (verbose) message("    • No data")
        data_list[[i]] <- NULL
        is_error[i] <- FALSE
        next
      }
      
      data_list[[i]] <- chunk_data
      is_error[i] <- FALSE
      if (verbose) message("    ✓ ", nrow(chunk_data), " rows")
      
      if (nrow(chunk_data) == 25000) {
        warning(sprintf("Chunk %d returned exactly 25,000 rows - data may be truncated", i))
      }
      
      Sys.sleep(1)
    }
  }
  
  if (any(is_error)) {
    warning(sprintf("%d chunk(s) failed after %d retries", sum(is_error), max_retries))
  }
  
  successful_data <- data_list[!sapply(data_list, is.null)]
  
  if (length(successful_data) == 0) {
    warning("No AAA data returned for this window.")
    return(tibble())
  }
  
  combined <- dplyr::bind_rows(successful_data)
  
  chunk_sizes <- sapply(successful_data, nrow)
  if (any(chunk_sizes == 25000)) {
    n_at_limit <- sum(chunk_sizes == 25000)
    warning(sprintf("%d chunk(s) returned exactly 25,000 rows. Data are likely missing.", n_at_limit))
  }
  
  if (verbose) message("✓ Total AAA rows: ", nrow(combined))
  
  tibble::as_tibble(combined)
}

# ---------------------- Feature helpers --------------------------------------
canonical_pitch <- function(pt) {
  wl <- c("FF","SI","FT","FC","FS","CH","SL","CU","KC","ST","SV","CS","KN","FO")
  if (is.na(pt) || pt == "") return("OTHER")
  up <- toupper(pt); ifelse(up %in% wl, up, "OTHER")
}

base_state_row <- function(on1, on2, on3) {
  b1 <- ifelse(!is.na(on1), 1L, 0L)
  b2 <- ifelse(!is.na(on2), 1L, 0L)
  b3 <- ifelse(!is.na(on3), 1L, 0L)
  b3 * 4L + b2 * 2L + b1
}

# Vectorized versions for use on whole columns
is_contact_vec <- function(desc) {
  grepl("foul|hit_into_play|foul_tip", tolower(desc)) & !is.na(desc)
}
is_swing_vec <- function(desc) {
  (grepl("swinging", tolower(desc)) | is_contact_vec(desc)) & !is.na(desc)
}
in_strike_zone_vec <- function(zone) {
  z <- suppressWarnings(as.integer(zone))
  # NA for unknown zone: an unlocated pitch is neither in nor out of the zone.
  # Returning FALSE (the old behaviour) silently counted every unlocated pitch
  # as out-of-zone, inflating the o_swing_pct / chase_contact_pct denominators.
  ifelse(is.na(z), NA, z >= 1L & z <= 9L)
}

compute_batter_metrics <- function(df) {
  if (!"zone" %in% names(df)) df$zone <- NA
  if (!"description" %in% names(df)) df$description <- NA_character_
  sub <- df %>% transmute(
    batter   = .data$batter,
    in_zone  = in_strike_zone_vec(.data$zone),
    swing    = is_swing_vec(.data$description),
    contact  = is_contact_vec(.data$description),
    out_zone = !in_zone
  )
  sub %>% group_by(batter) %>% summarise(
    pitches_seen      = n(),
    pitches_out_zone  = sum(out_zone, na.rm = TRUE),
    swings            = sum(swing, na.rm = TRUE),
    swings_in_zone    = sum(swing & in_zone, na.rm = TRUE),
    swings_out_zone   = sum(swing & out_zone, na.rm = TRUE),
    contact_in_zone   = sum(contact & in_zone, na.rm = TRUE),
    contact_out_zone  = sum(contact & out_zone, na.rm = TRUE),
    .groups = "drop"
  ) %>% mutate(
    o_swing_pct       = safe_div(swings_out_zone, pitches_out_zone),
    z_contact_pct     = safe_div(contact_in_zone, pmax(swings_in_zone, 1)),
    swing_pct         = safe_div(swings, pitches_seen),
    chase_contact_pct = safe_div(contact_out_zone, pmax(swings_out_zone, 1))
  ) %>% select(batter, o_swing_pct, z_contact_pct, swing_pct, chase_contact_pct)
}

synthesize_pitch_type <- function(df) {
  if (!"pitch_type" %in% names(df)) df$pitch_type <- NA_character_
  if (!"pitch_name" %in% names(df)) df$pitch_name <- NA_character_
  if (!"mlb_pitch_name" %in% names(df)) df$mlb_pitch_name <- NA_character_
  name_col <- dplyr::coalesce(df$pitch_name, df$mlb_pitch_name)
  idx <- which(is.na(df$pitch_type) & !is.na(name_col))
  if (length(idx) == 0) return(df)
  nms <- tolower(name_col[idx]); map <- function(p) grepl(p, nms, perl = TRUE, ignore.case = TRUE)
  pt <- df$pitch_type
  pt[idx][ map("4[ -]?seam|four[ -]?seam|fourseam|4-seam") ] <- "FF"
  pt[idx][ map("\\btwo[ -]?seam\\b|\\b2[ -]?seam\\b|twoseam|2-seam") ] <- "FT"
  pt[idx][ map("\\bsinker\\b") ] <- "SI"; pt[idx][ map("\\bcutter\\b") ] <- "FC"; pt[idx][ map("split|splitter") ] <- "FS"
  pt[idx][ map("change|chg|change[- ]?up") ] <- "CH"; pt[idx][ map("\\bslider\\b|sweeper") ] <- "SL"
  pt[idx][ map("curveball|\\bcurve\\b|slow curve") ] <- "CU"; pt[idx][ map("knuckle[ -]?curve|\\bkc\\b") ] <- "KC"
  pt[idx][ map("\\bslurve\\b") ] <- "SV"; pt[idx][ map("\\bknuckleball\\b") ] <- "KN"; pt[idx][ map("\\bfork\\b|forkball") ] <- "FO"
  df$pitch_type <- pt; df
}

add_last_pitch <- function(df) {
  if (!"at_bat_number" %in% names(df)) df$at_bat_number <- NA
  if (!"pitch_number" %in% names(df)) df$pitch_number <- NA
  # Lag pitch_class, not the raw pitch_type.  pitch_class is the canonicalised
  # vocabulary the response uses (rare codes collapsed to "OTHER"); the raw
  # pitch_type carries oddities like "EP"/"PO"/"FA" that appear in the test
  # period but not in a given pitcher's training window.  Those became unseen
  # factor levels -> NA -> the whole test pitch was dropped by complete.cases().
  lag_src <- if ("pitch_class" %in% names(df)) df$pitch_class else df$pitch_type
  # Group by pitcher within each game so that the last pitch of one PA carries
  # into the first pitch of the next PA.  The old approach (grouping by at_bat_number)
  # reset to "NONE" at every PA boundary, silencing sequence context for ~50% of pitches.
  df %>%
    mutate(.lag_src = lag_src) %>%
    arrange(.data$game_pk, .data$pitcher_id, .data$at_bat_number, .data$pitch_number) %>%
    group_by(.data$game_pk, .data$pitcher_id) %>%
    mutate(last_pitch_type = dplyr::lag(.data$.lag_src)) %>%
    ungroup() %>%
    select(-".lag_src") %>%
    mutate(last_pitch_type = if_else(
      is.na(.data$last_pitch_type) | .data$last_pitch_type == "" | .data$last_pitch_type == "NA",
      "NONE", toupper(.data$last_pitch_type)
    ))
}

# ---------------------- Pitcher ID coalescer ---------------------------------
coalesce_pitcher_id <- function(df) {
  id_cols <- c("pitcher", "pitcher_id", "pitcher_mlbam", "pitcherId",
               "pitcher.1", "player_id_pitcher", "mlbam_pitcher_id")
  for (nm in id_cols) if (!nm %in% names(df)) df[[nm]] <- NA
  id_vec <- Reduce(function(x, y) dplyr::coalesce(x, y), df[id_cols])
  id_num <- suppressWarnings(as.numeric(id_vec))
  ifelse(is.finite(id_num), id_num, NA_real_)
}

# ---------------------- Feature engineering ----------------------------------
engineer_features <- function(raw, include_batter_metrics = TRUE) {
  if (is.null(raw) || nrow(raw) == 0) return(tibble())
  core <- c("game_date","game_pk","batter","pitch_type","balls","strikes",
            "outs_when_up","inning","inning_topbot","on_1b","on_2b","on_3b",
            "home_score","away_score","stand","p_throws","description","zone")
  for (nm in core) if (!nm %in% names(raw)) raw[[nm]] <- NA
  
  raw$pitcher_id <- coalesce_pitcher_id(raw)
  raw <- synthesize_pitch_type(raw)
  
  df <- raw %>% filter(!is.na(.data$pitch_type))
  if (nrow(df) == 0) return(tibble())
  
  df <- df %>% mutate(
    pitch_class     = vapply(.data$pitch_type, canonical_pitch, character(1)),
    is_top          = if_else(stringr::str_to_upper(.data$inning_topbot) == "TOP", 1L, 0L),
    inning          = suppressWarnings(as.integer(.data$inning)),  # Keep for leverage calculation
    outs            = suppressWarnings(as.integer(coalesce(.data$outs_when_up, 0))),
    balls           = suppressWarnings(as.integer(coalesce(.data$balls, 0))),
    strikes         = suppressWarnings(as.integer(coalesce(.data$strikes, 0))),
    two_strikes     = if_else(.data$strikes == 2L, 1L, 0L),
    ahead_in_count  = if_else(.data$strikes > .data$balls, 1L, 0L),
    # Joint count state as a 12-level factor.
    #
    # `balls` and `strikes` as separate numeric terms force the model to be
    # linear in the logit, and count effects are emphatically not linear — 3-0 is
    # a fastball count, 0-2 is a breaking-ball count, and no monotone function of
    # (balls, strikes) captures both. More importantly, the conditional baseline
    # is a saturated cross-tab over count cells, so a linear-in-count model is
    # STRICTLY LESS expressive than the baseline it is scored against. That made
    # the ratio rise with predictability over most of its range. Modelling the
    # count jointly makes the full model nest the baseline, which is what the
    # comparison assumes.
    count           = factor(paste0(pmin(pmax(.data$balls, 0L), 3L), "-",
                                    pmin(pmax(.data$strikes, 0L), 2L)),
                             levels = as.vector(t(outer(0:3, 0:2, paste, sep = "-")))),
    # Ordinal with three values; a factor costs one extra dummy and drops the
    # unwarranted "2 outs is twice 1 out" assumption.
    outs            = factor(.data$outs, levels = c(0L, 1L, 2L)),
    # Factor, not numeric: the 0-7 code is a bit-mask of occupied bases, so
    # "runner on 3rd" (4) is not four times "runner on 1st" (1).  Treating it as
    # a continuous predictor — as the old code did — forced a nonsensical linear
    # ordering on the base-out state.  METHODOLOGY.md always described it as an
    # 8-level categorical.
    base_state      = factor(base_state_row(.data$on_1b, .data$on_2b, .data$on_3b),
                             levels = 0:7),
    is_risp         = if_else(!is.na(.data$on_2b) | !is.na(.data$on_3b), 1L, 0L),
    # score_diff from the pitcher's perspective:
    # positive = pitcher's team leads, negative = pitcher's team trails
    # TOP of inning: visiting team bats, HOME pitcher is on the mound
    # BOT of inning: home team bats, VISITING pitcher is on the mound
    score_diff      = if_else(
      stringr::str_to_upper(.data$inning_topbot) == "TOP",
      coalesce(.data$home_score, 0) - coalesce(.data$away_score, 0),  # home pitcher (pitches in top half)
      coalesce(.data$away_score, 0) - coalesce(.data$home_score, 0)   # visiting pitcher (pitches in bottom half)
    ),
    stand           = coalesce(.data$stand, "R"),
    p_throws        = coalesce(.data$p_throws, "R"),
    game_date       = as_datetime(.data$game_date),
    # High leverage: late inning + close game
    high_leverage   = if_else(
      .data$inning >= 7 & abs(.data$score_diff) <= 3,
      1L, 0L
    )
  )
  
  # Times through order - use existing Statcast variable
  if ("n_thruorder_pitcher" %in% names(df)) {
    df$times_through_order <- suppressWarnings(as.integer(df$n_thruorder_pitcher))
    df$times_through_order[is.na(df$times_through_order)] <- 1L
  } else {
    # Fallback if variable not available (shouldn't happen with modern Statcast data)
    df$times_through_order <- 1L
    warning("n_thruorder_pitcher not found in data; setting times_through_order to 1")
  }
  
  df <- add_last_pitch(df)

  if (include_batter_metrics) {
    bmet <- compute_batter_metrics(df)
    df <- df %>% left_join(bmet, by = "batter") %>%
      mutate(
        o_swing_pct       = coalesce(.data$o_swing_pct, 0.5),
        z_contact_pct     = coalesce(.data$z_contact_pct, 0.5),
        swing_pct         = coalesce(.data$swing_pct, 0.5),
        chase_contact_pct = coalesce(.data$chase_contact_pct, 0.5)
      )
  } else {
    # Batter metrics not needed for per-pitcher models — skip the expensive
    # group-by/join and fill with neutral defaults so downstream code doesn't break.
    df <- df %>% mutate(
      o_swing_pct = 0.5, z_contact_pct = 0.5,
      swing_pct = 0.5, chase_contact_pct = 0.5
    )
  }

  df
}

# ---------------------- NA-safe prep & constant-drop -------------------------
na_safe_factor <- function(x) { x <- as.character(x); x[is.na(x) | x == ""] <- "UNK"; factor(x) }
na_safe_numeric <- function(x) { if (all(is.na(x))) return(rep(0, length(x))); m <- suppressWarnings(median(x, na.rm = TRUE)); x[is.na(x)] <- ifelse(is.finite(m), m, 0); as.numeric(x) }

clean_one_feature <- function(vec) {
  if (is.factor(vec) || is.character(vec)) { v <- na_safe_factor(vec); v <- droplevels(v); list(v = v, ok = nlevels(v) >= 2, fill = NA_real_) }
  else {
    m <- suppressWarnings(median(vec, na.rm = TRUE))
    fill <- if (is.finite(m)) m else 0
    v <- na_safe_numeric(vec); uniq <- unique(v)
    list(v = v, ok = length(uniq) >= 2, fill = fill)
  }
}

prepare_features <- function(df, feature_names) {
  present <- intersect(feature_names, names(df))
  keep <- c(); out <- df; fills <- list()
  for (nm in present) {
    res <- clean_one_feature(out[[nm]])
    out[[nm]] <- res$v
    fills[[nm]] <- res$fill
    if (res$ok) keep <- c(keep, nm)
  }
  list(data = out, features = unique(keep), fills = fills)
}

#' Put test features on exactly the training representation
#'
#' Factor levels come from training, and so do the values used to fill NAs.
#' Re-deriving either from the test set is a train/test skew: imputing a missing
#' score_diff with the median of the day being scored means the model is handed a
#' value it was never fit against, and on a single-game test window that median can
#' be far from the training one.
align_features_to_train <- function(te, tr, feats, fills = NULL) {
  for (nm in feats) {
    if (!nm %in% names(te)) next
    if (is.factor(tr[[nm]])) {
      # na_safe_factor first so NA/"" map to "UNK" the same way they did in
      # training; the relevel then keeps only levels the model actually saw.
      te[[nm]] <- factor(as.character(na_safe_factor(te[[nm]])), levels = levels(tr[[nm]]))
    } else {
      v <- suppressWarnings(as.numeric(te[[nm]]))
      fill <- if (!is.null(fills) && !is.null(fills[[nm]])) fills[[nm]] else 0
      v[is.na(v)] <- fill
      te[[nm]] <- v
    }
  }
  te
}

#' Warn when the full model cannot represent what the baseline conditions on
#'
#' Deception+ reads a ratio above 1 as "the full model, despite knowing more, is
#' still surprised — this pitcher is unpredictable." That reading is only valid if
#' the full model *can* reproduce the baseline. The baseline is a saturated
#' cross-tab over its keys, so a key it cells on must reach the model as a factor
#' (or as something with at most two values, where linear and saturated coincide).
#'
#' When that fails, the comparison inverts: a pitcher with a strong pattern in a
#' key the baseline cells on but the model can only approximate linearly scores as
#' MORE unpredictable the more rigid their pattern is.
check_baseline_nesting <- function(df, feature_names, baseline_keys) {
  problems <- character(0)
  for (k in baseline_keys) {
    if (!k %in% names(df)) next
    if (!k %in% feature_names) {
      problems <- c(problems, sprintf(
        "'%s' is a baseline key but not a model feature", k))
      next
    }
    v <- df[[k]]
    if (!is.factor(v) && !is.character(v) && length(unique(v[!is.na(v)])) > 2) {
      problems <- c(problems, sprintf(
        "'%s' cells the baseline but reaches the model as a numeric with %d levels",
        k, length(unique(v[!is.na(v)]))))
    }
  }
  if (length(problems) > 0) {
    warning("The baseline is more expressive than the full model, so the ",
            "unpredictability ratio may rise with predictability rather than fall:\n  - ",
            paste(problems, collapse = "\n  - "),
            "\nPrefer the joint `count` factor over numeric balls/strikes.",
            call. = FALSE)
  }
  invisible(problems)
}

prune_baseline_keys <- function(df, baseline_keys) {
  keys <- intersect(baseline_keys, names(df)); good <- c()
  for (k in keys) {
    v <- df[[k]]
    if (is.factor(v) || is.character(v)) { v <- na_safe_factor(v); v <- droplevels(v); if (nlevels(v) >= 2) good <- c(good, k) }
    else { v <- na_safe_numeric(v); if (length(unique(v)) >= 2) good <- c(good, k) }
  }
  unique(good)
}

# ---------------------- Baseline Models --------------------------------------
# Build the "key" column (the conditioning cell) the same way for train and test.
.baseline_key <- function(d, keys) {
  d %>%
    mutate(across(all_of(keys), ~ if (is.factor(.x) || is.character(.x)) na_safe_factor(.x) else na_safe_numeric(.x))) %>%
    mutate(key = do.call(paste, c(across(all_of(keys)), sep = "_"))) %>%
    pull(key)
}

#' Baseline pitch-type probabilities
#'
#' @param baseline_alpha Pseudo-count mass for the conditional cells.  Smoothing
#'   backs off toward the pitcher's marginal mix rather than toward a uniform
#'   distribution: the old `(n + 1) / sum(n + 1)` put one pseudo-count on EVERY
#'   class, so a cell holding 4 real pitches from an 8-pitch vocabulary was 2/3
#'   prior, and that prior insisted the pitcher was equally likely to throw all
#'   eight.  It penalised deep arsenals purely for being deep, and — because
#'   Deception+ divides by baseline surprise — quietly lifted their scores.
compute_baseline_probs <- function(tr_data, te_data, baseline_type = "conditional",
                                   baseline_keys = c("count","is_risp","stand","p_throws"),
                                   baseline_alpha = 5) {
  classes <- levels(tr_data$pitch_class)

  # Marginal mix over the training window (Laplace-smoothed so no class is 0).
  probs_marginal <- class_prior(tr_data$pitch_class, classes, alpha = 1)
  marginal_mat <- function(n) {
    matrix(probs_marginal, nrow = n, ncol = length(classes),
           byrow = TRUE, dimnames = list(NULL, classes))
  }

  if (baseline_type == "marginal") return(marginal_mat(nrow(te_data)))

  if (!baseline_type %in% c("conditional", "hybrid")) {
    stop("Unknown baseline_type: ", baseline_type, ". Use 'marginal', 'conditional', or 'hybrid'.")
  }

  keys_tr <- prune_baseline_keys(tr_data, baseline_keys)
  if (length(keys_tr) == 0) {
    if (baseline_type == "conditional") {
      warning("No valid baseline keys; falling back to marginal baseline.")
    }
    return(marginal_mat(nrow(te_data)))
  }

  key_tr <- .baseline_key(tr_data, keys_tr)
  key_te <- .baseline_key(te_data, keys_tr)

  # Counts per (cell, class), then back off toward the marginal mix:
  #   P(class | cell) = (n_cell_class + alpha * P_marginal(class)) / (n_cell + alpha)
  # Sums to 1 by construction and converges on the raw cell frequencies as the
  # cell fills up.
  raw_counts <- tibble::tibble(key = key_tr, pitch_class = tr_data$pitch_class) %>%
    count(key, pitch_class, name = "n")
  n_by_key <- tibble::tibble(key = key_tr) %>% count(key, name = "n_key")

  counts <- tidyr::expand_grid(
      key = unique(key_tr),
      pitch_class = factor(classes, levels = classes)
    ) %>%
    dplyr::left_join(raw_counts, by = c("key", "pitch_class")) %>%
    dplyr::left_join(n_by_key, by = "key") %>%
    dplyr::mutate(
      n = dplyr::coalesce(n, 0L),
      prior = probs_marginal[match(as.character(pitch_class), classes)],
      prob  = (n + baseline_alpha * prior) / (n_key + baseline_alpha)
    )

  # "hybrid" only trusts a cell once it holds enough pitches; "conditional" uses
  # every cell it saw.  Either way, cells absent from training fall back to the
  # marginal mix — NOT to a uniform 1/K.  Uniform was both arbitrary and
  # punishing: it charged ~log(K) nats of baseline surprise for a context the
  # baseline had simply never encountered, deflating the ratio.
  usable_keys <- if (baseline_type == "hybrid") {
    n_by_key %>% filter(n_key >= 5) %>% pull(key)
  } else {
    unique(key_tr)
  }

  P_base <- marginal_mat(nrow(te_data))

  counts_usable <- counts %>% filter(key %in% usable_keys)
  if (nrow(counts_usable) > 0) {
    counts_wide <- counts_usable %>%
      select(key, pitch_class, prob) %>%
      tidyr::pivot_wider(names_from = pitch_class, values_from = prob, values_fill = NA)
    matched <- dplyr::left_join(
      tibble::tibble(key = key_te, .row = seq_len(length(key_te))),
      counts_wide, by = "key"
    )
    hit <- !is.na(matched[[classes[1]]])
    if (any(hit)) {
      for (cls in classes) {
        if (cls %in% names(matched)) P_base[matched$.row[hit], cls] <- matched[[cls]][hit]
      }
    }
  }

  P_base
}

# ---------------------- Name resolver: StatsAPI + cache (+ baseballr fallback)
resolve_pitcher_names_statsapi <- function(df_with_ids,
                                           cache_file = "cache/mlbam_name_cache.csv",
                                           batch_size = 100,
                                           verbose = TRUE) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Please install 'jsonlite'.")
  if (!requireNamespace("httr", quietly = TRUE))     stop("Please install 'httr'.")
  
  ids <- df_with_ids %>%
    dplyr::filter(!is.na(pitcher_id)) %>%
    dplyr::distinct(pitcher_id) %>%
    dplyr::pull(pitcher_id) %>%
    as.integer()
  if (length(ids) == 0) {
    return(tibble::tibble(pitcher_id = integer(0), pitcher_name = character(0)))
  }
  
  cache <- if (file.exists(cache_file)) {
    suppressWarnings(readr::read_csv(cache_file, show_col_types = FALSE)) %>%
      dplyr::mutate(pitcher_id = as.integer(pitcher_id)) %>% dplyr::distinct()
  } else tibble::tibble(pitcher_id = integer(0), pitcher_name = character(0))
  
  need_ids <- setdiff(ids, cache$pitcher_id)
  
  fetch_batch <- function(id_vec) {
    q <- paste(id_vec, collapse = ",")
    url <- paste0("https://statsapi.mlb.com/api/v1/people?personIds=", q)
    resp <- httr::GET(url, httr::user_agent("ppi-name-resolver/1.0"))
    if (httr::http_error(resp)) {
      if (verbose) message("StatsAPI error: ", httr::status_code(resp), " for batch of ", length(id_vec))
      return(tibble::tibble(pitcher_id = as.integer(id_vec), pitcher_name = NA_character_))
    }
    txt <- httr::content(resp, as = "text", encoding = "UTF-8")
    dat <- jsonlite::fromJSON(txt, simplifyDataFrame = TRUE)
    if (!is.list(dat) || is.null(dat$people) || nrow(dat$people) == 0) {
      return(tibble::tibble(pitcher_id = as.integer(id_vec), pitcher_name = NA_character_))
    }
    people <- tibble::as_tibble(dat$people)
    if (!"id" %in% names(people)) people$id <- NA_integer_
    if (!"fullName" %in% names(people)) people$fullName <- NA_character_
    out <- people %>%
      dplyr::transmute(pitcher_id = as.integer(.data$id),
                       pitcher_name = .data$fullName) %>%
      dplyr::filter(!is.na(pitcher_id)) %>% dplyr::distinct()
    missing_ids <- setdiff(id_vec, out$pitcher_id)
    if (length(missing_ids) > 0) {
      out <- dplyr::bind_rows(out, tibble::tibble(pitcher_id = as.integer(missing_ids), pitcher_name = NA_character_))
    }
    out
  }
  
  new_rows <- tibble::tibble(pitcher_id = integer(0), pitcher_name = character(0))
  if (length(need_ids) > 0) {
    if (verbose) message("Resolving ", length(need_ids), " new IDs via StatsAPI (batch ", batch_size, ")...")
    for (i in seq(1, length(need_ids), by = batch_size)) {
      slice <- need_ids[i:min(i + batch_size - 1, length(need_ids))]
      got <- fetch_batch(slice)
      new_rows <- dplyr::bind_rows(new_rows, got)
    }
    cache <- cache %>% dplyr::bind_rows(new_rows) %>% dplyr::distinct()
    tmpfile <- paste0(cache_file, ".tmp")
    readr::write_csv(cache, tmpfile)
    file.rename(tmpfile, cache_file)
  }
  
  cache %>%
    dplyr::filter(pitcher_id %in% ids) %>%
    dplyr::mutate(pitcher_name = dplyr::if_else(is.na(pitcher_name),
                                                paste0("Pitcher_", pitcher_id),
                                                pitcher_name)) %>%
    dplyr::distinct()
}

resolve_pitcher_names_with_fallback <- function(df_with_ids,
                                                cache_file = "cache/mlbam_name_cache.csv",
                                                verbose = TRUE) {
  map_api <- resolve_pitcher_names_statsapi(df_with_ids, cache_file = cache_file, verbose = verbose)
  
  # If any remain synthetic, try baseballr lookup as a courtesy (optional)
  to_fill <- map_api %>% dplyr::filter(startsWith(pitcher_name, "Pitcher_"))
  if (nrow(to_fill) == 0) return(map_api)
  
  if (requireNamespace("baseballr", quietly = TRUE)) {
    ids <- to_fill$pitcher_id
    has_playername <- "playername_lookup" %in% getNamespaceExports("baseballr")
    lu <- try(if (has_playername) baseballr::playername_lookup(ids)
              else baseballr::chadwick_player_name_lu(ids), silent = TRUE)
    if (!inherits(lu, "try-error") && !is.null(lu) && nrow(lu) > 0) {
      lu <- tibble::as_tibble(lu)
      if (!"key_mlbam" %in% names(lu)) lu$key_mlbam <- NA
      if (!"mlbam_id" %in% names(lu))  lu$mlbam_id  <- NA
      if (!"id" %in% names(lu))        lu$id        <- NA
      for (nm in c("name_first","name_last","name_full","first_name","last_name",
                   "name_last_first","full_name")) if (!nm %in% names(lu)) lu[[nm]] <- NA_character_
      nm_first <- dplyr::coalesce(lu$name_first, lu$first_name)
      nm_last  <- dplyr::coalesce(lu$name_last,  lu$last_name)
      nm_full  <- dplyr::coalesce(lu$name_full, lu$full_name, lu$name_last_first,
                                  trimws(paste(nm_first, nm_last)))
      lu_map <- lu %>%
        dplyr::mutate(pitcher_id = suppressWarnings(as.integer(dplyr::coalesce(.data$key_mlbam, .data$mlbam_id, .data$id))),
                      pitcher_name = nm_full) %>%
        dplyr::filter(!is.na(pitcher_id), !is.na(pitcher_name)) %>%
        dplyr::select(pitcher_id, pitcher_name) %>% dplyr::distinct()
      map_api <- map_api %>% dplyr::rows_update(lu_map, by = "pitcher_id")
    }
  }
  
  map_api
}

# ---------------------- Per-Pitcher Model Evaluation --------------------------
#' Evaluate pitcher unpredictability using per-pitcher models
#'
#' For each pitcher, trains a multinomial model on their historical pitches
#' and evaluates surprise on their test pitches. This measures how unpredictable
#' each pitcher is relative to their OWN patterns, not league-wide patterns.
#'
#' @param df_history Historical data for training (e.g., last 500 pitches per pitcher)
#' @param df_test Test data to evaluate (e.g., today's pitches)
#' @param min_train_pitches Minimum training pitches per pitcher (default: 100)
#' @param min_test_pitches Minimum test pitches per pitcher (default: 10)
#' @param feature_names Features to use in per-pitcher models
#' @param decay Weight decay (L2 penalty) for nnet::multinom. Per-pitcher models
#'   are fit on a few hundred pitches with a wide factor (`last_pitch_type`), so
#'   complete separation is routine. Unpenalised, that drives fitted
#'   probabilities to 0/1 and the resulting -log(p) is a numerical artefact
#'   rather than a measure of surprise.
#' @param prob_shrinkage Mass mixed into the model's predicted probabilities from
#'   the pitcher's own marginal mix before taking -log(). Bounds the surprise on
#'   the same scale as the smoothed baseline.
#' @param verbose Print progress messages
#' @return List with:
#'   - results: Data frame with per-pitcher unpredictability metrics
#'   - excluded: Data frame with pitchers who couldn't be evaluated and why
evaluate_per_pitcher <- function(df_history,
                                  df_test,
                                  min_train_pitches = 100,
                                  min_test_pitches = 10,
                                  feature_names = c("count", "outs", "is_risp",
                                                    "stand", "last_pitch_type"),
                                  baseline_keys = c("count", "stand"),
                                  baseline_type = "conditional",
                                  baseline_alpha = 5,
                                  decay = 0.01,
                                  prob_shrinkage = 0.02,
                                  max_weights = 10000,
                                  verbose = TRUE) {

  # Prepare factor columns for history data
  if (nrow(df_history) > 0) {
    df_history <- df_history %>%
      mutate(
        pitch_class = factor(pitch_class),
        stand = na_safe_factor(stand),
        p_throws = na_safe_factor(p_throws),
        last_pitch_type = na_safe_factor(last_pitch_type)
      )
  }

  # Prepare factor columns for test data
  df_test <- df_test %>%
    mutate(
      pitch_class = factor(pitch_class),
      stand = na_safe_factor(stand),
      p_throws = na_safe_factor(p_throws),
      last_pitch_type = na_safe_factor(last_pitch_type)
    )

  check_baseline_nesting(df_test, feature_names, baseline_keys)

  # Get all pitchers in test data
  all_test_pitchers <- df_test %>%
    group_by(pitcher_id) %>%
    summarise(n_test = n(), .groups = "drop")

  # Get history counts (0 for pitchers not in history)
  history_counts <- df_history %>%
    group_by(pitcher_id) %>%
    summarise(n_history = n(), .groups = "drop")

  # Join to get full picture
  pitcher_status <- all_test_pitchers %>%
    left_join(history_counts, by = "pitcher_id") %>%
    mutate(n_history = coalesce(n_history, 0L))

  # Classify pitchers
  pitcher_status <- pitcher_status %>%
    mutate(
      status = case_when(
        n_history == 0 ~ "debut_no_history",
        n_history < min_train_pitches ~ "insufficient_history",
        n_test < min_test_pitches ~ "insufficient_test_pitches",
        TRUE ~ "evaluated"
      )
    )

  # Track excluded pitchers
  excluded <- pitcher_status %>%
    filter(status != "evaluated") %>%
    select(pitcher_id, n_history, n_test, status)

  if (verbose && nrow(excluded) > 0) {
    n_debut <- sum(excluded$status == "debut_no_history")
    n_insufficient <- sum(excluded$status == "insufficient_history")
    if (n_debut > 0) message("  ", n_debut, " pitcher(s) excluded: debut (no historical data)")
    if (n_insufficient > 0) message("  ", n_insufficient, " pitcher(s) excluded: insufficient history (<", min_train_pitches, " pitches)")
  }

  # Get valid pitchers to evaluate
  valid_pitchers <- pitcher_status %>%
    filter(status == "evaluated") %>%
    pull(pitcher_id)

  if (length(valid_pitchers) == 0) {
    warning("No pitchers with sufficient data in both history and test")
    return(list(
      results = tibble(
        pitcher_id = integer(),
        n_history = integer(),
        n_test = integer(),
        n_classes = integer(),
        mean_surp_model = numeric(),
        mean_surp_base = numeric(),
        unpredictability_ratio = numeric(),
        surp_excess = numeric(),
        normed_surprise = numeric()
      ),
      excluded = excluded
    ))
  }

  n_pitchers <- length(valid_pitchers)
  if (verbose) message("Evaluating ", n_pitchers, " pitchers with per-pitcher models...")

  eps <- 1e-9

  # Each pitcher's model is fully independent — parallelize across available cores.
  # parallel::mclapply forks on Unix (GitHub Actions, Linux, macOS); falls back to
  # lapply on Windows where forking is unavailable.
  n_cores <- if (.Platform$OS.type == "unix") {
    min(parallel::detectCores(logical = FALSE), n_pitchers)
  } else {
    1L
  }

  eval_one_pitcher <- function(pid) {
    ptr_history <- df_history %>% filter(pitcher_id == pid)
    ptr_test    <- df_test    %>% filter(pitcher_id == pid)

    # Scope the outcome vocabulary to THIS pitcher.
    #
    # df_history$pitch_class carries the league-wide level set, and dplyr::filter
    # does not drop unused levels, so every pitcher's model previously nominally
    # ranged over every pitch type in MLB. Those empty levels were then dropped
    # by nnet::multinom itself, leaving the fitted model narrower than `classes`
    # and de-synchronising model output from baseline output.
    #
    # The support is the pitcher's own history plus anything they actually threw
    # in the test window. Keeping the test-only classes matters: a pitcher
    # unveiling a new pitch IS being unpredictable, and dropping those rows would
    # discard exactly the evidence the metric exists to capture. They are scored
    # against the smoothed prior, so they earn high — but finite — surprise.
    classes <- union(
      levels(droplevels(factor(as.character(ptr_history$pitch_class)))),
      levels(droplevels(factor(as.character(ptr_test$pitch_class))))
    )
    classes <- sort(classes[!is.na(classes)])
    if (length(classes) < 2) return(NULL)

    ptr_history$pitch_class <- factor(as.character(ptr_history$pitch_class), levels = classes)
    ptr_test$pitch_class    <- factor(as.character(ptr_test$pitch_class),    levels = classes)
    ptr_history <- ptr_history %>% filter(!is.na(pitch_class))
    ptr_test    <- ptr_test    %>% filter(!is.na(pitch_class))
    if (nrow(ptr_history) == 0 || nrow(ptr_test) == 0) return(NULL)
    if (nlevels(droplevels(ptr_history$pitch_class)) < 2) return(NULL)

    pf_tr <- prepare_features(ptr_history, feature_names)
    tr2   <- pf_tr$data
    feats <- pf_tr$features

    te2 <- ptr_test
    if (length(feats) > 0) {
      te2 <- align_features_to_train(te2, tr2, feats, pf_tr$fills)
      feat_complete <- complete.cases(te2[, intersect(feats, names(te2)), drop = FALSE])
      te2 <- te2[feat_complete, , drop = FALSE]
    }
    if (nrow(te2) == 0) return(NULL)

    form <- if (length(feats) == 0) as.formula("pitch_class ~ 1")
            else as.formula(paste("pitch_class ~", paste(feats, collapse = " + ")))

    mod <- try(suppressWarnings(
      nnet::multinom(form, data = tr2, trace = FALSE, maxit = 200,
                     decay = decay, MaxNWts = max_weights)
    ), silent = TRUE)
    if (inherits(mod, "try-error")) return(NULL)

    # Shrinkage target: the pitcher's own smoothed pitch mix over `classes`.
    prior <- class_prior(tr2$pitch_class, classes, alpha = 1)

    P <- try(safe_predict_probs(mod, te2, classes), silent = TRUE)
    if (inherits(P, "try-error")) return(NULL)
    P <- shrink_to_prior(P, prior, lambda = prob_shrinkage)

    idx_true <- match(as.character(te2$pitch_class), classes)
    if (any(is.na(idx_true))) return(NULL)

    p_true    <- P[cbind(seq_len(nrow(te2)), idx_true)]
    surp_model <- -log(pmax(p_true, eps))

    # Baseline: conditional on count/situation (matches seasonal model methodology).
    # Shrunk with the identical rule so numerator and denominator of the ratio
    # sit on the same probability floor.
    P_base <- compute_baseline_probs(tr2, te2,
                                      baseline_type = baseline_type,
                                      baseline_keys = baseline_keys,
                                      baseline_alpha = baseline_alpha)
    P_base <- shrink_to_prior(P_base, prior, lambda = prob_shrinkage)
    p_base <- P_base[cbind(seq_len(nrow(te2)), idx_true)]
    surp_base <- -log(pmax(as.numeric(p_base), eps))

    tibble(
      pitcher_id             = pid,
      n_history              = nrow(ptr_history),
      n_test                 = nrow(te2),
      n_classes              = length(classes),
      mean_surp_model        = mean(surp_model),
      mean_surp_base         = mean(surp_base),
      unpredictability_ratio = mean(surp_model) / pmax(mean(surp_base), eps),
      # Excess surprise in nats. Same comparison as the ratio, expressed as a
      # difference. Reported because the ratio is unstable exactly where the
      # baseline surprise is small: a one-pitch reliever whose model and
      # baseline differ by 0.01 nats — noise — scores a ratio near 1.2, above a
      # genuinely coin-flip pitcher at 1.0, purely from dividing noise by noise.
      # The difference does not have that failure mode and is the sounder basis
      # for any future respecification of Deception+.
      surp_excess            = mean(surp_model) - mean(surp_base),
      # Surprise+ input: see "The two unpredictability scales" at the top.
      normed_surprise        = normalized_surprise(mean(surp_model), length(classes))
    )
  }

  raw_results <- if (n_cores > 1) {
    parallel::mclapply(valid_pitchers, eval_one_pitcher, mc.cores = n_cores)
  } else {
    lapply(valid_pitchers, eval_one_pitcher)
  }
  # Filter out NULLs (skipped pitchers) and any try-error objects from failed workers
  results <- Filter(function(x) !is.null(x) && !inherits(x, "try-error"), raw_results)

  result_df <- bind_rows(results)

  if (verbose && nrow(result_df) > 0) {
    message("  Successfully evaluated ", nrow(result_df), " pitchers")
    message("  Mean unpredictability ratio: ", round(mean(result_df$unpredictability_ratio), 4))
  }

  list(results = result_df, excluded = excluded)
}

#' Get pitcher's last N pitches from historical data
#'
#' @param df_all All available pitch data
#' @param pitcher_ids Vector of pitcher IDs to retrieve
#' @param n_pitches Number of most recent pitches per pitcher (default: 500)
#' @return Data frame with last N pitches for each pitcher
get_pitcher_history <- function(df_all, pitcher_ids, n_pitches = 500) {
  df_all %>%
    filter(pitcher_id %in% pitcher_ids) %>%
    arrange(pitcher_id, desc(game_date), desc(game_pk), desc(at_bat_number), desc(pitch_number)) %>%
    group_by(pitcher_id) %>%
    slice_head(n = n_pitches) %>%
    ungroup()
}

#' Load baseline parameters for standardization
#'
#' @param baseline_file Path to baseline_params.rds file
#' @return List with mu, sd, and metadata
load_baseline_params <- function(baseline_file = "baseline_params.rds") {
  if (!file.exists(baseline_file)) {
    stop("Baseline file not found: ", baseline_file, "\n",
         "Run compute_baseline.R to generate baseline parameters.")
  }
  readRDS(baseline_file)
}

#' Identify starting pitchers for each game
#'
#' Determines which pitchers were starters (threw first pitch for their team)
#' vs relievers (entered the game later) for each game in the dataset.
#'
#' @param df Data frame with pitch data (must have game_pk, pitcher_id, inning_topbot, at_bat_number)
#' @return Data frame with pitcher_id, game_pk, and role (starter/reliever)
identify_starter_reliever <- function(df) {
  # A pitcher is classified as a starter if they:
  # (a) threw the first pitch of the game (inning 1) for their team, OR
  # (b) threw more than 50 pitches in the game

  # Count pitches per pitcher per game
  pitch_counts <- df %>%
    dplyr::count(game_pk, pitcher_id, name = "n_game_pitches")

  # Condition (a): pitcher who threw first in inning 1 for each team side
  starters_by_order <- df %>%
    dplyr::filter(inning == 1) %>%
    dplyr::group_by(game_pk, inning_topbot) %>%
    dplyr::arrange(at_bat_number) %>%
    dplyr::slice_head(n = 1) %>%
    dplyr::ungroup() %>%
    dplyr::select(game_pk, pitcher_id) %>%
    dplyr::distinct() %>%
    dplyr::mutate(is_starter_by_order = TRUE)

  # Get all pitcher-game combinations
  all_appearances <- df %>%
    dplyr::select(game_pk, pitcher_id) %>%
    dplyr::distinct()

  # Classify: starter if opened inning 1 OR threw > 50 pitches in game
  pitcher_roles <- all_appearances %>%
    dplyr::left_join(starters_by_order, by = c("game_pk", "pitcher_id")) %>%
    dplyr::left_join(pitch_counts, by = c("game_pk", "pitcher_id")) %>%
    dplyr::mutate(
      is_starter_by_order = dplyr::coalesce(is_starter_by_order, FALSE),
      is_starter = is_starter_by_order | (dplyr::coalesce(n_game_pitches, 0L) > 50L),
      role = dplyr::if_else(is_starter, "starter", "reliever")
    ) %>%
    dplyr::select(game_pk, pitcher_id, role)

  pitcher_roles
}

#' Get pitcher role summary for a single day
#'
#' For each pitcher, determines if they were a starter or reliever that day.
#' If a pitcher appeared in multiple games (doubleheader), uses starter if they
#' started any game, otherwise reliever.
#'
#' @param df Data frame with pitch data for a single day
#' @return Data frame with pitcher_id and role
get_daily_pitcher_roles <- function(df) {
  roles <- identify_starter_reliever(df)

  # If pitcher appeared in multiple games, prioritize "starter" role
  daily_roles <- roles %>%
    dplyr::group_by(pitcher_id) %>%
    dplyr::summarise(
      role = dplyr::if_else(any(role == "starter"), "starter", "reliever"),
      .groups = "drop"
    )

  daily_roles
}

# ---------------------- Train / Evaluate / PPI -------------------------------
train_ppi <- function(train_start, train_end,
                      test_start = NULL, test_end = NULL,
                      min_test_pitches = 10,
                      min_total_pitches = 50,
                      feature_names = c("count","is_top","outs","score_diff","base_state","is_risp",
                                        "high_leverage","times_through_order",
                                        "stand","p_throws","last_pitch_type",
                                        "o_swing_pct","z_contact_pct","swing_pct","chase_contact_pct"),
                      baseline_keys = c("count","is_risp","stand","p_throws"),
                      baseline_type = "conditional",
                      baseline_alpha = 5,
                      train_game_type = "R",
                      test_game_type = "R",
                      train_level = "MLB",
                      test_level = "MLB",
                      split_method = "temporal",
                      random_seed = NULL,
                      decay = 1e-4,
                      prob_shrinkage = 0.02,
                      standardize = c("test", "train"),
                      max_weights = 10000,
                      verbose = TRUE) {

  standardize <- match.arg(standardize)

  # Validate split_method
  if (!split_method %in% c("temporal", "random")) {
    stop("split_method must be 'temporal' or 'random'")
  }

  # For temporal split, test dates are required
  if (split_method == "temporal" && (is.null(test_start) || is.null(test_end))) {
    stop("For temporal split, test_start and test_end are required")
  }

  # For random split, use train dates as the period of interest
  if (split_method == "random") {
    test_start <- train_start
    test_end <- train_end
    test_game_type <- train_game_type
    if (verbose) message("Using random split: 50% of each pitcher's pitches for train/test")
  }

  ensure_directories()

  # Set random seed if provided (for reproducibility)
  if (!is.null(random_seed)) {
    set.seed(random_seed)
    if (verbose) message("Random seed set to: ", random_seed)
  }

  # Only compute batter metrics (o_swing_pct, z_contact_pct, etc.) when the
  # caller's feature set actually uses them — avoids an expensive join on large data.
  batter_metric_cols <- c("o_swing_pct", "z_contact_pct", "swing_pct", "chase_contact_pct")
  needs_batter_metrics <- any(batter_metric_cols %in% feature_names)

  # ========== STEP 1: Load data ==========
  if (split_method == "random") {
    # RANDOM SPLIT: Load all data for period, then split by pitcher
    if (verbose) {
      message("\n========================================")
      message("RANDOM SPLIT MODE")
      message("PERIOD: ", train_start, " to ", train_end)
      message("========================================")
    }

    cachefile <- sprintf("cache/savant_raw_%s_%s_%s_%s.Rds", train_start, train_end, train_game_type, train_level)
    if (file.exists(cachefile)) {
      message("✅ Using cached data: ", cachefile)
      raw_all <- readRDS(cachefile)
    } else {
      message("⬇️ Downloading data...")
      raw_all <- load_statcast_range(train_start, train_end, game_type = train_game_type, level = train_level, verbose = verbose)
      if (nrow(raw_all) > 0) {
        saveRDS(raw_all, cachefile)
        message("💾 Cached data to ", cachefile)
      } else stop("No data found for the given range.")
    }

    df_all <- engineer_features(raw_all, include_batter_metrics = needs_batter_metrics)
    if (nrow(df_all) == 0) stop("No usable rows after feature engineering.")
    df_all <- df_all %>% filter(!is.na(pitcher_id))

    if (verbose) message("Total data: ", nrow(df_all), " pitches from ", length(unique(df_all$pitcher_id)), " pitchers")

    # Perform random split: 50% of each pitcher's pitches to train, 50% to test
    if (verbose) message("🎲 Performing random 50/50 split per pitcher...")

    df_all <- df_all %>%
      group_by(pitcher_id) %>%
      mutate(
        pitch_row = row_number(),
        n_total = n(),
        random_order = sample(n()),
        is_train = random_order <= ceiling(n() / 2)
      ) %>%
      ungroup()

    df_train <- df_all %>% filter(is_train) %>% select(-pitch_row, -n_total, -random_order, -is_train)
    df_test <- df_all %>% filter(!is_train) %>% select(-pitch_row, -n_total, -random_order, -is_train)

    if (verbose) {
      message("Training data: ", nrow(df_train), " pitches from ", length(unique(df_train$pitcher_id)), " pitchers")
      message("Test data: ", nrow(df_test), " pitches from ", length(unique(df_test$pitcher_id)), " pitchers")
    }

  } else {
    # TEMPORAL SPLIT: Load separate train and test data
    if (verbose) {
      message("\n========================================")
      message("TEMPORAL SPLIT MODE")
      message("TRAINING PERIOD: ", train_start, " to ", train_end)
      message("========================================")
    }

    train_cachefile <- sprintf("cache/savant_raw_%s_%s_%s_%s.Rds", train_start, train_end, train_game_type, train_level)
    if (file.exists(train_cachefile)) {
      message("✅ Using cached training data: ", train_cachefile)
      raw_train <- readRDS(train_cachefile)
    } else {
      message("⬇️ Downloading training data...")
      raw_train <- load_statcast_range(train_start, train_end, game_type = train_game_type, level = train_level, verbose = verbose)
      if (nrow(raw_train) > 0) {
        saveRDS(raw_train, train_cachefile)
        message("💾 Cached training data to ", train_cachefile)
      } else stop("No training data found for the given range.")
    }

    df_train <- engineer_features(raw_train, include_batter_metrics = needs_batter_metrics)
    if (nrow(df_train) == 0) stop("No usable training rows after feature engineering.")
    df_train <- df_train %>% filter(!is.na(pitcher_id))

    if (verbose) message("Training data: ", nrow(df_train), " pitches from ", length(unique(df_train$pitcher_id)), " pitchers")

    # ========== Load TEST data ==========
    if (verbose) {
      message("\n========================================")
      message("TEST PERIOD: ", test_start, " to ", test_end)
      message("========================================")
    }

    test_cachefile <- sprintf("cache/savant_raw_%s_%s_%s_%s.Rds", test_start, test_end, test_game_type, test_level)
    if (file.exists(test_cachefile)) {
      message("✅ Using cached test data: ", test_cachefile)
      raw_test <- readRDS(test_cachefile)
    } else {
      message("⬇️ Downloading test data...")
      raw_test <- load_statcast_range(test_start, test_end, game_type = test_game_type, level = test_level, verbose = verbose)
      if (nrow(raw_test) > 0) {
        saveRDS(raw_test, test_cachefile)
        message("💾 Cached test data to ", test_cachefile)
      } else stop("No test data found for the given range.")
    }

    df_test <- engineer_features(raw_test, include_batter_metrics = needs_batter_metrics)
    if (nrow(df_test) == 0) stop("No usable test rows after feature engineering.")
    df_test <- df_test %>% filter(!is.na(pitcher_id))

    if (verbose) message("Test data: ", nrow(df_test), " pitches from ", length(unique(df_test$pitcher_id)), " pitchers")
  }
  
  # ========== STEP 3: Prepare training data and fit model ==========
  if (verbose) message("\n🔧 Fitting multinomial model...")
  
  df_train <- df_train %>% mutate(
    pitch_class     = factor(pitch_class),
    stand           = na_safe_factor(stand),
    p_throws        = na_safe_factor(p_throws),
    last_pitch_type = na_safe_factor(last_pitch_type)
  )
  
  if (nlevels(droplevels(df_train$pitch_class)) < 2) {
    stop("Training data has < 2 pitch classes; widen date range.")
  }
  
  check_baseline_nesting(df_train, feature_names, baseline_keys)

  pf_tr <- prepare_features(df_train, feature_names)
  tr2   <- pf_tr$data
  feats <- pf_tr$features

  form <- if (length(feats) == 0) as.formula("pitch_class ~ 1")
  else as.formula(paste("pitch_class ~", paste(feats, collapse = " + ")))
  
  if (verbose) message("Model formula: ", deparse(form))
  
  mod <- try(nnet::multinom(form, data = tr2, trace = FALSE, maxit = 500,
                            decay = decay, MaxNWts = max_weights), silent = TRUE)
  if (inherits(mod, "try-error")) {
    warning("Multinomial fit failed; retrying with intercept-only model.")
    form <- as.formula("pitch_class ~ 1")
    mod  <- nnet::multinom(form, data = tr2, trace = FALSE, maxit = 500,
                           decay = decay, MaxNWts = max_weights)
    feats <- character(0)
  }
  
  classes <- levels(tr2$pitch_class)
  if (verbose) {
    message("✅ Model trained")
    message("   Pitch classes: ", paste(classes, collapse = ", "))
    message("   Features: ", paste(feats, collapse = ", "))
  }
  
  # ========== STEP 4: Prepare test data ==========
  df_test <- df_test %>% mutate(
    pitch_class     = factor(pitch_class, levels = levels(df_train$pitch_class)),
    stand           = na_safe_factor(stand),
    p_throws        = na_safe_factor(p_throws),
    last_pitch_type = na_safe_factor(last_pitch_type)
  )
  
  # Put test features on the training representation: training factor levels and
  # training NA-fill values (never test-set medians).
  te2 <- df_test
  if (length(feats) > 0) te2 <- align_features_to_train(te2, tr2, feats, pf_tr$fills)

  # Drop test rows with unseen pitch classes or unseen factor levels
  te2 <- te2 %>% dplyr::filter(!is.na(pitch_class))
  if (length(feats) > 0) {
    feat_complete <- complete.cases(te2[, feats, drop = FALSE])
    if (any(!feat_complete) && verbose) {
      message("  Dropped ", sum(!feat_complete), " test rows with unseen factor levels")
    }
    te2 <- te2[feat_complete, , drop = FALSE]
  }

  if (nrow(te2) == 0) stop("No valid test rows after aligning factor levels with training data.")

  # ========== STEP 5: Model predictions and surprise ==========
  if (verbose) message("\n🎯 Evaluating test pitches...")

  eps <- 1e-9
  # League-wide pitch mix over the training window: the shrinkage target that
  # keeps model and baseline surprise on a common probability floor.
  prior <- class_prior(tr2$pitch_class, classes, alpha = 1)

  P <- shrink_to_prior(safe_predict_probs(mod, te2, classes), prior, lambda = prob_shrinkage)

  idx_true <- match(as.character(te2$pitch_class), classes)
  p_true   <- P[cbind(seq_len(nrow(te2)), idx_true)]
  surp_model <- -log(pmax(p_true, eps))

  # ========== STEP 6: Baseline predictions and surprise ==========
  if (verbose) message("📐 Computing baseline...")

  P_base <- compute_baseline_probs(tr2, te2, baseline_type = baseline_type,
                                   baseline_keys = baseline_keys, baseline_alpha = baseline_alpha)
  P_base <- shrink_to_prior(P_base, prior, lambda = prob_shrinkage)
  p_true_base <- P_base[cbind(seq_len(nrow(te2)), idx_true)]
  surp_base   <- -log(pmax(p_true_base, eps))

  # ========== STEP 6b: Compute training baseline for standardization ==========
  # In-sample unpredictability ratios on the training data. These give a
  # test-period-independent anchor (`standardize = "train"`), which is useful when
  # comparing several test windows fit from one training window — but they are
  # measured IN SAMPLE, so model surprise here is optimistically low and the
  # resulting μ sits below the out-of-sample μ. Standardising against it shifts
  # every score upward, which is why it is no longer the default.
  if (verbose) message("📊 Computing training baseline for standardization...")

  # In-sample predictions on training data
  P_train <- shrink_to_prior(safe_predict_probs(mod, tr2, classes), prior, lambda = prob_shrinkage)

  idx_true_train <- match(as.character(tr2$pitch_class), classes)
  p_true_train <- P_train[cbind(seq_len(nrow(tr2)), idx_true_train)]
  surp_model_train <- -log(pmax(p_true_train, eps))

  # Baseline for training data (in-sample)
  P_base_train <- compute_baseline_probs(tr2, tr2, baseline_type = baseline_type,
                                         baseline_keys = baseline_keys, baseline_alpha = baseline_alpha)
  P_base_train <- shrink_to_prior(P_base_train, prior, lambda = prob_shrinkage)
  p_true_base_train <- P_base_train[cbind(seq_len(nrow(tr2)), idx_true_train)]
  surp_base_train <- -log(pmax(p_true_base_train, eps))

  # Each pitcher's own arsenal size, from the training window. Surprise+ divides
  # by log of this, NOT by log of the league-wide class count — the seasonal model
  # shares one class vocabulary across every pitcher, but "how much uncertainty
  # could you possibly create" is a property of the pitches you actually throw.
  arsenal_size <- tr2 %>%
    group_by(pitcher_id) %>%
    summarise(n_classes = n_distinct(as.character(pitch_class)), .groups = "drop")

  # Per-pitcher aggregation for training data (for standardization reference)
  per_pitcher_train <- tr2 %>%
    select(pitcher_id) %>%
    mutate(surp_model = surp_model_train, surp_base = surp_base_train) %>%
    group_by(pitcher_id) %>%
    summarise(
      n_pitches_train = n(),
      mean_surp_model = mean(surp_model),
      mean_surp_base  = mean(surp_base),
      .groups = "drop"
    ) %>%
    filter(n_pitches_train >= min_total_pitches) %>%
    left_join(arsenal_size, by = "pitcher_id") %>%
    mutate(unpredictability_ratio = mean_surp_model / pmax(mean_surp_base, 1e-9),
           normed_surprise = normalized_surprise(mean_surp_model, n_classes))

  # Compute standardization parameters from TRAINING population
  # This establishes "league average" based on the training period
  train_u_mu <- mean(per_pitcher_train$unpredictability_ratio, na.rm = TRUE)
  train_u_sd <- sd(per_pitcher_train$unpredictability_ratio, na.rm = TRUE)
  train_s_mu <- mean(per_pitcher_train$normed_surprise, na.rm = TRUE)
  train_s_sd <- sd(per_pitcher_train$normed_surprise, na.rm = TRUE)

  if (verbose) {
    message("   Training baseline: ratio μ=", round(train_u_mu, 4), " σ=", round(train_u_sd, 4),
            " | normed surprise μ=", round(train_s_mu, 4), " σ=", round(train_s_sd, 4))
    message("   Based on ", nrow(per_pitcher_train), " pitchers in training period")
  }
  
  # ========== STEP 7: Per-pitcher aggregation ==========
  if (verbose) message("👥 Aggregating by pitcher...")
  
  per_pitcher_test <- te2 %>%
    select(pitcher_id) %>%
    mutate(surp_model = surp_model, surp_base = surp_base) %>%
    group_by(pitcher_id) %>%
    summarise(
      n_pitches_test  = n(),
      mean_surp_model = mean(surp_model),
      mean_surp_base  = mean(surp_base),
      .groups = "drop"
    ) %>%
    filter(n_pitches_test >= min_test_pitches) %>%
    left_join(arsenal_size, by = "pitcher_id") %>%
    mutate(ppi = 1 - (mean_surp_model / pmax(mean_surp_base, 1e-9)),
           ppi = pmin(pmax(ppi, -1), 1),
           unpredictability_ratio = mean_surp_model / pmax(mean_surp_base, 1e-9),
           # See evaluate_per_pitcher(): difference-scale companion to the ratio,
           # stable where the baseline surprise is near zero.
           surp_excess = mean_surp_model - mean_surp_base,
           normed_surprise = normalized_surprise(mean_surp_model, n_classes))
  
  # Total pitches across both periods (for reference)
  all_pitchers <- bind_rows(df_train, df_test) %>%
    filter(!is.na(pitcher_id)) %>%
    group_by(pitcher_id) %>%
    summarise(total_pitches = n(), .groups = "drop")
  
  # ========== STEP 8: Resolve names ==========
  if (verbose) message("🔍 Resolving pitcher names...")
  
  all_pitcher_ids <- bind_rows(df_train, df_test) %>% 
    filter(!is.na(pitcher_id)) %>%
    distinct(pitcher_id)
  name_map <- resolve_pitcher_names_with_fallback(all_pitcher_ids, cache_file = "cache/mlbam_name_cache.csv", verbose = verbose)
  
  # ========== STEP 9: Calculate Deception+ ==========
  # Join and filter to get the final population
  pitcher_ppi <- all_pitchers %>%
    left_join(per_pitcher_test, by = "pitcher_id") %>%
    left_join(name_map, by = "pitcher_id") %>%
    filter(total_pitches >= min_total_pitches) %>%
    filter(!is.na(ppi)) %>%
    mutate(pitcher_name = if_else(is.na(pitcher_name), paste0("Pitcher_", pitcher_id), pitcher_name))

  # Standardisation anchor.
  #   "test"  (default) — μ and σ of the evaluated population, so the output
  #           really does have mean 100 and SD 10 as README/METHODOLOGY promise.
  #   "train" — the in-sample training population, stable across test windows
  #           fit from one training window, but optimistically biased (see 6b).
  test_u_mu <- mean(pitcher_ppi$unpredictability_ratio, na.rm = TRUE)
  test_u_sd <- sd(pitcher_ppi$unpredictability_ratio, na.rm = TRUE)
  test_s_mu <- mean(pitcher_ppi$normed_surprise, na.rm = TRUE)
  test_s_sd <- sd(pitcher_ppi$normed_surprise, na.rm = TRUE)
  if (!is.finite(test_u_sd) || test_u_sd <= 0 || !is.finite(test_s_sd) || test_s_sd <= 0) {
    warning("Test-population SD is not usable; falling back to the training anchor.")
    standardize <- "train"
  }

  anchor_mu <- if (standardize == "test") test_u_mu else train_u_mu
  anchor_sd <- if (standardize == "test") test_u_sd else train_u_sd
  s_mu      <- if (standardize == "test") test_s_mu else train_s_mu
  s_sd      <- if (standardize == "test") test_s_sd else train_s_sd

  if (verbose) {
    message("   Standardising on the ", standardize, " population:")
    message("     Deception+ from ratio           μ=", round(anchor_mu, 4), " σ=", round(anchor_sd, 4))
    message("     Surprise+  from normed surprise μ=", round(s_mu, 4), " σ=", round(s_sd, 4))
  }

  pitcher_ppi <- pitcher_ppi %>%
    mutate(
      deception_plus = scale_plus(unpredictability_ratio, anchor_mu, anchor_sd),
      surprise_plus  = scale_plus(normed_surprise, s_mu, s_sd)
    ) %>%
    select(
      pitcher_id, pitcher_name, total_pitches, n_pitches_test, n_classes,
      mean_surp_model, mean_surp_base, ppi,
      unpredictability_ratio, surp_excess, deception_plus,
      normed_surprise, surprise_plus
    ) %>%
    arrange(desc(deception_plus))
  
  if (verbose) {
    message("\n✅ Analysis complete!")
    message("   Pitchers evaluated: ", nrow(pitcher_ppi))
    message("   Mean Deception+: ", round(mean(pitcher_ppi$deception_plus, na.rm = TRUE), 1))
    message("   Range: ", round(min(pitcher_ppi$deception_plus, na.rm = TRUE), 1), 
            " to ", round(max(pitcher_ppi$deception_plus, na.rm = TRUE), 1))
  }
  
  list(model = mod,
       train = df_train, test = df_test,
       classes = classes,
       pitcher_ppi = pitcher_ppi,
       features_used = feats,
       baseline_keys = baseline_keys,
       baseline_type = baseline_type,
       split_method = split_method,
       train_period = paste(train_start, "to", train_end),
       test_period = if (split_method == "random") "random 50/50 split" else paste(test_start, "to", test_end),
       standardization = list(
         anchor = standardize,
         mu = anchor_mu,
         sd = anchor_sd,
         surprise_mu = s_mu,
         surprise_sd = s_sd,
         train_mean = train_u_mu,
         train_sd = train_u_sd,
         test_mean = test_u_mu,
         test_sd = test_u_sd,
         train_surprise_mean = train_s_mu,
         train_surprise_sd = train_s_sd,
         test_surprise_mean = test_s_mu,
         test_surprise_sd = test_s_sd,
         n_train_pitchers = nrow(per_pitcher_train)
       ))
}

# ---------------------- Public API: train & save -----------------------------
train_and_save <- function(train_start, train_end,
                           test_start = NULL, test_end = NULL,
                           min_test_pitches = 10,
                           min_total_pitches = 50,
                           feature_names = c("count","is_top","outs","score_diff","base_state","is_risp",
                                             "high_leverage","times_through_order",
                                             "stand","p_throws","last_pitch_type",
                                             "o_swing_pct","z_contact_pct","swing_pct","chase_contact_pct"),
                           baseline_keys = c("count","is_risp","stand","p_throws"),
                           baseline_type = "conditional",
                           baseline_alpha = 5,
                           train_game_type = "R",
                           test_game_type = "R",
                           train_level = "MLB",
                           test_level = "MLB",
                           split_method = "temporal",
                           random_seed = NULL,
                           decay = 1e-4,
                           prob_shrinkage = 0.02,
                           standardize = c("test", "train"),
                           max_weights = 10000,
                           out_model = "models/ppi_model.rds",
                           out_ppi   = "output/pitcher_ppi.csv",
                           verbose   = TRUE) {

  standardize <- match.arg(standardize)

  res <- train_ppi(train_start, train_end,
                   test_start, test_end,
                   min_test_pitches = min_test_pitches,
                   min_total_pitches = min_total_pitches,
                   feature_names = feature_names,
                   baseline_keys = baseline_keys,
                   baseline_type = baseline_type,
                   baseline_alpha = baseline_alpha,
                   train_game_type = train_game_type,
                   test_game_type = test_game_type,
                   train_level = train_level,
                   test_level = test_level,
                   split_method = split_method,
                   random_seed = random_seed,
                   decay = decay,
                   prob_shrinkage = prob_shrinkage,
                   standardize = standardize,
                   max_weights = max_weights,
                   verbose = verbose)

  saveRDS(list(
    classes = res$classes,
    features_used = res$features_used,
    baseline_keys = res$baseline_keys,
    baseline_type = res$baseline_type,
    standardization = res$standardization,
    decay = decay,
    prob_shrinkage = prob_shrinkage,
    split_method = split_method,
    train_start = train_start,
    train_end = train_end,
    test_start = test_start,
    test_end = test_end,
    train_game_type = train_game_type,
    test_game_type = test_game_type,
    min_test_pitches = min_test_pitches,
    min_total_pitches = min_total_pitches
  ), out_model)
  
  readr::write_csv(res$pitcher_ppi, out_ppi)
  message("✅ Saved model -> ", out_model, " | PPI CSV -> ", out_ppi)
  res
}

# ---------------------- Convenience wrapper for playoff analysis -------------
analyze_playoff_game <- function(game_date,
                                 regular_season_start = "2025-03-20",
                                 regular_season_end = "2025-09-28",
                                 min_test_pitches = 10,
                                 test_game_type = "P",
                                 baseline_type = "conditional",
                                 verbose = TRUE) {
  
  train_and_save(
    train_start = regular_season_start,
    train_end = regular_season_end,
    test_start = game_date,
    test_end = game_date,
    min_test_pitches = min_test_pitches,
    train_game_type = "R",
    test_game_type = test_game_type,
    baseline_type = baseline_type,
    out_model = sprintf("models/playoff_%s_model.rds", game_date),
    out_ppi = sprintf("output/playoff_%s_ppi.csv", game_date),
    verbose = verbose
  )
}

# ---------------------- Visualization Helpers --------------------------------
create_visualizations <- function(res, output_dir = "output/visualizations") {
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    warning("ggplot2 not available; skipping visualizations")
    return(invisible(NULL))
  }
  
  library(ggplot2)
  
  # 1. Deception+ distribution
  p1 <- ggplot(res$pitcher_ppi, aes(x = deception_plus)) +
    geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
    geom_vline(xintercept = 100, linetype = "dashed", color = "red") +
    labs(title = "Distribution of Deception+ Scores",
         subtitle = paste0("Mean = 100, SD = 10 | n = ", nrow(res$pitcher_ppi)),
         x = "Deception+", y = "Count") +
    theme_minimal()
  ggsave(file.path(output_dir, "deception_plus_distribution.png"), p1, width = 10, height = 6)
  
  # 2. PPI vs Total Pitches
  p2 <- ggplot(res$pitcher_ppi, aes(x = total_pitches, y = ppi)) +
    geom_point(alpha = 0.5, color = "darkgreen") +
    geom_smooth(method = "loess", se = TRUE, color = "red") +
    labs(title = "PPI vs Total Pitches",
         x = "Total Pitches", y = "PPI") +
    theme_minimal()
  ggsave(file.path(output_dir, "ppi_vs_pitches.png"), p2, width = 10, height = 6)
  
  # 3. Top 20 Most Predictable
  top20_pred <- res$pitcher_ppi %>% 
    arrange(deception_plus) %>% 
    head(20) %>%
    mutate(pitcher_name = reorder(pitcher_name, -deception_plus))
  
  p3 <- ggplot(top20_pred, aes(x = deception_plus, y = pitcher_name)) +
    geom_col(fill = "coral") +
    geom_vline(xintercept = 100, linetype = "dashed", color = "darkgray") +
    labs(title = "Top 20 Most Predictable Pitchers",
         subtitle = "Lower Deception+ = More Predictable",
         x = "Deception+", y = NULL) +
    theme_minimal()
  ggsave(file.path(output_dir, "top20_predictable.png"), p3, width = 10, height = 8)
  
  # 4. Top 20 Least Predictable
  top20_unpred <- res$pitcher_ppi %>% 
    arrange(desc(deception_plus)) %>% 
    head(20) %>%
    mutate(pitcher_name = reorder(pitcher_name, deception_plus))
  
  p4 <- ggplot(top20_unpred, aes(x = deception_plus, y = pitcher_name)) +
    geom_col(fill = "steelblue") +
    geom_vline(xintercept = 100, linetype = "dashed", color = "darkgray") +
    labs(title = "Top 20 Least Predictable Pitchers",
         subtitle = "Higher Deception+ = Less Predictable",
         x = "Deception+", y = NULL) +
    theme_minimal()
  ggsave(file.path(output_dir, "top20_unpredictable.png"), p4, width = 10, height = 8)
  
  message("✅ Created 4 visualizations in ", output_dir)
  invisible(list(p1 = p1, p2 = p2, p3 = p3, p4 = p4))
}

# ---------------------- Social Media Visualizations ---------------------------
#' @param score_col Which scale to plot: "surprise_plus" (default — reliable on a
#'   single day) or "deception_plus" (a season-scale statistic; ranking one day by
#'   it is mostly ranking noise). See "The two unpredictability scales".
#' @param score_name Display name for that scale, used in the subtitle.
create_social_media_graphics <- function(res,
                                          game_date,
                                          min_pitches_starter = 65,
                                          min_pitches_reliever = 15,
                                          output_dir = "output/visualizations",
                                          top_n = 5,
                                          score_col = "surprise_plus",
                                          score_name = "Surprise+") {
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    warning("ggplot2 not available; skipping social media visualizations")
    return(invisible(NULL))
  }

  library(ggplot2)

  # Load IBM Plex Sans from Google Fonts.
  # Per Datawrapper typography guidelines: sans-serif with tabular lining figures
  # keeps numbers the same height and equally spaced — essential for chart labels.
  chart_font <- ""  # empty string = ggplot2 system default
  if (requireNamespace("sysfonts",  quietly = TRUE) &&
      requireNamespace("showtext", quietly = TRUE)) {
    tryCatch({
      sysfonts::font_add_google("IBM Plex Sans", "ibm_plex_sans")
      showtext::showtext_auto()
      showtext::showtext_opts(dpi = 100)  # must match ggsave dpi
      chart_font <- "ibm_plex_sans"
    }, error = function(e) {
      message("Could not load IBM Plex Sans; falling back to system font. ", conditionMessage(e))
    })
  }

  # Apply status filter first (if status column exists)
  all_pitchers <- res$pitcher_ppi
  if ("status" %in% names(all_pitchers)) {
    all_pitchers <- all_pitchers %>% filter(status == "evaluated")
  }

  # Everything below plots a column named `deception_plus`; point that at
  # whichever scale was requested so the layout code stays in one place.
  if (!score_col %in% names(all_pitchers)) {
    warning("Column '", score_col, "' not found; falling back to deception_plus.")
    score_col <- "deception_plus"; score_name <- "Deception+"
  }
  all_pitchers$deception_plus <- all_pitchers[[score_col]]
  all_pitchers <- all_pitchers %>% filter(!is.na(deception_plus))
  if (nrow(all_pitchers) == 0) {
    warning("No pitchers with a usable ", score_name, " score")
    return(invisible(NULL))
  }

  # Apply role-specific pitch thresholds
  has_roles <- "role" %in% names(all_pitchers)
  if (has_roles) {
    qualified <- all_pitchers %>%
      filter(
        (role == "starter"  & n_pitches_test >= min_pitches_starter) |
        (role == "reliever" & n_pitches_test >= min_pitches_reliever)
      )
  } else {
    qualified <- all_pitchers %>%
      filter(n_pitches_test >= min_pitches_starter)
  }

  if (nrow(qualified) == 0) {
    warning("No pitchers met the minimum pitch thresholds")
    return(invisible(NULL))
  }

  # Format date for display
  date_display <- format(as.Date(game_date), "%B %d, %Y")
  date_short <- format(as.Date(game_date), "%Y-%m-%d")

  # Custom theme for social media — targets 1200x675 px (16:9, Twitter/Bluesky safe)
  # Typography follows Datawrapper guidelines:
  #   • IBM Plex Sans — clean sans-serif with tabular lining figures
  #   • Near-black (#1a1a2e) for primary text; mid-gray for secondary
  #   • Bold only for the title and pitcher names (two clear hierarchy levels)
  #   • Left-aligned text throughout; sentence case (no uppercase labels)
  theme_social <- function() {
    theme_minimal(base_size = 14, base_family = chart_font) +
      theme(
        # Title: bold, large — the only heavy weight in the layout
        plot.title      = element_text(face = "bold", size = 22, hjust = 0,
                                       color = "#1a1a2e", margin = margin(b = 4),
                                       family = chart_font),
        # Subtitle: regular weight, clearly smaller — second hierarchy level
        plot.subtitle   = element_text(size = 12, hjust = 0, color = "#777777",
                                       face = "plain", margin = margin(b = 14),
                                       family = chart_font),
        plot.caption    = element_text(size = 9, color = "#aaaaaa", hjust = 1,
                                       face = "plain", margin = margin(t = 10),
                                       family = chart_font),
        # Subtle vertical grid only; no horizontal clutter
        panel.grid.major.x = element_line(color = "#e8e8e8", linewidth = 0.5),
        panel.grid.major.y = element_blank(),
        panel.grid.minor   = element_blank(),
        # Pitcher names: bold for identity (the one other place bold is warranted)
        axis.text.y     = element_text(size = 13, face = "bold", color = "#1a1a2e",
                                       hjust = 1, family = chart_font),
        # Axis tick labels: regular weight, secondary gray
        axis.text.x     = element_text(size = 10, color = "#999999", face = "plain",
                                       family = chart_font),
        axis.title.x    = element_text(size = 10, color = "#888888", face = "plain",
                                       margin = margin(t = 8), family = chart_font),
        axis.title.y    = element_blank(),
        # Clean white canvas; light panel
        plot.background  = element_rect(fill = "#ffffff", color = NA),
        panel.background = element_rect(fill = "#f7f7f7", color = NA),
        # Extra right margin for score labels, generous top/bottom padding
        plot.margin = margin(24, 40, 16, 24)
      )
  }

  # Helper function to create a single graphic
  # Output: 1200x675 px (width=12", height=6.75", dpi=100) — fits Twitter & Bluesky
  # Twitter recommended: 1200x675 (16:9), max 5 MB
  # Bluesky recommended: 1200x675 (16:9), max 1 MB
  create_graphic <- function(data, title, subtitle, caption, fill_low, fill_high,
                              filename, reorder_desc = TRUE) {
    if (nrow(data) == 0) return(NULL)

    # Rank-order the rows, then build labelled factor for y-axis
    data <- data %>%
      arrange(if (reorder_desc) desc(deception_plus) else deception_plus) %>%
      mutate(
        rank       = row_number(),
        name_label = paste0("#", rank, "  ", pitcher_name),
        score_label = sprintf("%.0f", deception_plus)
      )

    # Factor levels: lowest bar at bottom, highest at top (ggplot reads bottom-up)
    data$name_label <- factor(
      data$name_label,
      levels = if (reorder_desc) data$name_label[order(data$deception_plus)]
               else               data$name_label[order(-data$deception_plus)]
    )

    # Axis bounds in original Deception+ units
    x_left  <- floor(min(data$deception_plus, 90)) - 4   # a few units left of the minimum
    x_track <- ceiling(max(data$deception_plus, 110)) + 20  # room for score labels

    # Shift origin to x_left so bars visually fill the chart from the left edge.
    # geom_col draws from 0, so we express every x as (value - x_left).
    data <- data %>% mutate(bar_shifted = deception_plus - x_left)
    track_shifted <- x_track - x_left
    avg_shifted   <- 100 - x_left

    # Nice axis labels in original Deception+ units
    raw_breaks  <- pretty(c(x_left, x_track), n = 5)
    raw_breaks  <- raw_breaks[raw_breaks >= x_left & raw_breaks <= x_track]
    plot_breaks <- raw_breaks - x_left

    p <- ggplot(data, aes(y = name_label)) +
      # Track bars (light gray background behind every bar)
      geom_col(aes(x = track_shifted),
               fill = "#e2e2e2", width = 0.62, show.legend = FALSE) +
      # Data bars with gradient fill — now properly anchored at the left edge
      geom_col(aes(x = bar_shifted, fill = deception_plus),
               width = 0.62, show.legend = FALSE) +
      # League-average reference line
      geom_vline(xintercept = avg_shifted, linetype = "dashed",
                 color = "#aaaaaa", linewidth = 0.9) +
      # "AVG" annotation on the reference line
      annotate("text", x = avg_shifted, y = 0.42, label = "AVG",
               size = 2.8, color = "#aaaaaa", hjust = 0.5) +
      # Score labels just outside the bar end — bold for emphasis, IBM Plex tabular nums
      geom_text(aes(x = bar_shifted, label = score_label),
                hjust = -0.32, size = 4.5, fontface = "bold", color = "#333333",
                family = chart_font) +
      scale_fill_gradient(low = fill_low, high = fill_high) +
      scale_x_continuous(
        expand = expansion(mult = c(0, 0.01)),
        limits = c(0, track_shifted),
        breaks = plot_breaks,
        labels = as.integer(raw_breaks)
      ) +
      labs(
        title    = title,
        subtitle = subtitle,
        x        = "Deception+  (100 = league average)",
        y        = NULL,
        caption  = caption
      ) +
      theme_social()

    # 1200x675 px — perfect 16:9 for Twitter/Bluesky
    ggsave(file.path(output_dir, filename), p, width = 12, height = 6.75, dpi = 100)
    p
  }

  plots <- list()

  if (has_roles) {
    starters <- qualified %>% filter(role == "starter")
    relievers <- qualified %>% filter(role == "reliever")

    # Starters - Least Predictable
    if (nrow(starters) > 0) {
      top_starters <- starters %>% arrange(desc(deception_plus)) %>% head(top_n)
      plots$starters_unpredictable <- create_graphic(
        top_starters,
        title = "Least Predictable Starters",
        subtitle = paste0(score_name, "  \u00b7  ", date_display, "  \u00b7  min ", min_pitches_starter, " pitches"),
        caption = paste0("Higher = less predictable  \u00b7  100 = league average  \u00b7  @DeceptionPlus  \u00b7  data: Baseball Savant"),
        fill_low = "#4361ee", fill_high = "#7209b7",
        filename = sprintf("social_starters_top%d_unpredictable_%s.png", top_n, date_short),
        reorder_desc = TRUE
      )

      # Starters - Most Predictable
      bottom_starters <- starters %>% arrange(deception_plus) %>% head(top_n)
      plots$starters_predictable <- create_graphic(
        bottom_starters,
        title = "Most Predictable Starters",
        subtitle = paste0(score_name, "  \u00b7  ", date_display, "  \u00b7  min ", min_pitches_starter, " pitches"),
        caption = paste0("Lower = more predictable  \u00b7  100 = league average  \u00b7  @DeceptionPlus  \u00b7  data: Baseball Savant"),
        fill_low = "#e63946", fill_high = "#f4a261",
        filename = sprintf("social_starters_top%d_predictable_%s.png", top_n, date_short),
        reorder_desc = FALSE
      )
    }

    # Relievers - Least Predictable
    if (nrow(relievers) > 0) {
      top_relievers <- relievers %>% arrange(desc(deception_plus)) %>% head(top_n)
      plots$relievers_unpredictable <- create_graphic(
        top_relievers,
        title = "Least Predictable Relievers",
        subtitle = paste0(score_name, "  \u00b7  ", date_display, "  \u00b7  min ", min_pitches_reliever, " pitches"),
        caption = paste0("Higher = less predictable  \u00b7  100 = league average  \u00b7  @DeceptionPlus  \u00b7  data: Baseball Savant"),
        fill_low = "#2a9d8f", fill_high = "#264653",
        filename = sprintf("social_relievers_top%d_unpredictable_%s.png", top_n, date_short),
        reorder_desc = TRUE
      )

      # Relievers - Most Predictable
      bottom_relievers <- relievers %>% arrange(deception_plus) %>% head(top_n)
      plots$relievers_predictable <- create_graphic(
        bottom_relievers,
        title = "Most Predictable Relievers",
        subtitle = paste0(score_name, "  \u00b7  ", date_display, "  \u00b7  min ", min_pitches_reliever, " pitches"),
        caption = paste0("Lower = more predictable  \u00b7  100 = league average  \u00b7  @DeceptionPlus  \u00b7  data: Baseball Savant"),
        fill_low = "#e76f51", fill_high = "#f4a261",
        filename = sprintf("social_relievers_top%d_predictable_%s.png", top_n, date_short),
        reorder_desc = FALSE
      )
    }

    n_graphics <- sum(!sapply(plots, is.null))
    message("Created ", n_graphics, " social media graphics (starters/relievers) in ", output_dir)
  } else {
    # Fallback: No role column, create overall graphics
    top_unpred <- qualified %>% arrange(desc(deception_plus)) %>% head(top_n)
    plots$top_unpredictable <- create_graphic(
      top_unpred,
      title = "Least Predictable Pitchers",
      subtitle = paste0(score_name, "  \u00b7  ", date_display, "  \u00b7  min ", min_pitches_starter, " pitches"),
      caption = paste0("Higher = less predictable  \u00b7  100 = league average  \u00b7  @DeceptionPlus  \u00b7  data: Baseball Savant"),
      fill_low = "#4361ee", fill_high = "#7209b7",
      filename = sprintf("social_top%d_unpredictable_%s.png", top_n, date_short),
      reorder_desc = TRUE
    )

    top_pred <- qualified %>% arrange(deception_plus) %>% head(top_n)
    plots$top_predictable <- create_graphic(
      top_pred,
      title = "Most Predictable Pitchers",
      subtitle = paste0(score_name, "  \u00b7  ", date_display, "  \u00b7  min ", min_pitches_starter, " pitches"),
      caption = paste0("Lower = more predictable  \u00b7  100 = league average  \u00b7  @DeceptionPlus  \u00b7  data: Baseball Savant"),
      fill_low = "#e63946", fill_high = "#f4a261",
      filename = sprintf("social_top%d_predictable_%s.png", top_n, date_short),
      reorder_desc = FALSE
    )

    message("Created 2 social media graphics in ", output_dir)
  }

  invisible(plots)
}

#' Create Baltimore Orioles Deception+ graphic
#'
#' Generates a 1200xN px PNG bar chart for all Orioles pitchers who appeared
#' on the given date, regardless of pitch count. Bars use an orange-to-black
#' gradient (Orioles colors). Pitcher name labels include their pitch count.
#'
#' @param orioles_data Data frame of Orioles pitchers (subset of pitcher_ppi)
#' @param game_date Character date string "YYYY-MM-DD"
#' @param output_dir Directory for PNG output
#' @param score_col Which scale to plot; see create_social_media_graphics()
#' @param score_name Display name for that scale
#' @return Invisible ggplot object
create_orioles_graphic <- function(orioles_data, game_date,
                                   output_dir = "output/visualizations",
                                   score_col = "surprise_plus",
                                   score_name = "Surprise+") {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    warning("ggplot2 not available; skipping Orioles graphic")
    return(invisible(NULL))
  }

  library(ggplot2)

  # As in create_social_media_graphics(): alias the requested scale onto the
  # column the layout code below expects.
  if (!score_col %in% names(orioles_data)) {
    warning("Column '", score_col, "' not found; falling back to deception_plus.")
    score_col <- "deception_plus"; score_name <- "Deception+"
  }
  orioles_data$deception_plus <- orioles_data[[score_col]]

  # Font setup — mirrors create_social_media_graphics()
  chart_font <- ""
  if (requireNamespace("sysfonts", quietly = TRUE) &&
      requireNamespace("showtext", quietly = TRUE)) {
    tryCatch({
      sysfonts::font_add_google("IBM Plex Sans", "ibm_plex_sans")
      showtext::showtext_auto()
      showtext::showtext_opts(dpi = 100)
      chart_font <- "ibm_plex_sans"
    }, error = function(e) {})
  }

  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  date_display <- format(as.Date(game_date), "%B %d, %Y")
  date_short   <- format(as.Date(game_date), "%Y-%m-%d")

  # Sort by deception+ descending (NAs last); embed pitch count in label
  data <- orioles_data %>%
    dplyr::arrange(dplyr::desc(dplyr::coalesce(deception_plus, -Inf))) %>%
    dplyr::mutate(
      name_label  = paste0(pitcher_name, " (", n_pitches_test, " pitches)"),
      score_label = dplyr::if_else(
        !is.na(deception_plus),
        sprintf("%.0f", deception_plus),
        "N/A"
      )
    )

  # Factor levels: lowest score at bottom so ggplot puts highest at top
  data$name_label <- factor(data$name_label, levels = rev(data$name_label))

  # Use only evaluated pitchers to set axis bounds
  evaluated <- data %>% dplyr::filter(!is.na(deception_plus))

  if (nrow(evaluated) == 0) {
    message("No evaluated Orioles pitchers to plot")
    return(invisible(NULL))
  }

  x_left  <- floor(min(evaluated$deception_plus, 90)) - 4
  x_track <- ceiling(max(evaluated$deception_plus, 110)) + 20

  # Shift bars so the left edge of the chart is x_left in Deception+ units
  data <- data %>%
    dplyr::mutate(
      bar_shifted = dplyr::if_else(
        !is.na(deception_plus),
        deception_plus - x_left,
        0
      )
    )
  track_shifted <- x_track - x_left
  avg_shifted   <- 100 - x_left

  raw_breaks  <- pretty(c(x_left, x_track), n = 5)
  raw_breaks  <- raw_breaks[raw_breaks >= x_left & raw_breaks <= x_track]
  plot_breaks <- raw_breaks - x_left

  # Orioles brand colors: black (#000000) → orange (#DF4601)
  # Higher deception+ → more orange; lower → closer to black
  p <- ggplot(data, aes(y = name_label)) +
    # Track (background) bars
    geom_col(aes(x = track_shifted),
             fill = "#e2e2e2", width = 0.62, show.legend = FALSE) +
    # Data bars with Orioles gradient
    geom_col(aes(x = bar_shifted, fill = bar_shifted),
             width = 0.62, show.legend = FALSE) +
    # League-average reference line
    geom_vline(xintercept = avg_shifted, linetype = "dashed",
               color = "#aaaaaa", linewidth = 0.9) +
    annotate("text", x = avg_shifted, y = 0.42, label = "AVG",
             size = 2.8, color = "#aaaaaa", hjust = 0.5) +
    # Score labels
    geom_text(aes(x = bar_shifted, label = score_label),
              hjust = -0.32, size = 4.5, fontface = "bold", color = "#333333",
              family = chart_font) +
    scale_fill_gradient(low = "#000000", high = "#DF4601") +
    scale_x_continuous(
      expand = expansion(mult = c(0, 0.01)),
      limits = c(0, track_shifted),
      breaks = plot_breaks,
      labels = as.integer(raw_breaks)
    ) +
    labs(
      title    = paste0("Baltimore Orioles \u2014 ", score_name),
      subtitle = paste0(date_display, "  \u00b7  all pitchers"),
      x        = paste0(score_name, "  (100 = league average)"),
      y        = NULL,
      caption  = paste0(
        "Higher = less predictable  \u00b7  100 = league average",
        "  \u00b7  @DeceptionPlus  \u00b7  data: Baseball Savant"
      )
    ) +
    theme_minimal(base_size = 14, base_family = chart_font) +
    theme(
      plot.title      = element_text(face = "bold", size = 22, hjust = 0,
                                     color = "#1a1a2e", margin = margin(b = 4),
                                     family = chart_font),
      plot.subtitle   = element_text(size = 12, hjust = 0, color = "#777777",
                                     face = "plain", margin = margin(b = 14),
                                     family = chart_font),
      plot.caption    = element_text(size = 9, color = "#aaaaaa", hjust = 1,
                                     face = "plain", margin = margin(t = 10),
                                     family = chart_font),
      panel.grid.major.x = element_line(color = "#e8e8e8", linewidth = 0.5),
      panel.grid.major.y = element_blank(),
      panel.grid.minor   = element_blank(),
      axis.text.y     = element_text(size = 11, face = "bold", color = "#1a1a2e",
                                     hjust = 1, family = chart_font),
      axis.text.x     = element_text(size = 10, color = "#999999", face = "plain",
                                     family = chart_font),
      axis.title.x    = element_text(size = 10, color = "#888888", face = "plain",
                                     margin = margin(t = 8), family = chart_font),
      axis.title.y    = element_blank(),
      plot.background  = element_rect(fill = "#ffffff", color = NA),
      panel.background = element_rect(fill = "#f7f7f7", color = NA),
      plot.margin = margin(24, 40, 16, 24)
    )

  filename <- sprintf("orioles_%s.png", date_short)
  # Scale height with number of pitchers; 0.65" per row, min 6, max 14
  n_rows <- nrow(data)
  height <- min(max(6.0, 2.5 + n_rows * 0.65), 14)
  ggsave(file.path(output_dir, filename), p, width = 12, height = height, dpi = 100)
  message("Orioles graphic saved: ", file.path(output_dir, filename))
  invisible(p)
}

# ---------------------- CLI --------------------------------------------------
parse_csv_list <- function(x) { if (is.null(x) || is.na(x) || x == "") return(character(0)); trimws(unlist(strsplit(x, ","))) }

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0 && any(args == "--train_start")) {
  get_arg <- function(flag, default = NULL) { hit <- which(args == flag); if (length(hit) == 1 && hit < length(args)) args[hit + 1] else default }
  
  train_start <- get_arg("--train_start")
  train_end <- get_arg("--train_end")
  test_start <- get_arg("--test_start")
  test_end <- get_arg("--test_end")
  
  min_test <- suppressWarnings(as.integer(get_arg("--min_test_pitches", "10")))
  min_total <- suppressWarnings(as.integer(get_arg("--min_total_pitches", "50")))
  out_model <- get_arg("--out_model", "models/ppi_model.rds")
  out_ppi   <- get_arg("--out_ppi", "output/pitcher_ppi.csv")
  feat_str  <- get_arg("--features", "count,is_top,outs,score_diff,base_state,is_risp,high_leverage,times_through_order,stand,p_throws,last_pitch_type,o_swing_pct,z_contact_pct,swing_pct,chase_contact_pct")
  base_str  <- get_arg("--baseline_keys", "count,is_risp,stand,p_throws")
  baseline_type <- get_arg("--baseline_type", "conditional")
  train_game_type <- get_arg("--train_game_type", "R")
  test_game_type <- get_arg("--test_game_type", "R")
  features  <- parse_csv_list(feat_str); base_keys <- parse_csv_list(base_str)
  
  train_level <- get_arg("--train_level", "MLB")
  test_level <- get_arg("--test_level", "MLB")
  split_method <- get_arg("--split_method", "temporal")
  random_seed_str <- get_arg("--random_seed", NA)
  random_seed <- if (is.na(random_seed_str)) NULL else suppressWarnings(as.integer(random_seed_str))

  decay          <- suppressWarnings(as.numeric(get_arg("--decay", "1e-4")))
  prob_shrinkage <- suppressWarnings(as.numeric(get_arg("--prob_shrinkage", "0.02")))
  standardize    <- get_arg("--standardize", "test")

  # For temporal split, test dates are required; for random split, they're optional
  if (split_method == "temporal" && (is.null(test_start) || is.null(test_end))) {
    stop("Usage: Rscript pitch_ppi.R --train_start YYYY-MM-DD --train_end YYYY-MM-DD --test_start YYYY-MM-DD --test_end YYYY-MM-DD [options]\n",
         "       Rscript pitch_ppi.R --train_start YYYY-MM-DD --train_end YYYY-MM-DD --split_method random [options]\n",
         "Options:\n",
         "  --split_method METHOD    temporal/random (default: temporal)\n",
         "  --random_seed N          Random seed for reproducibility (random split only)\n",
         "  --min_test_pitches N     Minimum pitches in test period (default: 10)\n",
         "  --min_total_pitches N    Minimum total pitches (default: 50)\n",
         "  --baseline_type TYPE     marginal/conditional/hybrid (default: conditional)\n",
         "  --standardize WHICH      test/train population anchor for Deception+ (default: test)\n",
         "  --decay N                multinom weight decay / L2 penalty (default: 1e-4)\n",
         "  --prob_shrinkage N       prior mass mixed in before -log(p) (default: 0.02)\n",
         "  --train_game_type TYPE   R/P/S (default: R)\n",
         "  --test_game_type TYPE    R/P/S/W (default: R)\n",
         "  --out_model PATH         Output model path\n",
         "  --out_ppi PATH           Output CSV path\n")
  }

  if (is.null(train_start) || is.null(train_end)) {
    stop("Usage: Rscript pitch_ppi.R --train_start YYYY-MM-DD --train_end YYYY-MM-DD [options]\n")
  }
  
  res <- train_and_save(
    train_start, train_end,
    test_start, test_end,
    min_test_pitches = min_test,
    min_total_pitches = min_total,
    feature_names = features,
    baseline_keys = base_keys,
    baseline_type = baseline_type,
    train_game_type = train_game_type,
    test_game_type = test_game_type,
    train_level = train_level,
    test_level = test_level,
    split_method = split_method,
    random_seed = random_seed,
    decay = decay,
    prob_shrinkage = prob_shrinkage,
    standardize = standardize,
    out_model = out_model,
    out_ppi   = out_ppi,
    verbose   = TRUE
  )
  
  cat("\n==== Head of pitcher_ppi (written to CSV) ====\n")
  print(head(res$pitcher_ppi, 10))
  
  # Generate visualizations
  create_visualizations(res)
}
