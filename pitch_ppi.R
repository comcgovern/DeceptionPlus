# ============================================================================
# pitch_ppi.R — Pitch Type Predictability (PPI) with Flexible Period Selection
# ----------------------------------------------------------------------------
# - Downloads (and caches) Statcast pitches via sabRmetrics::download_baseballsavant()
# - Resolves pitcher names from MLB StatsAPI with on-disk cache; baseballr fallback
# - Trains multinomial model on one period; evaluates on another (can be same/overlapping)
# - Outputs per-pitcher table with PPI + Predict+ (avg=100; 10 pts = 1 SD)
# - Supports multiple baseline models: "marginal", "conditional", "hybrid"
# - Organizes outputs into subfolders: cache/, models/, output/
# ============================================================================

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(purrr); library(stringr); library(lubridate)
  library(nnet); library(readr); library(tibble); library(forcats)
})

`%||%` <- function(a, b) if (!is.null(a)) a else b
safe_div <- function(a, b) ifelse(b > 0, a / b, 0)

# Safe wrapper for multinom predict with 2-class edge case handling.
# When nnet::multinom has exactly 2 classes, predict(type="probs") returns
# a vector of P(class2) only. This converts it to a proper probability matrix.
safe_predict_probs <- function(mod, newdata, classes) {
  P <- predict(mod, newdata = newdata, type = "probs")
  if (is.matrix(P) && ncol(P) == length(classes)) return(P)
  # 2-class case: P is a vector of P(second class)
  if (length(classes) == 2 && is.numeric(P) && !is.matrix(P)) {
    P <- cbind(1 - P, P)
    colnames(P) <- classes
    return(P)
  }
  # General fallback: force to matrix
  P <- as.matrix(P)
  if (ncol(P) != length(classes)) {
    P <- matrix(P, nrow = nrow(newdata), ncol = length(classes), dimnames = list(NULL, classes))
  }
  P
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
      stop("Please install 'sabRmetrics' (install.packages('sabRmetrics')).")
    }
    if (verbose) message("Downloading Savant (MLB): ", start_date, " -> ", end_date, " | game_type=", game_type)
    df <- try(sabRmetrics::download_baseballsavant(
      start_date = start_date,
      end_date   = end_date,
      game_type  = game_type,
      cl         = NULL,
      verbose    = verbose
    ), silent = TRUE)
    if (inherits(df, "try-error") || is.null(df) || nrow(df) == 0) {
      warning("No Savant rows returned for this window.")
      return(tibble())
    }
    return(tibble::as_tibble(df))
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
  !is.na(z) & z >= 1L & z <= 9L
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
  df %>% arrange(.data$game_pk, .data$at_bat_number, .data$pitch_number) %>%
    group_by(.data$game_pk, .data$at_bat_number) %>%
    mutate(last_pitch_type = dplyr::lag(.data$pitch_type)) %>%
    ungroup() %>%
    mutate(last_pitch_type = if_else(is.na(.data$last_pitch_type) |
                                       .data$last_pitch_type == "" |
                                       .data$last_pitch_type == "NA",
                                     "NONE", toupper(.data$last_pitch_type)))
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
engineer_features <- function(raw) {
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
    ahead_in_count  = if_else(.data$balls > .data$strikes, 1L, 0L),
    base_state      = base_state_row(.data$on_1b, .data$on_2b, .data$on_3b),
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
  
  bmet <- compute_batter_metrics(df)
  df <- df %>% left_join(bmet, by = "batter") %>%
    mutate(
      o_swing_pct       = coalesce(.data$o_swing_pct, 0.5),
      z_contact_pct     = coalesce(.data$z_contact_pct, 0.5),
      swing_pct         = coalesce(.data$swing_pct, 0.5),
      chase_contact_pct = coalesce(.data$chase_contact_pct, 0.5)
    )
  
  df
}

# ---------------------- NA-safe prep & constant-drop -------------------------
na_safe_factor <- function(x) { x <- as.character(x); x[is.na(x) | x == ""] <- "UNK"; factor(x) }
na_safe_numeric <- function(x) { if (all(is.na(x))) return(rep(0, length(x))); m <- suppressWarnings(median(x, na.rm = TRUE)); x[is.na(x)] <- ifelse(is.finite(m), m, 0); as.numeric(x) }

clean_one_feature <- function(vec) {
  if (is.factor(vec) || is.character(vec)) { v <- na_safe_factor(vec); v <- droplevels(v); list(v = v, ok = nlevels(v) >= 2) }
  else { v <- na_safe_numeric(vec); uniq <- unique(v); list(v = v, ok = length(uniq) >= 2) }
}

prepare_features <- function(df, feature_names) {
  present <- intersect(feature_names, names(df))
  keep <- c(); out <- df
  for (nm in present) {
    res <- clean_one_feature(out[[nm]])
    out[[nm]] <- res$v
    if (res$ok) keep <- c(keep, nm)
  }
  list(data = out, features = unique(keep))
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
compute_baseline_probs <- function(tr_data, te_data, baseline_type = "conditional", 
                                   baseline_keys = c("balls","strikes","is_risp","stand","p_throws","two_strikes")) {
  classes <- levels(tr_data$pitch_class)
  eps <- 1e-12
  
  if (baseline_type == "marginal") {
    # Marginal baseline: overall pitch type distribution with proper Laplace smoothing
    # Ensure ALL classes get a pseudo-count (not just observed ones) so probs sum to 1
    freq <- tibble::tibble(pitch_class = factor(classes, levels = classes)) %>%
      dplyr::left_join(tr_data %>% dplyr::count(pitch_class, name = "n"), by = "pitch_class") %>%
      dplyr::mutate(n = dplyr::coalesce(n, 0L))
    probs <- (freq$n + 1) / sum(freq$n + 1)
    P_base <- matrix(rep(probs, each = nrow(te_data)),
                     nrow = nrow(te_data), ncol = length(classes),
                     dimnames = list(NULL, classes))
    return(P_base)
  }
  
  if (baseline_type == "conditional") {
    # Conditional baseline: conditioned on baseline_keys
    keys_tr <- prune_baseline_keys(tr_data, baseline_keys)
    if (length(keys_tr) == 0) {
      warning("No valid baseline keys; falling back to marginal baseline.")
      return(compute_baseline_probs(tr_data, te_data, baseline_type = "marginal", baseline_keys = baseline_keys))
    }

    keydf <- tr_data %>%
      mutate(across(all_of(keys_tr), ~ if (is.factor(.x) || is.character(.x)) na_safe_factor(.x) else na_safe_numeric(.x))) %>%
      mutate(key = do.call(paste, c(across(all_of(keys_tr)), sep = "_")))
    raw_counts <- keydf %>% count(key, pitch_class, name = "n")
    # Complete grid: every (key, class) pair gets a row — proper Laplace smoothing
    all_keys <- unique(raw_counts$key)
    complete_grid <- tidyr::expand_grid(
      key = all_keys,
      pitch_class = factor(classes, levels = classes)
    )
    counts <- complete_grid %>%
      dplyr::left_join(raw_counts, by = c("key", "pitch_class")) %>%
      dplyr::mutate(n = dplyr::coalesce(n, 0L)) %>%
      dplyr::group_by(key) %>%
      dplyr::mutate(prob = (n + 1) / sum(n + 1)) %>%
      dplyr::ungroup()
    key_te <- te_data %>%
      mutate(across(all_of(keys_tr), ~ if (is.factor(.x) || is.character(.x)) na_safe_factor(.x) else na_safe_numeric(.x))) %>%
      mutate(key = do.call(paste, c(across(all_of(keys_tr)), sep = "_")))
    # Initialize with uniform for test keys not seen in training
    P_base <- matrix(1/length(classes), nrow = nrow(te_data), ncol = length(classes),
                     dimnames = list(NULL, classes))
    counts_wide <- counts %>%
      select(key, pitch_class, prob) %>%
      tidyr::pivot_wider(names_from = pitch_class, values_from = prob, values_fill = NA)
    matched <- dplyr::left_join(
      tibble::tibble(key = key_te$key, .row = seq_len(nrow(key_te))),
      counts_wide, by = "key"
    )
    for (cls in classes) {
      if (cls %in% names(matched) && any(!is.na(matched[[cls]]))) {
        idx <- which(!is.na(matched[[cls]]))
        P_base[matched$.row[idx], cls] <- matched[[cls]][idx]
      }
    }
    return(P_base)
  }
  
  if (baseline_type == "hybrid") {
    # Hybrid: conditional when possible, marginal fallback
    keys_tr <- prune_baseline_keys(tr_data, baseline_keys)

    # Marginal distribution with proper Laplace smoothing over ALL classes
    freq <- tibble::tibble(pitch_class = factor(classes, levels = classes)) %>%
      dplyr::left_join(tr_data %>% dplyr::count(pitch_class, name = "n"), by = "pitch_class") %>%
      dplyr::mutate(n = dplyr::coalesce(n, 0L))
    probs_marginal <- (freq$n + 1) / sum(freq$n + 1)

    if (length(keys_tr) == 0) {
      # No valid keys, use marginal
      P_base <- matrix(rep(probs_marginal, each = nrow(te_data)),
                       nrow = nrow(te_data), ncol = length(classes),
                       dimnames = list(NULL, classes))
      return(P_base)
    }

    # Conditional counts with proper Laplace smoothing
    keydf <- tr_data %>%
      mutate(across(all_of(keys_tr), ~ if (is.factor(.x) || is.character(.x)) na_safe_factor(.x) else na_safe_numeric(.x))) %>%
      mutate(key = do.call(paste, c(across(all_of(keys_tr)), sep = "_")))
    raw_counts <- keydf %>% count(key, pitch_class, name = "n")
    # Complete grid ensures all classes get pseudo-count
    all_keys <- unique(raw_counts$key)
    complete_grid <- tidyr::expand_grid(
      key = all_keys,
      pitch_class = factor(classes, levels = classes)
    )
    counts <- complete_grid %>%
      dplyr::left_join(raw_counts, by = c("key", "pitch_class")) %>%
      dplyr::mutate(n = dplyr::coalesce(n, 0L)) %>%
      dplyr::group_by(key) %>%
      dplyr::mutate(prob = (n + 1) / sum(n + 1)) %>%
      dplyr::ungroup()
    key_counts <- keydf %>% count(key, name = "n_key")

    key_te <- te_data %>%
      mutate(across(all_of(keys_tr), ~ if (is.factor(.x) || is.character(.x)) na_safe_factor(.x) else na_safe_numeric(.x))) %>%
      mutate(key = do.call(paste, c(across(all_of(keys_tr)), sep = "_")))

    # Initialize with marginal (used for keys with insufficient data or unseen keys)
    P_base <- matrix(rep(probs_marginal, each = nrow(te_data)),
                     nrow = nrow(te_data), ncol = length(classes),
                     dimnames = list(NULL, classes))

    # Replace with conditional where we have sufficient data (>= 5 observations)
    min_obs <- 5
    qualified_keys <- key_counts %>% filter(n_key >= min_obs) %>% pull(key)
    counts_qualified <- counts %>% filter(key %in% qualified_keys)
    if (nrow(counts_qualified) > 0) {
      counts_wide <- counts_qualified %>%
        select(key, pitch_class, prob) %>%
        tidyr::pivot_wider(names_from = pitch_class, values_from = prob, values_fill = NA)
      matched <- dplyr::left_join(
        tibble::tibble(key = key_te$key, .row = seq_len(nrow(key_te))),
        counts_wide, by = "key"
      )
      for (cls in classes) {
        if (cls %in% names(matched) && any(!is.na(matched[[cls]]))) {
          idx <- which(!is.na(matched[[cls]]))
          P_base[matched$.row[idx], cls] <- matched[[cls]][idx]
        }
      }
    }
    return(P_base)
  }
  
  stop("Unknown baseline_type: ", baseline_type, ". Use 'marginal', 'conditional', or 'hybrid'.")
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
#' @param verbose Print progress messages
#' @return List with:
#'   - results: Data frame with per-pitcher unpredictability metrics
#'   - excluded: Data frame with pitchers who couldn't be evaluated and why
evaluate_per_pitcher <- function(df_history,
                                  df_test,
                                  min_train_pitches = 100,
                                  min_test_pitches = 10,
                                  feature_names = c("balls", "strikes", "two_strikes",
                                                    "ahead_in_count", "outs", "is_risp",
                                                    "stand", "last_pitch_type"),
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
        mean_surp_model = numeric(),
        mean_surp_base = numeric(),
        unpredictability_ratio = numeric()
      ),
      excluded = excluded
    ))
  }

  if (verbose) message("Evaluating ", length(valid_pitchers), " pitchers with per-pitcher models...")

  eps <- 1e-9
  results <- vector("list", length(valid_pitchers))

  for (i in seq_along(valid_pitchers)) {
    pid <- valid_pitchers[i]

    # Get this pitcher's data
    ptr_history <- df_history %>% filter(pitcher_id == pid)
    ptr_test <- df_test %>% filter(pitcher_id == pid)

    # Prepare features for this pitcher's historical data
    pf_tr <- prepare_features(ptr_history, feature_names)
    tr2 <- pf_tr$data
    feats <- pf_tr$features

    # Align test data with training pitch class levels
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
      # Align test factor levels to training levels
      for (nm in feats) {
        if (is.factor(tr2[[nm]]) && nm %in% names(te2) && is.factor(te2[[nm]])) {
          te2[[nm]] <- factor(te2[[nm]], levels = levels(tr2[[nm]]))
        }
      }
      # Drop rows with unseen factor levels (now NA)
      feat_complete <- complete.cases(te2[, intersect(feats, names(te2)), drop = FALSE])
      te2 <- te2[feat_complete, , drop = FALSE]
    }
    if (nrow(te2) == 0) next

    # Check we have enough pitch classes to fit model
    if (nlevels(droplevels(tr2$pitch_class)) < 2) next

    # Fit per-pitcher multinomial model
    form <- if (length(feats) == 0) as.formula("pitch_class ~ 1")
            else as.formula(paste("pitch_class ~", paste(feats, collapse = " + ")))

    mod <- try(nnet::multinom(form, data = tr2, trace = FALSE, maxit = 200), silent = TRUE)
    if (inherits(mod, "try-error")) next

    classes <- levels(tr2$pitch_class)

    # Model predictions on test data
    P <- safe_predict_probs(mod, te2, classes)

    idx_true <- match(as.character(te2$pitch_class), classes)
    if (any(is.na(idx_true))) next

    p_true <- P[cbind(seq_len(nrow(te2)), idx_true)]
    surp_model <- -log(pmax(p_true, eps))

    # Baseline: pitcher's own marginal pitch frequencies from history
    # This represents "how surprising is this pitch given the pitcher's overall arsenal?"
    pitch_freqs <- table(tr2$pitch_class)
    pitch_probs <- (pitch_freqs + 1) / sum(pitch_freqs + 1)  # Add-1 smoothing

    p_base <- pitch_probs[as.character(te2$pitch_class)]
    surp_base <- -log(pmax(as.numeric(p_base), eps))

    # Aggregate for this pitcher
    results[[i]] <- tibble(
      pitcher_id = pid,
      n_history = nrow(ptr_history),
      n_test = nrow(ptr_test),
      mean_surp_model = mean(surp_model),
      mean_surp_base = mean(surp_base),
      unpredictability_ratio = mean(surp_model) / pmax(mean(surp_base), eps)
    )
  }

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
  # For each game + team side (top/bottom), find the pitcher who threw the first pitch
  # The pitcher with the minimum at_bat_number in inning 1 is the starter

  # First, identify starters: the pitcher who pitched first for each team in each game
  starters <- df %>%
    dplyr::filter(inning == 1) %>%
    dplyr::group_by(game_pk, inning_topbot) %>%
    dplyr::arrange(at_bat_number) %>%
    dplyr::slice_head(n = 1) %>%
    dplyr::ungroup() %>%
    dplyr::select(game_pk, pitcher_id) %>%
    dplyr::distinct() %>%
    dplyr::mutate(is_starter = TRUE)


  # Get all pitcher-game combinations
  all_appearances <- df %>%
    dplyr::select(game_pk, pitcher_id) %>%
    dplyr::distinct()

  # Join and classify
  pitcher_roles <- all_appearances %>%
    dplyr::left_join(starters, by = c("game_pk", "pitcher_id")) %>%
    dplyr::mutate(
      is_starter = dplyr::coalesce(is_starter, FALSE),
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
                      feature_names = c("balls","strikes","two_strikes","ahead_in_count",
                                        "is_top","outs","score_diff","base_state","is_risp",
                                        "high_leverage","times_through_order",
                                        "stand","p_throws","last_pitch_type",
                                        "o_swing_pct","z_contact_pct","swing_pct","chase_contact_pct"),
                      baseline_keys = c("balls","strikes","is_risp","stand","p_throws","two_strikes"),
                      baseline_type = "conditional",
                      train_game_type = "R",
                      test_game_type = "R",
                      train_level = "MLB",
                      test_level = "MLB",
                      split_method = "temporal",
                      random_seed = NULL,
                      verbose = TRUE) {

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

    df_all <- engineer_features(raw_all)
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

    df_train <- engineer_features(raw_train)
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

    df_test <- engineer_features(raw_test)
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
  
  pf_tr <- prepare_features(df_train, feature_names)
  tr2   <- pf_tr$data
  feats <- pf_tr$features
  
  form <- if (length(feats) == 0) as.formula("pitch_class ~ 1")
  else as.formula(paste("pitch_class ~", paste(feats, collapse = " + ")))
  
  if (verbose) message("Model formula: ", deparse(form))
  
  mod <- try(nnet::multinom(form, data = tr2, trace = FALSE, maxit = 500), silent = TRUE)
  if (inherits(mod, "try-error")) {
    warning("Multinomial fit failed; retrying with intercept-only model.")
    form <- as.formula("pitch_class ~ 1")
    mod  <- nnet::multinom(form, data = tr2, trace = FALSE, maxit = 500)
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
  
  te2 <- df_test
  if (length(feats) > 0) {
    for (nm in feats) {
      res <- clean_one_feature(te2[[nm]])
      te2[[nm]] <- res$v
    }
  }

  # Align test factor levels to training levels (prevents predict errors from unseen levels)
  for (nm in feats) {
    if (is.factor(tr2[[nm]]) && is.factor(te2[[nm]])) {
      te2[[nm]] <- factor(te2[[nm]], levels = levels(tr2[[nm]]))
    }
  }

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
  P <- safe_predict_probs(mod, te2, classes)

  idx_true <- match(as.character(te2$pitch_class), classes)
  p_true   <- P[cbind(seq_len(nrow(te2)), idx_true)]
  surp_model <- -log(pmax(p_true, eps))
  
  # ========== STEP 6: Baseline predictions and surprise ==========
  if (verbose) message("📐 Computing baseline...")

  P_base <- compute_baseline_probs(tr2, te2, baseline_type = baseline_type, baseline_keys = baseline_keys)
  p_true_base <- P_base[cbind(seq_len(nrow(te2)), idx_true)]
  surp_base   <- -log(pmax(p_true_base, eps))

  # ========== STEP 6b: Compute training baseline for standardization ==========
  # We compute in-sample unpredictability ratios on training data to establish

  # "league average" unpredictability. This makes test period scores comparable
  # across different test dates (given the same training period).
  if (verbose) message("📊 Computing training baseline for standardization...")

  # In-sample predictions on training data
  P_train <- safe_predict_probs(mod, tr2, classes)

  idx_true_train <- match(as.character(tr2$pitch_class), classes)
  p_true_train <- P_train[cbind(seq_len(nrow(tr2)), idx_true_train)]
  surp_model_train <- -log(pmax(p_true_train, eps))

  # Baseline for training data (in-sample)
  P_base_train <- compute_baseline_probs(tr2, tr2, baseline_type = baseline_type, baseline_keys = baseline_keys)
  p_true_base_train <- P_base_train[cbind(seq_len(nrow(tr2)), idx_true_train)]
  surp_base_train <- -log(pmax(p_true_base_train, eps))

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
    mutate(unpredictability_ratio = mean_surp_model / pmax(mean_surp_base, 1e-9))

  # Compute standardization parameters from TRAINING population
  # This establishes "league average" based on the training period
  train_u_mu <- mean(per_pitcher_train$unpredictability_ratio, na.rm = TRUE)
  train_u_sd <- sd(per_pitcher_train$unpredictability_ratio, na.rm = TRUE)

  if (verbose) {
    message("   Training baseline: μ=", round(train_u_mu, 4), " σ=", round(train_u_sd, 4))
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
    mutate(ppi = 1 - (mean_surp_model / pmax(mean_surp_base, 1e-9)),
           ppi = pmin(pmax(ppi, -1), 1),
           unpredictability_ratio = mean_surp_model / pmax(mean_surp_base, 1e-9))
  
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
  
  # ========== STEP 9: Calculate Predict+ ==========
  # Join and filter to get the final population
  pitcher_ppi <- all_pitchers %>%
    left_join(per_pitcher_test, by = "pitcher_id") %>%
    left_join(name_map, by = "pitcher_id") %>%
    filter(total_pitches >= min_total_pitches) %>%
    filter(!is.na(ppi)) %>%
    mutate(pitcher_name = if_else(is.na(pitcher_name), paste0("Pitcher_", pitcher_id), pitcher_name))

  # Standardize using TRAINING population parameters (train_u_mu, train_u_sd)
  # This ensures scores are comparable across different test periods
  # (given the same training baseline). Test mean won't be exactly 100,
  # but scores are anchored to a stable reference population.
  pitcher_ppi <- pitcher_ppi %>%
    mutate(
      predict_plus = 100 + 10 * ((unpredictability_ratio - train_u_mu) / pmax(train_u_sd, 1e-9))
    ) %>%
    select(
      pitcher_id, pitcher_name, total_pitches, n_pitches_test,
      mean_surp_model, mean_surp_base, ppi,
      unpredictability_ratio, predict_plus
    ) %>%
    arrange(desc(predict_plus))
  
  if (verbose) {
    message("\n✅ Analysis complete!")
    message("   Pitchers evaluated: ", nrow(pitcher_ppi))
    message("   Mean Predict+: ", round(mean(pitcher_ppi$predict_plus, na.rm = TRUE), 1))
    message("   Range: ", round(min(pitcher_ppi$predict_plus, na.rm = TRUE), 1), 
            " to ", round(max(pitcher_ppi$predict_plus, na.rm = TRUE), 1))
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
         train_mean = train_u_mu,
         train_sd = train_u_sd,
         n_train_pitchers = nrow(per_pitcher_train)
       ))
}

# ---------------------- Public API: train & save -----------------------------
train_and_save <- function(train_start, train_end,
                           test_start = NULL, test_end = NULL,
                           min_test_pitches = 10,
                           min_total_pitches = 50,
                           feature_names = c("balls","strikes","two_strikes","ahead_in_count",
                                             "is_top","outs","score_diff","base_state","is_risp",
                                             "high_leverage","times_through_order",
                                             "stand","p_throws","last_pitch_type",
                                             "o_swing_pct","z_contact_pct","swing_pct","chase_contact_pct"),
                           baseline_keys = c("balls","strikes","is_risp","stand","p_throws","two_strikes"),
                           baseline_type = "conditional",
                           train_game_type = "R",
                           test_game_type = "R",
                           train_level = "MLB",
                           test_level = "MLB",
                           split_method = "temporal",
                           random_seed = NULL,
                           out_model = "models/ppi_model.rds",
                           out_ppi   = "output/pitcher_ppi.csv",
                           verbose   = TRUE) {

  res <- train_ppi(train_start, train_end,
                   test_start, test_end,
                   min_test_pitches = min_test_pitches,
                   min_total_pitches = min_total_pitches,
                   feature_names = feature_names,
                   baseline_keys = baseline_keys,
                   baseline_type = baseline_type,
                   train_game_type = train_game_type,
                   test_game_type = test_game_type,
                   train_level = train_level,
                   test_level = test_level,
                   split_method = split_method,
                   random_seed = random_seed,
                   verbose = verbose)
  
  saveRDS(list(
    classes = res$classes,
    features_used = res$features_used,
    baseline_keys = res$baseline_keys,
    baseline_type = res$baseline_type,
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
  
  # 1. Predict+ distribution
  p1 <- ggplot(res$pitcher_ppi, aes(x = predict_plus)) +
    geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
    geom_vline(xintercept = 100, linetype = "dashed", color = "red") +
    labs(title = "Distribution of Predict+ Scores",
         subtitle = paste0("Mean = 100, SD = 10 | n = ", nrow(res$pitcher_ppi)),
         x = "Predict+", y = "Count") +
    theme_minimal()
  ggsave(file.path(output_dir, "predict_plus_distribution.png"), p1, width = 10, height = 6)
  
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
    arrange(predict_plus) %>% 
    head(20) %>%
    mutate(pitcher_name = reorder(pitcher_name, -predict_plus))
  
  p3 <- ggplot(top20_pred, aes(x = predict_plus, y = pitcher_name)) +
    geom_col(fill = "coral") +
    geom_vline(xintercept = 100, linetype = "dashed", color = "darkgray") +
    labs(title = "Top 20 Most Predictable Pitchers",
         subtitle = "Lower Predict+ = More Predictable",
         x = "Predict+", y = NULL) +
    theme_minimal()
  ggsave(file.path(output_dir, "top20_predictable.png"), p3, width = 10, height = 8)
  
  # 4. Top 20 Least Predictable
  top20_unpred <- res$pitcher_ppi %>% 
    arrange(desc(predict_plus)) %>% 
    head(20) %>%
    mutate(pitcher_name = reorder(pitcher_name, predict_plus))
  
  p4 <- ggplot(top20_unpred, aes(x = predict_plus, y = pitcher_name)) +
    geom_col(fill = "steelblue") +
    geom_vline(xintercept = 100, linetype = "dashed", color = "darkgray") +
    labs(title = "Top 20 Least Predictable Pitchers",
         subtitle = "Higher Predict+ = Less Predictable",
         x = "Predict+", y = NULL) +
    theme_minimal()
  ggsave(file.path(output_dir, "top20_unpredictable.png"), p4, width = 10, height = 8)
  
  message("✅ Created 4 visualizations in ", output_dir)
  invisible(list(p1 = p1, p2 = p2, p3 = p3, p4 = p4))
}

# ---------------------- Social Media Visualizations ---------------------------
create_social_media_graphics <- function(res,
                                          game_date,
                                          min_pitches_starter = 65,
                                          min_pitches_reliever = 15,
                                          output_dir = "output/visualizations",
                                          top_n = 5) {
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    warning("ggplot2 not available; skipping social media visualizations")
    return(invisible(NULL))
  }

  library(ggplot2)

  # Apply status filter first (if status column exists)
  all_pitchers <- res$pitcher_ppi
  if ("status" %in% names(all_pitchers)) {
    all_pitchers <- all_pitchers %>% filter(status == "evaluated")
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
  theme_social <- function() {
    theme_minimal(base_size = 14) +
      theme(
        # Left-aligned title gives a cleaner editorial look
        plot.title      = element_text(face = "bold", size = 20, hjust = 0,
                                       color = "#1a1a2e", margin = margin(b = 3)),
        plot.subtitle   = element_text(size = 12, hjust = 0, color = "#666666",
                                       margin = margin(b = 10)),
        plot.caption    = element_text(size = 9, color = "#aaaaaa", hjust = 1,
                                       margin = margin(t = 8)),
        # Subtle vertical grid only; no horizontal clutter
        panel.grid.major.x = element_line(color = "#eeeeee", linewidth = 0.5),
        panel.grid.major.y = element_blank(),
        panel.grid.minor   = element_blank(),
        # Pitcher names: bold, easy to read
        axis.text.y     = element_text(size = 12, face = "bold", color = "#2d2d2d",
                                       hjust = 1),
        axis.text.x     = element_text(size = 10, color = "#999999"),
        axis.title.x    = element_text(size = 10, color = "#888888",
                                       margin = margin(t = 6)),
        axis.title.y    = element_blank(),
        # Clean white canvas; faint warm-gray panel area
        plot.background  = element_rect(fill = "#ffffff", color = NA),
        panel.background = element_rect(fill = "#fafafa", color = NA),
        # Extra right/top margin to avoid clipping score labels and title
        plot.margin = margin(22, 36, 14, 20)
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
      arrange(if (reorder_desc) desc(predict_plus) else predict_plus) %>%
      mutate(
        rank       = row_number(),
        name_label = paste0("#", rank, "  ", pitcher_name),
        score_label = sprintf("%.0f", predict_plus)
      )

    # Factor levels: lowest bar at bottom, highest at top (ggplot reads bottom-up)
    data$name_label <- factor(
      data$name_label,
      levels = if (reorder_desc) data$name_label[order(data$predict_plus)]
               else               data$name_label[order(-data$predict_plus)]
    )

    # Track bar ceiling: a bit above the maximum score so labels have room
    x_track <- max(data$predict_plus, 110) * 1.20
    # x-axis left edge: don't start at zero — Predict+ is centred at 100
    x_left  <- min(data$predict_plus, 90) * 0.94
    data$x_track <- x_track   # attach as column for geom_col reference

    p <- ggplot(data, aes(y = name_label)) +
      # Track bars (light gray background behind every bar)
      geom_col(aes(x = x_track),
               fill = "#efefef", width = 0.55, show.legend = FALSE) +
      # Data bars with gradient fill
      geom_col(aes(x = predict_plus, fill = predict_plus),
               width = 0.55, show.legend = FALSE) +
      # League-average reference line
      geom_vline(xintercept = 100, linetype = "dashed",
                 color = "#bbbbbb", linewidth = 0.8) +
      # "AVG" annotation on the reference line
      annotate("text", x = 100, y = 0.42, label = "AVG",
               size = 2.8, color = "#bbbbbb", hjust = 0.5) +
      # Score labels just outside the bar
      geom_text(aes(x = predict_plus, label = score_label),
                hjust = -0.28, size = 4.2, fontface = "bold", color = "#333333") +
      scale_fill_gradient(low = fill_low, high = fill_high) +
      scale_x_continuous(
        expand = expansion(mult = c(0, 0)),
        limits = c(x_left, x_track)
      ) +
      labs(
        title    = title,
        subtitle = subtitle,
        x        = "Predict+  (100 = league average)",
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
      top_starters <- starters %>% arrange(desc(predict_plus)) %>% head(top_n)
      plots$starters_unpredictable <- create_graphic(
        top_starters,
        title = "Least Predictable Starters",
        subtitle = paste0(date_display, "  \u00b7  min ", min_pitches_starter, " pitches"),
        caption = paste0("Higher = less predictable  \u00b7  100 = league average  \u00b7  @PredictPlus  \u00b7  data: Baseball Savant"),
        fill_low = "#4361ee", fill_high = "#7209b7",
        filename = sprintf("social_starters_top%d_unpredictable_%s.png", top_n, date_short),
        reorder_desc = TRUE
      )

      # Starters - Most Predictable
      bottom_starters <- starters %>% arrange(predict_plus) %>% head(top_n)
      plots$starters_predictable <- create_graphic(
        bottom_starters,
        title = "Most Predictable Starters",
        subtitle = paste0(date_display, "  \u00b7  min ", min_pitches_starter, " pitches"),
        caption = paste0("Lower = more predictable  \u00b7  100 = league average  \u00b7  @PredictPlus  \u00b7  data: Baseball Savant"),
        fill_low = "#e63946", fill_high = "#f4a261",
        filename = sprintf("social_starters_top%d_predictable_%s.png", top_n, date_short),
        reorder_desc = FALSE
      )
    }

    # Relievers - Least Predictable
    if (nrow(relievers) > 0) {
      top_relievers <- relievers %>% arrange(desc(predict_plus)) %>% head(top_n)
      plots$relievers_unpredictable <- create_graphic(
        top_relievers,
        title = "Least Predictable Relievers",
        subtitle = paste0(date_display, "  \u00b7  min ", min_pitches_reliever, " pitches"),
        caption = paste0("Higher = less predictable  \u00b7  100 = league average  \u00b7  @PredictPlus  \u00b7  data: Baseball Savant"),
        fill_low = "#2a9d8f", fill_high = "#264653",
        filename = sprintf("social_relievers_top%d_unpredictable_%s.png", top_n, date_short),
        reorder_desc = TRUE
      )

      # Relievers - Most Predictable
      bottom_relievers <- relievers %>% arrange(predict_plus) %>% head(top_n)
      plots$relievers_predictable <- create_graphic(
        bottom_relievers,
        title = "Most Predictable Relievers",
        subtitle = paste0(date_display, "  \u00b7  min ", min_pitches_reliever, " pitches"),
        caption = paste0("Lower = more predictable  \u00b7  100 = league average  \u00b7  @PredictPlus  \u00b7  data: Baseball Savant"),
        fill_low = "#e76f51", fill_high = "#f4a261",
        filename = sprintf("social_relievers_top%d_predictable_%s.png", top_n, date_short),
        reorder_desc = FALSE
      )
    }

    n_graphics <- sum(!sapply(plots, is.null))
    message("Created ", n_graphics, " social media graphics (starters/relievers) in ", output_dir)
  } else {
    # Fallback: No role column, create overall graphics
    top_unpred <- qualified %>% arrange(desc(predict_plus)) %>% head(top_n)
    plots$top_unpredictable <- create_graphic(
      top_unpred,
      title = "Least Predictable Pitchers",
      subtitle = paste0(date_display, "  \u00b7  min ", min_pitches_starter, " pitches"),
      caption = paste0("Higher = less predictable  \u00b7  100 = league average  \u00b7  @PredictPlus  \u00b7  data: Baseball Savant"),
      fill_low = "#4361ee", fill_high = "#7209b7",
      filename = sprintf("social_top%d_unpredictable_%s.png", top_n, date_short),
      reorder_desc = TRUE
    )

    top_pred <- qualified %>% arrange(predict_plus) %>% head(top_n)
    plots$top_predictable <- create_graphic(
      top_pred,
      title = "Most Predictable Pitchers",
      subtitle = paste0(date_display, "  \u00b7  min ", min_pitches_starter, " pitches"),
      caption = paste0("Lower = more predictable  \u00b7  100 = league average  \u00b7  @PredictPlus  \u00b7  data: Baseball Savant"),
      fill_low = "#e63946", fill_high = "#f4a261",
      filename = sprintf("social_top%d_predictable_%s.png", top_n, date_short),
      reorder_desc = FALSE
    )

    message("Created 2 social media graphics in ", output_dir)
  }

  invisible(plots)
}

# ---------------------- CLI --------------------------------------------------
parse_csv_list <- function(x) { if (is.null(x) || is.na(x) || x == "") return(character(0)); trimws(unlist(strsplit(x, ","))) }

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  get_arg <- function(flag, default = NULL) { hit <- which(args == flag); if (length(hit) == 1 && hit < length(args)) args[hit + 1] else default }
  
  train_start <- get_arg("--train_start")
  train_end <- get_arg("--train_end")
  test_start <- get_arg("--test_start")
  test_end <- get_arg("--test_end")
  
  min_test <- suppressWarnings(as.integer(get_arg("--min_test_pitches", "10")))
  min_total <- suppressWarnings(as.integer(get_arg("--min_total_pitches", "50")))
  out_model <- get_arg("--out_model", "models/ppi_model.rds")
  out_ppi   <- get_arg("--out_ppi", "output/pitcher_ppi.csv")
  feat_str  <- get_arg("--features", "balls,strikes,two_strikes,ahead_in_count,is_top,outs,score_diff,base_state,is_risp,high_leverage,times_through_order,stand,p_throws,last_pitch_type,o_swing_pct,z_contact_pct,swing_pct,chase_contact_pct")
  base_str  <- get_arg("--baseline_keys", "balls,strikes,is_risp,stand,p_throws,two_strikes")
  baseline_type <- get_arg("--baseline_type", "conditional")
  train_game_type <- get_arg("--train_game_type", "R")
  test_game_type <- get_arg("--test_game_type", "R")
  features  <- parse_csv_list(feat_str); base_keys <- parse_csv_list(base_str)
  
  train_level <- get_arg("--train_level", "MLB")
  test_level <- get_arg("--test_level", "MLB")
  split_method <- get_arg("--split_method", "temporal")
  random_seed_str <- get_arg("--random_seed", NA)
  random_seed <- if (is.na(random_seed_str)) NULL else suppressWarnings(as.integer(random_seed_str))

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
    out_model = out_model,
    out_ppi   = out_ppi,
    verbose   = TRUE
  )
  
  cat("\n==== Head of pitcher_ppi (written to CSV) ====\n")
  print(head(res$pitcher_ppi, 10))
  
  # Generate visualizations
  create_visualizations(res)
}
