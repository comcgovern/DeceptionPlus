# ============================================================================
# statcast_fetch.R — Independent Statcast pitch-by-pitch fetcher
# ----------------------------------------------------------------------------
# Pulls pitch-level Statcast data from baseballsavant.mlb.com for any level
# of professional baseball that Statcast publishes.
#
# Supported levels (case-insensitive aliases listed):
#   MLB                    Major League (sabRmetrics::download_baseballsavant)
#   AAA  / TripleA         Triple-A           (sport_id 11)
#   AA   / DoubleA         Double-A           (sport_id 12)
#   A+   / HighA           High-A             (sport_id 13)
#   A    / SingleA / LowA  Single-A           (sport_id 14)
#   ROK  / Rookie          All Rookie/Complex (sport_id 16)
#   FCL                    Florida Complex League  (sport_id 16, hfLevel=ROK)
#   ACL                    Arizona Complex League  (sport_id 17, hfLevel=ROK)
#
# Library use:
#   source("statcast_fetch.R")
#   data <- load_statcast_range("2026-04-15", "2026-04-30", level = "FCL")
#
# CLI use:
#   Rscript statcast_fetch.R --level FCL --start 2026-04-15 --end 2026-04-30 \
#                            [--game-type R] [--out cache/fcl_2026.Rds] [--quiet]
#
#   Diagnostic ping (small window, prints row counts only):
#   Rscript statcast_fetch.R --diagnose FCL [--start 2026-04-15 --end 2026-04-21]
#
#   List supported levels:
#   Rscript statcast_fetch.R --list-levels
#
# Note: FCL Statcast coverage begins in 2026 — earlier dates will return 0 rows
# even when the URL/parameters are correct.
# ============================================================================

suppressPackageStartupMessages({
  library(dplyr); library(tibble); library(readr)
})

# ---------------------- Level configuration ---------------------------------
# Each entry maps a normalized level key to the URL parameters needed by the
# baseballsavant.mlb.com/statcast-search-minors/csv endpoint.
#
#   hf_level: value to use in &hfLevel= (NULL omits filter)
#   sport_id: value to use in &sport_id= (NULL omits filter)
#   minors:   TRUE for the minors endpoint, FALSE for MLB
#
# When the user passes "FCL" or "ACL" we send hfLevel=ROK PLUS sport_id so
# Savant narrows from "all rookie/complex" to the specific complex league.
.statcast_level_map <- list(
  "MLB"     = list(minors = FALSE, hf_level = NULL,  sport_id = 1,  display = "MLB"),
  "AAA"     = list(minors = TRUE,  hf_level = "AAA", sport_id = 11, display = "AAA (Triple-A)"),
  "AA"      = list(minors = TRUE,  hf_level = "AA",  sport_id = 12, display = "AA (Double-A)"),
  "A+"      = list(minors = TRUE,  hf_level = "A+",  sport_id = 13, display = "A+ (High-A)"),
  "A"       = list(minors = TRUE,  hf_level = "A",   sport_id = 14, display = "A (Single-A)"),
  "ROK"     = list(minors = TRUE,  hf_level = "ROK", sport_id = NULL, display = "ROK (all Rookie/Complex)"),
  "FCL"     = list(minors = TRUE,  hf_level = "ROK", sport_id = 16, display = "FCL (Florida Complex League)"),
  "ACL"     = list(minors = TRUE,  hf_level = "ROK", sport_id = 17, display = "ACL (Arizona Complex League)")
)

.statcast_aliases <- list(
  "MAJORS" = "MLB", "MAJOR" = "MLB", "ML" = "MLB",
  "TRIPLEA" = "AAA", "TRIPLE-A" = "AAA",
  "DOUBLEA" = "AA",  "DOUBLE-A" = "AA",
  "HIGHA" = "A+",    "HIGH-A" = "A+",
  "SINGLEA" = "A",   "SINGLE-A" = "A", "LOWA" = "A", "LOW-A" = "A",
  "ROOKIE" = "ROK",
  "FLORIDA" = "FCL", "FLORIDACOMPLEX" = "FCL", "FLORIDACOMPLEXLEAGUE" = "FCL",
  "ARIZONA" = "ACL", "ARIZONACOMPLEX" = "ACL", "ARIZONACOMPLEXLEAGUE" = "ACL"
)

normalize_level <- function(level) {
  if (is.null(level) || is.na(level) || !nzchar(level)) return("MLB")
  key <- toupper(gsub("[[:space:]_]", "", level))
  if (key %in% names(.statcast_level_map)) return(key)
  if (key %in% names(.statcast_aliases)) return(.statcast_aliases[[key]])
  stop("Unknown level: '", level, "'. Run statcast_supported_levels() to see options.")
}

statcast_supported_levels <- function() {
  tibble::tibble(
    key       = names(.statcast_level_map),
    display   = vapply(.statcast_level_map, `[[`, character(1), "display"),
    minors    = vapply(.statcast_level_map, `[[`, logical(1),   "minors"),
    sport_id  = vapply(.statcast_level_map, function(x) {
                  v <- x$sport_id; if (is.null(v)) NA_integer_ else as.integer(v)
                }, integer(1)),
    hf_level  = vapply(.statcast_level_map, function(x) {
                  v <- x$hf_level; if (is.null(v)) NA_character_ else as.character(v)
                }, character(1))
  )
}

statcast_level_config <- function(level) {
  key <- normalize_level(level)
  cfg <- .statcast_level_map[[key]]
  cfg$key <- key
  cfg
}

# ---------------------- URL construction ------------------------------------
# Builds a single statcast-search-minors CSV URL for one date chunk.
build_minors_url <- function(start_chunk, end_chunk, season, game_type, cfg) {
  start_str <- format(as.Date(start_chunk), "%Y-%m-%d")
  end_str   <- format(as.Date(end_chunk),   "%Y-%m-%d")
  pieces <- c(
    "all=true",
    "type=details",
    "minors=true",
    "player_type=pitcher",
    paste0("hfGT=", utils::URLencode(game_type, reserved = TRUE), "%7C"),
    sprintf("game_date_gt=%s&game_date_lt=%s", start_str, end_str),
    paste0("hfSea=", season, "%7C"),
    "group_by=name",
    "min_pitches=0",
    "min_results=0"
  )
  if (!is.null(cfg$hf_level)) {
    pieces <- c(pieces, paste0("hfLevel=", utils::URLencode(cfg$hf_level, reserved = TRUE), "%7C"))
  }
  if (!is.null(cfg$sport_id)) {
    pieces <- c(pieces, paste0("sport_id=", cfg$sport_id))
  }
  paste0("https://baseballsavant.mlb.com/statcast-search-minors/csv?",
         paste(pieces, collapse = "&"))
}

# ---------------------- MLB loader (sabRmetrics) ----------------------------
.load_mlb_range <- function(start_date, end_date, game_type, verbose) {
  if (!requireNamespace("sabRmetrics", quietly = TRUE)) {
    stop("Please install 'sabRmetrics': devtools::install_github('saberpowers/sabRmetrics')")
  }
  if (verbose) message("Downloading Savant (MLB): ", start_date, " -> ", end_date,
                       " | game_type=", game_type)

  start_d <- as.Date(start_date); end_d <- as.Date(end_date)
  years   <- seq(as.integer(format(start_d, "%Y")), as.integer(format(end_d, "%Y")))

  # sabRmetrics::download_baseballsavant() does not support cross-year queries.
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
  dplyr::bind_rows(year_chunks)
}

# ---------------------- Minors loader (chunked HTTP) ------------------------
.load_minors_range <- function(start_date, end_date, game_type, cfg, verbose,
                                chunk_days = 5, max_retries = 3,
                                request_timeout = 120) {
  if (!requireNamespace("httr", quietly = TRUE)) {
    stop("Please install 'httr': install.packages('httr')")
  }
  if (!requireNamespace("readr", quietly = TRUE)) {
    stop("Please install 'readr': install.packages('readr')")
  }

  if (verbose) message("Downloading Savant (", cfg$display, "): ",
                       start_date, " -> ", end_date, " | game_type=", game_type)

  start <- as.Date(start_date); end <- as.Date(end_date)
  days  <- as.numeric(end - start)

  starts  <- seq(start, by = chunk_days, length.out = ceiling((days + 1) / chunk_days))
  ends    <- pmin(starts + (chunk_days - 1), end)
  season  <- format(start, "%Y")
  urls    <- vapply(seq_along(starts),
                    function(i) build_minors_url(starts[i], ends[i], season, game_type, cfg),
                    character(1))
  payload <- tibble::tibble(
    start_chunk = starts,
    end_chunk   = ends,
    chunk_id    = seq_along(starts),
    url         = urls
  )

  n_chunks <- nrow(payload)
  if (verbose) message("Downloading ", n_chunks, " chunk(s) of ", chunk_days, " day(s)...")

  # Step 1: Submit initial requests so Savant warms up the query (sabRmetrics pattern).
  if (verbose) message("Submitting initial API requests...")
  invisible(lapply(payload$url, function(url) {
    try(httr::GET(url, httr::timeout(1)), silent = TRUE)
  }))

  if (verbose) message("Downloading data chunks...")
  data_list  <- vector("list", n_chunks)
  is_error   <- rep(TRUE, n_chunks)

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

      response <- try(httr::GET(payload$url[i], httr::timeout(request_timeout)),
                      silent = TRUE)
      if (inherits(response, "try-error")) {
        if (verbose) message("    x Connection error"); next
      }
      if (httr::http_error(response)) {
        if (verbose) message("    x HTTP ", httr::status_code(response)); next
      }

      content <- httr::content(response, as = "text", encoding = "UTF-8")
      if (nchar(content) < 100) {
        if (verbose) message("    . No data (likely no games)")
        data_list[[i]] <- NULL; is_error[i] <- FALSE; next
      }
      if (grepl("<html", content, ignore.case = TRUE)) {
        if (verbose) message("    x Got HTML error page"); next
      }

      chunk_data <- try(readr::read_csv(content, show_col_types = FALSE), silent = TRUE)
      if (inherits(chunk_data, "try-error")) {
        if (verbose) message("    x CSV parse error"); next
      }
      if (nrow(chunk_data) == 0) {
        if (verbose) message("    . No data")
        data_list[[i]] <- NULL; is_error[i] <- FALSE; next
      }

      data_list[[i]] <- chunk_data
      is_error[i]    <- FALSE
      if (verbose) message("    + ", nrow(chunk_data), " rows")

      if (nrow(chunk_data) == 25000) {
        warning(sprintf("Chunk %d returned exactly 25,000 rows - data may be truncated", i))
      }
      Sys.sleep(1)
    }
  }

  if (any(is_error)) {
    warning(sprintf("%d chunk(s) failed after %d retries", sum(is_error), max_retries))
  }

  successful <- data_list[!sapply(data_list, is.null)]
  if (length(successful) == 0) {
    warning("No ", cfg$display, " data returned for this window.")
    return(tibble())
  }

  combined <- dplyr::bind_rows(successful)
  if (any(sapply(successful, nrow) == 25000)) {
    warning(sprintf("%d chunk(s) returned exactly 25,000 rows. Data are likely missing.",
                    sum(sapply(successful, nrow) == 25000)))
  }
  if (verbose) message("Total ", cfg$display, " rows: ", nrow(combined))
  tibble::as_tibble(combined)
}

# ---------------------- Public entry point ----------------------------------
load_statcast_range <- function(start_date, end_date, game_type = "R",
                                level = "MLB", verbose = TRUE) {
  cfg <- statcast_level_config(level)
  if (!cfg$minors) {
    return(.load_mlb_range(start_date, end_date, game_type, verbose))
  }
  .load_minors_range(start_date, end_date, game_type, cfg, verbose)
}

# ---------------------- CLI entry point -------------------------------------
.parse_cli_arg <- function(args, flag, default = NULL) {
  hit <- which(args == flag)
  if (length(hit) == 1 && hit < length(args)) args[hit + 1] else default
}

.cli_main <- function(args) {
  if ("--list-levels" %in% args || "-l" %in% args) {
    print(statcast_supported_levels())
    return(invisible(0))
  }

  diagnose <- "--diagnose" %in% args
  if (diagnose) {
    idx <- which(args == "--diagnose")
    level <- if (idx < length(args)) args[idx + 1] else stop("--diagnose requires a level argument")
  } else {
    level <- .parse_cli_arg(args, "--level", "MLB")
  }

  start <- .parse_cli_arg(args, "--start", as.character(Sys.Date() - 7))
  end   <- .parse_cli_arg(args, "--end",   as.character(Sys.Date() - 1))
  gtype <- .parse_cli_arg(args, "--game-type", "R")
  out   <- .parse_cli_arg(args, "--out", NULL)
  quiet <- "--quiet" %in% args

  cfg <- statcast_level_config(level)
  message("Statcast fetch: level=", cfg$display,
          " | start=", start, " | end=", end, " | game_type=", gtype)

  if (cfg$minors) {
    sample_url <- build_minors_url(start, end, format(as.Date(start), "%Y"), gtype, cfg)
    message("Sample URL:\n  ", sample_url)
  }

  data <- load_statcast_range(start, end, game_type = gtype, level = cfg$key,
                              verbose = !quiet)

  if (is.null(data) || nrow(data) == 0) {
    message("RESULT: 0 rows returned. ",
            if (cfg$key == "FCL") {
              "FCL Statcast coverage begins in 2026 - confirm the date window is in 2026 or later. If it is, try --diagnose ROK over the same window: if ROK returns rows but FCL does not, the public minors endpoint isn't honoring sport_id=16 and FCL pitches will need to be filtered post-hoc by team."
            } else if (cfg$key == "ACL") {
              "If ACL is empty, try --diagnose ROK to confirm any rookie data is exposed; if ROK returns rows but ACL does not, the public minors endpoint isn't honoring sport_id=17."
            } else "")
    return(invisible(1))
  }

  message("RESULT: ", nrow(data), " rows, ", ncol(data), " columns")
  if (!quiet) {
    cols_to_show <- intersect(c("game_date", "game_pk", "home_team", "away_team",
                                 "pitcher", "player_name", "pitch_type", "level"),
                              names(data))
    if (length(cols_to_show)) {
      message("Sample (first 5 rows):")
      print(utils::head(data[, cols_to_show], 5))
    }
    if ("game_date" %in% names(data)) {
      gd <- as.character(data$game_date)
      message("Date range in data: ", min(gd, na.rm = TRUE), " to ", max(gd, na.rm = TRUE))
    }
    if ("home_team" %in% names(data)) {
      message("Distinct teams: ", length(unique(c(data$home_team, data$away_team))))
    }
  }

  if (!is.null(out)) {
    out_dir <- dirname(out)
    if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
    saveRDS(data, out)
    message("Saved to: ", out)
  }
  invisible(0)
}

# Run CLI only when invoked directly via Rscript (not when sourced).
if (sys.nframe() == 0L && !interactive()) {
  .cli_args <- commandArgs(trailingOnly = TRUE)
  status <- tryCatch(.cli_main(.cli_args),
                     error = function(e) { message("Error: ", conditionMessage(e)); 1 })
  quit(save = "no", status = if (is.null(status)) 0 else as.integer(status))
}
