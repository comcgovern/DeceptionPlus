# ============================================================================
# statcast_fetch_by_gamepk.R — Fetch Statcast pitch-by-pitch data by game_pk
# ----------------------------------------------------------------------------
# Pulls pitch-level Statcast data for one or more specific game_pk IDs from
# baseballsavant.mlb.com.  Tries the MLB endpoint first; if that returns no
# rows it automatically falls back to the minors endpoint.
#
# Library use:
#   source("statcast_fetch_by_gamepk.R")
#   data <- fetch_statcast_by_gamepk(843975)
#   data <- fetch_statcast_by_gamepk(c(843975, 843976), verbose = TRUE)
#
# CLI use:
#   Rscript statcast_fetch_by_gamepk.R --gamepk 843975
#   Rscript statcast_fetch_by_gamepk.R --gamepk 843975,843976 [--out results.Rds] [--quiet]
#   Rscript statcast_fetch_by_gamepk.R                          # uses default 843975
# ============================================================================

suppressPackageStartupMessages({
  library(dplyr); library(tibble); library(readr)
})

.SAVANT_MLB_BASE    <- "https://baseballsavant.mlb.com/statcast_search/csv"
.SAVANT_MINORS_BASE <- "https://baseballsavant.mlb.com/statcast-search-minors/csv"

# ---------------------- URL builders ----------------------------------------
.build_mlb_gamepk_url <- function(game_pk) {
  paste0(.SAVANT_MLB_BASE,
         "?all=true&type=details&player_type=pitcher",
         "&game_pk=", as.integer(game_pk))
}

.build_minors_gamepk_url <- function(game_pk) {
  paste0(.SAVANT_MINORS_BASE,
         "?all=true&type=details&minors=true&player_type=pitcher",
         "&game_pk=", as.integer(game_pk))
}

# ---------------------- Single-game fetch ------------------------------------
# Returns a tibble (empty tibble on failure). `source` output arg tells the
# caller which endpoint succeeded: "mlb", "minors", or "none".
.fetch_one_gamepk <- function(game_pk, verbose = TRUE,
                               max_retries = 3, request_timeout = 120) {
  if (!requireNamespace("httr", quietly = TRUE)) {
    stop("Please install 'httr': install.packages('httr')")
  }

  endpoints <- list(
    list(label = "MLB",    url = .build_mlb_gamepk_url(game_pk)),
    list(label = "minors", url = .build_minors_gamepk_url(game_pk))
  )

  for (ep in endpoints) {
    if (verbose) message(sprintf("  game_pk %d | trying %s endpoint...", game_pk, ep$label))
    if (verbose) message("    URL: ", ep$url)

    # Warm-up ping (mirrors the pattern in statcast_fetch.R)
    try(httr::GET(ep$url, httr::timeout(1)), silent = TRUE)

    result <- NULL
    for (attempt in seq_len(max_retries)) {
      if (attempt > 1 && verbose) {
        message(sprintf("    Retry %d/%d...", attempt, max_retries))
      }

      resp <- try(httr::GET(ep$url, httr::timeout(request_timeout)), silent = TRUE)

      if (inherits(resp, "try-error")) {
        if (verbose) message("    x Connection error"); next
      }
      if (httr::http_error(resp)) {
        if (verbose) message("    x HTTP ", httr::status_code(resp)); next
      }

      content <- httr::content(resp, as = "text", encoding = "UTF-8")

      if (nchar(content) < 100) {
        if (verbose) message("    . Empty response")
        result <- tibble::tibble(); break
      }
      if (grepl("<html", content, ignore.case = TRUE)) {
        if (verbose) message("    x Got HTML error page"); next
      }

      parsed <- try(readr::read_csv(content, show_col_types = FALSE), silent = TRUE)
      if (inherits(parsed, "try-error")) {
        if (verbose) message("    x CSV parse error"); next
      }

      if (nrow(parsed) == 0) {
        if (verbose) message("    . 0 rows from CSV")
        result <- tibble::tibble(); break
      }

      if (nrow(parsed) == 25000) {
        warning(sprintf("game_pk %d returned exactly 25,000 rows — data may be truncated", game_pk))
      }

      if (verbose) message(sprintf("    + %d rows (%s endpoint)", nrow(parsed), ep$label))
      result <- tibble::as_tibble(parsed)
      break
    }

    if (!is.null(result) && nrow(result) > 0) return(result)
  }

  if (verbose) message(sprintf("  game_pk %d | no data found on any endpoint", game_pk))
  tibble::tibble()
}

# ---------------------- Public entry point ----------------------------------
#' Fetch Statcast pitch-by-pitch data for one or more game_pk IDs.
#'
#' @param game_pks  Integer vector of game_pk values.
#' @param verbose   Print progress messages (default TRUE).
#' @return A tibble with all pitches; empty tibble if nothing is found.
fetch_statcast_by_gamepk <- function(game_pks = 843975L, verbose = TRUE) {
  game_pks <- as.integer(unique(game_pks))
  if (length(game_pks) == 0) stop("game_pks must contain at least one value")

  if (verbose) {
    message(sprintf("Fetching Statcast data for %d game_pk(s): %s",
                    length(game_pks), paste(game_pks, collapse = ", ")))
  }

  results <- lapply(game_pks, function(pk) {
    .fetch_one_gamepk(pk, verbose = verbose)
  })

  combined <- dplyr::bind_rows(results)

  if (verbose) {
    if (nrow(combined) == 0) {
      message("No data returned for any of the requested game_pk(s).")
    } else {
      message(sprintf("Total rows fetched: %d across %d column(s)",
                      nrow(combined), ncol(combined)))
    }
  }

  combined
}

# ---------------------- CLI helpers -----------------------------------------
.parse_cli_arg <- function(args, flag, default = NULL) {
  hit <- which(args == flag)
  if (length(hit) == 1L && hit < length(args)) args[hit + 1L] else default
}

.cli_main <- function(args) {
  raw_pks <- .parse_cli_arg(args, "--gamepk", "843975")
  game_pks <- as.integer(strsplit(raw_pks, "[,;[:space:]]+")[[1]])
  game_pks <- game_pks[!is.na(game_pks)]
  if (length(game_pks) == 0) stop("Could not parse any valid game_pk from: ", raw_pks)

  out   <- .parse_cli_arg(args, "--out", NULL)
  quiet <- "--quiet" %in% args

  message("statcast_fetch_by_gamepk | game_pk(s): ", paste(game_pks, collapse = ", "))

  data <- fetch_statcast_by_gamepk(game_pks, verbose = !quiet)

  if (is.null(data) || nrow(data) == 0) {
    message("RESULT: 0 rows returned.")
    return(invisible(1L))
  }

  message(sprintf("RESULT: %d rows, %d columns", nrow(data), ncol(data)))

  if (!quiet) {
    preview_cols <- intersect(
      c("game_date", "game_pk", "home_team", "away_team",
        "pitcher", "player_name", "pitch_type", "release_speed",
        "events", "description"),
      names(data)
    )
    if (length(preview_cols)) {
      message("Sample (first 5 rows):")
      print(utils::head(data[, preview_cols, drop = FALSE], 5))
    }
    if ("pitch_type" %in% names(data)) {
      message("Pitch type breakdown:")
      print(sort(table(data$pitch_type), decreasing = TRUE))
    }
  }

  if (!is.null(out)) {
    out_dir <- dirname(out)
    if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
    saveRDS(data, out)
    message("Saved to: ", out)
  }

  invisible(0L)
}

# Run CLI only when invoked directly via Rscript (not when sourced).
if (sys.nframe() == 0L && !interactive()) {
  .cli_args <- commandArgs(trailingOnly = TRUE)
  status <- tryCatch(.cli_main(.cli_args),
                     error = function(e) { message("Error: ", conditionMessage(e)); 1L })
  quit(save = "no", status = if (is.null(status)) 0L else as.integer(status))
}
