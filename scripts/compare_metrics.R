# ============================================================================
# scripts/compare_metrics.R — which unpredictability metric should Deception+ use?
# ----------------------------------------------------------------------------
# Deception+ currently standardises `unpredictability_ratio`, which measures
# unpredictability RELATIVE to a count-and-handedness baseline. Three other
# quantities are available from the same fit:
#
#   absolute  mean_surp_model                  — nats of surprise, baseline ignored.
#                                                "How hard is this pitcher to guess?"
#   normed    mean_surp_model / log(n_classes) — the same, as a share of the most
#                                                uncertainty that arsenal could
#                                                deliver. "How close to a coin flip
#                                                among your own pitches are you?"
#   excess    mean_surp_model - mean_surp_base — the ratio's comparison, on a
#                                                difference scale.
#
# They answer different questions and, importantly, need different amounts of
# data. This script measures, for whichever metric:
#
#   reliability   between-pitcher variance / total variance, over repeated
#                 independent test windows. 1.0 = every wobble is real; 0.0 =
#                 you are ranking noise.
#   R^2 ~ K       how much of the metric is just "how many pitch types".
#
# Usage:
#   Rscript scripts/compare_metrics.R                 # synthetic archetypes
#   Rscript scripts/compare_metrics.R --real 2025     # a cached real season
# ============================================================================

suppressPackageStartupMessages(source("pitch_ppi.R"))

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(f, d = NULL) { h <- which(args == f); if (length(h) == 1 && h < length(args)) args[h + 1] else d }
REAL_SEASON <- get_arg("--real", NA)

add_metrics <- function(r) {
  r$absolute <- r$mean_surp_model
  r$normed   <- r$mean_surp_model / log(pmax(r$n_classes, 2))
  r$ratio    <- r$unpredictability_ratio
  r$excess   <- r$surp_excess
  r
}
METRICS <- c("absolute", "normed", "ratio", "excess")

# Reliability: split the per-pitcher observations into repeated windows and ask
# how much of the spread is between pitchers rather than within one.
reliability <- function(P, v) {
  m <- P %>% group_by(pitcher_id) %>% summarise(mu = mean(.data[[v]]), n = n(), .groups = "drop")
  m <- m %>% filter(n >= 2)
  if (nrow(m) < 3) return(NA_real_)
  P <- P %>% filter(pitcher_id %in% m$pitcher_id)
  between <- var(m$mu)
  within  <- mean((P[[v]] - m$mu[match(P$pitcher_id, m$pitcher_id)])^2)
  between / (between + within)
}

report <- function(P, title) {
  cat("\n=== ", title, " ===\n", sep = "")
  cat(sprintf("%d pitcher-windows, %d pitchers\n\n",
              nrow(P), length(unique(P$pitcher_id))))
  agg <- P %>% group_by(pitcher_id) %>%
    summarise(K = round(mean(n_classes)),
              across(all_of(METRICS), mean), .groups = "drop")
  cat(sprintf("%-10s %13s %12s\n", "metric", "reliability", "R^2 ~ K"))
  for (v in METRICS) {
    r2 <- tryCatch(summary(lm(agg[[v]] ~ agg$K))$r.squared, error = function(e) NA_real_)
    cat(sprintf("%-10s %13.3f %12.3f\n", v, reliability(P, v), r2))
  }
  invisible(agg)
}

# ---------------------------------------------------------------- synthetic --
if (is.na(REAL_SEASON)) {
  arch <- list(
    list(id=1, lab="1-pitch closer (99% FC)",  types=c("FC","SL"),               mix=c(.99,.01), patt=0),
    list(id=2, lab="2-pitch, no patterns",     types=c("FF","SL"),               mix=c(.5,.5),   patt=0),
    list(id=3, lab="2-pitch, rigid by count",  types=c("FF","SL"),               mix=c(.5,.5),   patt=.9),
    list(id=4, lab="4-pitch, no patterns",     types=c("FF","SL","CH","CU"),     mix=rep(.25,4), patt=0),
    list(id=5, lab="4-pitch, rigid by count",  types=c("FF","SL","CH","CU"),     mix=rep(.25,4), patt=.9),
    list(id=6, lab="5-pitch, no patterns",     types=c("FF","SI","SL","CH","CU"),mix=rep(.2,5),  patt=0),
    list(id=7, lab="position player (95% FF)", types=c("FF","CU"),               mix=c(.95,.05), patt=0)
  )
  gen <- function(a, n, seed, day0 = 1, per_day = 20) {
    set.seed(seed); K <- length(a$types)
    b <- sample(0:3, n, TRUE); s <- sample(0:2, n, TRUE)
    pt <- vapply(seq_len(n), function(i) {
      m <- (1 - a$patt) * a$mix
      j <- 1 + ((b[i]*3 + s[i]) %% K); m[j] <- m[j] + a$patt
      sample(a$types, 1, prob = m) }, character(1))
    day <- day0 + (seq_len(n) - 1) %/% per_day
    data.frame(game_date = as.Date("2025-04-01") + day, game_pk = day, pitcher = a$id,
      batter = sample(1:60,n,TRUE), pitch_type = pt, balls = b, strikes = s,
      outs_when_up = sample(0:2,n,TRUE), inning = sample(1:9,n,TRUE),
      inning_topbot = sample(c("Top","Bot"),n,TRUE), on_1b = NA,
      on_2b = sample(c(NA,1),n,TRUE), on_3b = NA, home_score = 1, away_score = 0,
      stand = sample(c("L","R"),n,TRUE), p_throws = "R", description = "ball",
      zone = sample(1:14,n,TRUE),
      at_bat_number = rep(seq_len(ceiling(n/4)), each = 4)[seq_len(n)],
      pitch_number = rep(1:4, length.out = n), n_thruorder_pitcher = 1L,
      stringsAsFactors = FALSE)
  }
  H <- engineer_features(do.call(rbind, lapply(arch, function(a) gen(a, 600, a$id*101))),
                         include_batter_metrics = FALSE)
  H <- H[!is.na(H$pitcher_id), ]

  cat("Synthetic archetypes with known structure.\n")
  cat("`patt` pitchers follow a rigid count rule; the others follow none.\n")

  for (w in c(20, 100, 1500)) {
    T <- engineer_features(
      do.call(rbind, lapply(arch, function(a) gen(a, w*20, a$id*307, day0 = 31, per_day = w))),
      include_batter_metrics = FALSE)
    T <- T[!is.na(T$pitcher_id), ]
    acc <- list()
    for (d in unique(T$game_pk)) {
      r <- evaluate_per_pitcher(H, T[T$game_pk == d, ], min_train_pitches = 100,
                                min_test_pitches = 5, verbose = FALSE)$results
      if (nrow(r)) acc[[length(acc) + 1]] <- r
    }
    agg <- report(add_metrics(bind_rows(acc)), paste0("test window = ", w, " pitches"))
    if (w == 20) {
      lab <- sapply(arch, `[[`, "lab"); names(lab) <- sapply(arch, `[[`, "id")
      agg$archetype <- lab[as.character(agg$pitcher_id)]
      cat("\n  per-archetype means:\n")
      print(as.data.frame(agg[order(-agg$normed), c("archetype","K",METRICS)]),
            digits = 3, row.names = FALSE)
    }
  }
  quit(save = "no", status = 0)
}

# --------------------------------------------------------------------- real --
# Score a run of real game-days the way run_daily.R does, then compare metrics.
season <- as.integer(REAL_SEASON)
cands <- c(sprintf("cache/savant_partial_%d_R_MLB.Rds", season),
           Sys.glob(sprintf("cache/savant_raw_%d-*_%d-*_R_MLB.Rds", season, season)))
cachefile <- cands[file.exists(cands)][1]
if (is.na(cachefile)) {
  stop("No cached ", season, " MLB data found. Looked for:\n  ",
       paste(cands, collapse = "\n  "),
       "\nRun run_daily.R or compute_baseline.R first to populate cache/.")
}
message("Using ", cachefile)
df_all <- engineer_features(readRDS(cachefile), include_batter_metrics = FALSE) %>%
  filter(!is.na(pitcher_id))
df_all$game_day <- as.Date(df_all$game_date)

days <- sort(unique(df_all$game_day))
days <- days[days >= min(days) + 45]
set.seed(1); sampled <- sort(sample(days, min(25, length(days))))
message("Scoring ", length(sampled), " game-days...")

acc <- list()
for (d in sampled) {
  d <- as.Date(d, origin = "1970-01-01")
  te <- df_all %>% filter(game_day == d)
  hi <- get_pitcher_history(df_all %>% filter(game_day < d), unique(te$pitcher_id), 500)
  if (!nrow(hi) || !nrow(te)) next
  r <- evaluate_per_pitcher(hi, te, min_train_pitches = 100, min_test_pitches = 5,
                            verbose = FALSE)$results
  if (nrow(r)) acc[[length(acc) + 1]] <- r
}
P <- add_metrics(bind_rows(acc))
report(P, paste0(season, " MLB, real game-days (single-day test windows)"))

cat("\nInterpretation:\n")
cat("  A metric with low reliability at this window size cannot support a daily\n")
cat("  leaderboard, however sound it is conceptually — the ordering is mostly noise.\n")
cat("  A metric with high R^2 ~ K is largely reporting arsenal size.\n")
