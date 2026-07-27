# ============================================================================
# tests/test_math.R — regression tests for the Deception+ scoring math
# ----------------------------------------------------------------------------
# Run from the repo root:  Rscript tests/test_math.R
#
# These are pure-math tests on synthetic pitch data; they never touch the
# network or the Statcast cache.  They pin down the invariants that a
# surprise-ratio metric has to satisfy, several of which were silently violated
# before:
#   • predicted probabilities land on the class they belong to
#   • every probability row is a real distribution (positive, sums to 1)
#   • surprise stays finite and bounded, so the ratio cannot explode
#   • a genuinely random pitcher outscores a rigidly patterned one
# ============================================================================

suppressPackageStartupMessages(source("pitch_ppi.R"))

failures <- 0L
ok <- function(label, expr) {
  passed <- isTRUE(tryCatch(expr, error = function(e) {
    message("    error: ", conditionMessage(e)); FALSE
  }))
  cat(if (passed) "  PASS  " else "  FAIL  ", label, "\n", sep = "")
  if (!passed) failures <<- failures + 1L
  invisible(passed)
}

# ---------------------------------------------------------------- synthetic --
# Build Statcast-shaped rows for one pitcher.  `mix_fn(balls, strikes)` returns
# the pitch-type distribution for a count, which lets us dial predictability.
make_pitcher <- function(pitcher_id, n, types, mix_fn, seed) {
  set.seed(seed)
  balls   <- sample(0:3, n, TRUE)
  strikes <- sample(0:2, n, TRUE)
  pt <- vapply(seq_len(n), function(i)
    sample(types, 1, prob = mix_fn(balls[i], strikes[i])), character(1))
  data.frame(
    game_date = rep(seq(as.Date("2025-04-01"), by = "day", length.out = 40), length.out = n),
    game_pk = rep(seq_len(40), length.out = n),
    pitcher = pitcher_id, batter = sample(1:60, n, TRUE),
    pitch_type = pt, balls = balls, strikes = strikes,
    outs_when_up = sample(0:2, n, TRUE), inning = sample(1:9, n, TRUE),
    inning_topbot = sample(c("Top","Bot"), n, TRUE),
    on_1b = NA, on_2b = NA, on_3b = NA,
    home_score = 2, away_score = 1,
    stand = sample(c("L","R"), n, TRUE), p_throws = "R",
    description = "ball", zone = sample(1:14, n, TRUE),
    at_bat_number = rep(seq_len(ceiling(n / 4)), each = 4)[seq_len(n)],
    pitch_number = rep(1:4, length.out = n),
    n_thruorder_pitcher = 1L, stringsAsFactors = FALSE
  )
}

# 1  rigid: pitch is fully determined by the count  -> maximally predictable
rigid  <- function(b, s) if (s == 2) c(0, 1) else c(1, 0)
# 2  random: 50/50 regardless of count              -> maximally unpredictable
random <- function(b, s) c(0.5, 0.5)
# 3  one-pitch-plus: 99% cutter, 1% slider          -> the Kenley Jansen shape
onepitch <- function(b, s) c(0.99, 0.01)

raw <- rbind(
  make_pitcher(101, 600, c("FF","SL"), rigid,    11),
  make_pitcher(102, 600, c("FF","SL"), random,   22),
  make_pitcher(103, 600, c("FC","SL"), onepitch, 33),
  # a fourth pitcher widens the league vocabulary, which is what used to leak
  # empty response levels into every other pitcher's model
  make_pitcher(104, 600, c("CH","CU","KC"), function(b, s) c(1/3, 1/3, 1/3), 44)
)

df <- engineer_features(raw, include_batter_metrics = FALSE)
df <- df[!is.na(df$pitcher_id), ]
hist_df <- df[seq(1, nrow(df), by = 2), ]
test_df <- df[seq(2, nrow(df), by = 2), ]

cat("\n--- feature engineering ---\n")
ok("base_state is categorical, not a continuous 0-7 code",
   is.factor(df$base_state))
ok("last_pitch_type uses the canonical pitch_class vocabulary",
   all(unique(df$last_pitch_type) %in% c(unique(df$pitch_class), "NONE")))

res <- evaluate_per_pitcher(hist_df, test_df, min_train_pitches = 50,
                            min_test_pitches = 10, verbose = FALSE)
r <- res$results

cat("\n--- per-pitcher evaluation ---\n")
ok("every pitcher was evaluated", nrow(r) == 4)
ok("surprises are finite", all(is.finite(r$mean_surp_model)) && all(is.finite(r$mean_surp_base)))

# The old eps=1e-9 clamp let a single mis-scored pitch contribute 20.7 nats.
# Nothing honest can exceed log(n_classes) by much once probabilities are
# smoothed toward the pitcher's own mix.
ok("model surprise stays on a sane scale (< 5 nats)", all(r$mean_surp_model < 5))
ok("the one-pitch reliever is not scored as maximally surprising",
   r$mean_surp_model[r$pitcher_id == 103] < 1)

# Each pitcher's model is scoped to their own arsenal, not the league's.
ok("class support is per-pitcher, not league-wide",
   all(r$n_classes[r$pitcher_id %in% c(101, 102, 103)] == 2) &&
     r$n_classes[r$pitcher_id == 104] == 3)

cat("\n--- ordering: does the metric rank what it claims to rank? ---\n")
print(r[, c("pitcher_id","n_test","n_classes","mean_surp_model","mean_surp_base","unpredictability_ratio")],
      digits = 4)
ok("the count-random pitcher outranks the count-rigid pitcher",
   r$unpredictability_ratio[r$pitcher_id == 102] >
     r$unpredictability_ratio[r$pitcher_id == 101])

cat("\n--- a pitch type absent from the pitcher's history ---\n")
# Pitcher 101 unveils a changeup they have never thrown before. This is the
# purest form of the thing Deception+ exists to detect, so it must (a) survive
# rather than be dropped as an unseen factor level, and (b) score as surprising
# without blowing the scale up.
new_pitch <- test_df[test_df$pitcher_id == 101, ][1:20, ]
new_pitch$pitch_class <- "CH"
test_new <- rbind(test_df, new_pitch)
rn <- evaluate_per_pitcher(hist_df, test_new, min_train_pitches = 50,
                           min_test_pitches = 10, verbose = FALSE)$results
row_new <- rn[rn$pitcher_id == 101, ]
row_old <- r[r$pitcher_id == 101, ]
ok("the never-thrown pitch is scored, not silently discarded",
   row_new$n_test == row_old$n_test + 20 && row_new$n_classes == row_old$n_classes + 1)
ok("it raises surprise", row_new$mean_surp_model > row_old$mean_surp_model)
ok("but stays finite and bounded",
   is.finite(row_new$mean_surp_model) && row_new$mean_surp_model < 5)
cat("  surprise ", round(row_old$mean_surp_model, 3), " -> ",
    round(row_new$mean_surp_model, 3), " nats\n", sep = "")

cat("\n--- scale invariance ---\n")
# A ratio built from means must not depend on how many test pitches happen to be
# available.  Score the random pitcher on a half-sized test window.
sub <- test_df[test_df$pitcher_id != 102 |
                 seq_len(nrow(test_df)) %in% which(test_df$pitcher_id == 102)[1:150], ]
r2 <- evaluate_per_pitcher(hist_df, sub, min_train_pitches = 50,
                           min_test_pitches = 10, verbose = FALSE)$results
d <- abs(r2$unpredictability_ratio[r2$pitcher_id == 102] -
           r$unpredictability_ratio[r$pitcher_id == 102])
cat("  ratio shift when the test window halves:", round(d, 4), "\n")
ok("halving the test window does not move the ratio much", d < 0.25)


# ---------------------------------------------------------------------------
# The comparison is only meaningful if the full model can reproduce the
# baseline. Where it cannot, the ratio inverts: it rises with predictability.
# ---------------------------------------------------------------------------
cat("\n--- the full model must nest the baseline ---\n")

# `dep` controls how strongly the pitch is a function of the COUNT — which is
# exactly what the conditional baseline cells on.
sim_count <- function(pid, n, K, dep, seed) {
  make_pitcher(pid, n, c("FF","SL","CH","CU")[1:K], function(b, s) {
    mix <- rep((1 - dep)/K, K); mix[1 + ((b * 3 + s) %% K)] <- (1 - dep)/K + dep; mix
  }, seed)
}
ratios_for <- function(feats, bkeys) {
  deps <- c(0, 0.25, 0.5, 0.75)
  raw2 <- do.call(rbind, lapply(seq_along(deps),
    function(i) sim_count(800 + i, 1500, 4, deps[i], 900 + i)))
  d2 <- engineer_features(raw2, include_batter_metrics = FALSE)
  d2 <- d2[!is.na(d2$pitcher_id), ]
  rr <- suppressWarnings(evaluate_per_pitcher(
    d2[d2$game_pk <= 30, ], d2[d2$game_pk > 30, ],
    min_train_pitches = 50, min_test_pitches = 5,
    feature_names = feats, baseline_keys = bkeys, verbose = FALSE)$results)
  rr$unpredictability_ratio[order(rr$pitcher_id)]
}
# Numeric balls/strikes cannot express a saturated count cross-tab, so the
# "sophisticated" model loses to the "simple" baseline as the pattern sharpens.
old_r <- ratios_for(c("balls","strikes","two_strikes","ahead_in_count","outs",
                      "is_risp","stand","last_pitch_type"),
                    c("balls","strikes","stand","two_strikes"))
new_r <- ratios_for(c("count","outs","is_risp","stand","last_pitch_type"),
                    c("count","stand"))
cat("  numeric balls/strikes: ", paste(sprintf("%.3f", old_r), collapse = "  "), "\n")
cat("  joint `count` factor : ", paste(sprintf("%.3f", new_r), collapse = "  "), "\n")
cat("  (pitchers ordered by increasing count-determinism)\n")
ok("a stronger count pattern no longer inflates the score",
   (max(new_r) - min(new_r)) < (max(old_r) - min(old_r)) / 2)
ok("scores stay near 1 when the pattern is one the baseline also sees",
   all(abs(new_r - 1) < 0.1))

cat("\n--- the nesting guard warns on a non-nesting configuration ---\n")
warn <- tryCatch({
  check_baseline_nesting(df, c("balls","strikes","stand"), c("balls","strikes","stand"))
  NULL
}, warning = function(w) conditionMessage(w))
ok("numeric baseline keys raise a warning",
   !is.null(warn) && grepl("more expressive", warn))
ok("the corrected configuration is silent",
   is.null(tryCatch({
     check_baseline_nesting(df, c("count","stand"), c("count","stand")); NULL
   }, warning = function(w) conditionMessage(w))))

# ---------------------------------------------------------------------------
# The ratio is not invariant to how much data each model is fit on, so the
# calibration in compute_baseline.R has to use production-sized histories.
# ---------------------------------------------------------------------------
cat("\n--- arsenal size must not drive the score ---\n")
raw3 <- do.call(rbind, lapply(2:5, function(K)
  make_pitcher(850 + K, 1400, c("FF","SL","CH","CU","SI")[1:K],
               function(b, s) rep(1/K, K), 400 + K)))
d3 <- engineer_features(raw3, include_batter_metrics = FALSE)
d3 <- d3[!is.na(d3$pitcher_id), ]
r3 <- evaluate_per_pitcher(d3[d3$game_pk <= 30, ], d3[d3$game_pk > 30, ],
                           min_train_pitches = 50, min_test_pitches = 5,
                           verbose = FALSE)$results
cat("  arsenal size : ", paste(r3$n_classes, collapse = "     "), "\n")
cat("  ratio        : ", paste(sprintf("%.3f", r3$unpredictability_ratio), collapse = "  "), "\n")
# All four are equally (maximally) context-independent; any spread is bias.
ok("equally unpredictable pitchers score alike regardless of arsenal size",
   diff(range(r3$unpredictability_ratio)) < 0.05)


# ---------------------------------------------------------------------------
# Two scales. Deception+ (ratio) is arsenal-neutral but needs a season;
# Surprise+ (normalised surprise) survives a single outing.
# ---------------------------------------------------------------------------
cat("\n--- Surprise+ scale ---\n")
ok("normed_surprise is reported and finite",
   "normed_surprise" %in% names(r) && all(is.finite(r$normed_surprise)))
# 102 is the 50/50 coin-flip pitcher: nothing to learn, so essentially all of
# the arsenal's entropy should survive.
ok("a coin-flip pitcher retains ~all of their arsenal's uncertainty",
   abs(r$normed_surprise[r$pitcher_id == 102] - 1) < 0.15)
# 101 is fully determined by the count; 103 throws one pitch 99% of the time.
ok("a count-determined pitcher retains little of it",
   r$normed_surprise[r$pitcher_id == 101] < 0.3)
ok("a one-pitch reliever retains little of it",
   r$normed_surprise[r$pitcher_id == 103] < 0.3)

cat("\n--- Surprise+ is arsenal-neutral among pattern-free pitchers ---\n")
# Same construction as the arsenal test above: equally unpredictable, 2-5 pitches.
r3n <- r3$mean_surp_model / log(pmax(r3$n_classes, 2))
cat("  arsenal size    : ", paste(r3$n_classes, collapse = "     "), "\n")
cat("  raw surprise    : ", paste(sprintf("%.3f", r3$mean_surp_model), collapse = "  "), "\n")
cat("  normed surprise : ", paste(sprintf("%.3f", r3n), collapse = "  "), "\n")
ok("normalising removes most of the arsenal-size spread",
   diff(range(r3n)) < diff(range(r3$mean_surp_model)) / 4)

cat("\n--- reliability: which scale can carry a single outing? ---\n")
# Score the same fixed pitchers over many independent short windows and compare
# between-pitcher variance against window-to-window noise.
rel_windows <- list()
for (w in 1:14) {
  tw <- do.call(rbind, lapply(c(101, 102, 103, 104), function(pid) {
    rows <- which(test_df$pitcher_id == pid)
    test_df[rows[((w - 1) * 20 + 1):(w * 20)], ]
  }))
  tw <- tw[!is.na(tw$pitcher_id), ]
  rw <- evaluate_per_pitcher(hist_df, tw, min_train_pitches = 50,
                             min_test_pitches = 5, verbose = FALSE)$results
  if (nrow(rw)) rel_windows[[length(rel_windows) + 1]] <- rw
}
W <- bind_rows(rel_windows)
reliab <- function(v) {
  m <- W %>% group_by(pitcher_id) %>% summarise(mu = mean(.data[[v]]), .groups = "drop")
  b <- var(m$mu)
  wi <- mean((W[[v]] - m$mu[match(W$pitcher_id, m$pitcher_id)])^2)
  b / (b + wi)
}
rel_s <- reliab("normed_surprise"); rel_d <- reliab("unpredictability_ratio")
cat(sprintf("  Surprise+  reliability on 20-pitch windows: %.3f\n", rel_s))
cat(sprintf("  Deception+ reliability on 20-pitch windows: %.3f\n", rel_d))
ok("Surprise+ is the more reliable of the two on a single outing", rel_s > rel_d)
ok("Surprise+ is usable at that window size", rel_s > 0.7)

cat("\n", if (failures == 0L) "All tests passed.\n" else
    paste0(failures, " test(s) FAILED.\n"), sep = "")
quit(status = if (failures == 0L) 0L else 1L)
