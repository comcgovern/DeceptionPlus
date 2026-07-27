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

cat("\n", if (failures == 0L) "All tests passed.\n" else
    paste0(failures, " test(s) FAILED.\n"), sep = "")
quit(status = if (failures == 0L) 0L else 1L)
