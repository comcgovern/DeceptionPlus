# ============================================================================
# generate_test_image.R — Preview social media graphic design with mock data
# ----------------------------------------------------------------------------
# Generates sample Deception+ social media images (1200x675 px, 16:9) without
# needing to run the full MLB data pipeline.
#
# Usage:
#   Rscript generate_test_image.R
#
# Output:
#   output/test_visuals/  (4 PNG files — starters & relievers, both directions)
# ============================================================================

# Minimal package check
for (pkg in c("ggplot2", "dplyr")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

source("pitch_ppi.R")

# ---------------------------------------------------------------------------
# Mock pitcher data — realistic Deception+ score distribution (mean=100, sd=10)
# ---------------------------------------------------------------------------
set.seed(42)

mock_starters <- data.frame(
  pitcher_id      = 1:14,
  pitcher_name    = c(
    "Gerrit Cole", "Spencer Strider", "Zack Wheeler", "Pablo Lopez",
    "Logan Webb", "Corbin Burnes", "Max Scherzer", "Dylan Cease",
    "Freddy Peralta", "Luis Castillo", "Sandy Alcantara", "Joe Musgrove",
    "Chris Sale", "Nestor Cortes"
  ),
  role            = "starter",
  total_pitches   = sample(800:2500, 14),
  n_pitches_test  = sample(65:105, 14),
  mean_surp_model = runif(14, 0.3, 0.8),
  mean_surp_base  = runif(14, 0.4, 0.7),
  ppi             = runif(14, 0.6, 1.4),
  deception_plus    = c(128, 121, 116, 112, 109, 106, 103, 99, 97, 94, 91, 88, 85, 81),
  status          = "evaluated",
  stringsAsFactors = FALSE
)

mock_relievers <- data.frame(
  pitcher_id      = 101:112,
  pitcher_name    = c(
    "Edwin Diaz", "Emmanuel Clase", "Ryan Helsley", "Josh Hader",
    "Devin Williams", "Felix Bautista", "Jordan Romano", "David Bednar",
    "Alexis Diaz", "Bryan Abreu", "Pete Fairbanks", "Clay Holmes"
  ),
  role            = "reliever",
  total_pitches   = sample(300:900, 12),
  n_pitches_test  = sample(15:35, 12),
  mean_surp_model = runif(12, 0.3, 0.9),
  mean_surp_base  = runif(12, 0.4, 0.7),
  ppi             = runif(12, 0.6, 1.5),
  deception_plus    = c(133, 125, 119, 114, 110, 107, 102, 97, 93, 89, 84, 78),
  status          = "evaluated",
  stringsAsFactors = FALSE
)

mock_pitchers <- rbind(mock_starters, mock_relievers)

# Mock Orioles pitchers — mix of starters, relievers, debuts, and short outings
mock_orioles <- data.frame(
  pitcher_id      = c(201, 202, 203, 204, 205, 206),
  pitcher_name    = c(
    "Corbin Burnes", "Kyle Bradish", "Felix Bautista",
    "Yennier Cano", "Cole Irvin", "Jacob Webb"
  ),
  role            = c("starter", "starter", "reliever", "reliever", "reliever", "reliever"),
  total_pitches   = c(1800, 1200, 700, 400, 300, 50),
  n_pitches_test  = c(92, 78, 18, 22, 11, 4),
  mean_surp_model = c(0.61, 0.55, 0.72, 0.48, 0.53, NA),
  mean_surp_base  = c(0.58, 0.52, 0.60, 0.55, 0.51, NA),
  ppi             = c(0.05, 0.06, 0.20, -0.13, 0.04, NA),
  deception_plus  = c(118, 112, 125, 88, 102, NA),
  status          = c("evaluated", "evaluated", "evaluated",
                      "evaluated", "evaluated", "debut_no_history"),
  stringsAsFactors = FALSE
)

# Wrap in the same structure create_social_media_graphics() expects
mock_res <- list(pitcher_ppi = mock_pitchers)

# ---------------------------------------------------------------------------
# Generate images
# ---------------------------------------------------------------------------
out_dir <- "output/test_visuals"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

create_social_media_graphics(
  res                  = mock_res,
  game_date            = as.character(Sys.Date()),
  min_pitches_starter  = 50,   # starters: >= 50 pitches
  min_pitches_reliever = 10,   # relievers: >= 10 pitches
  output_dir           = out_dir,
  top_n                = 5
)

create_orioles_graphic(
  orioles_data = mock_orioles,
  game_date    = as.character(Sys.Date()),
  output_dir   = out_dir
)

cat("\nTest images written to:", normalizePath(out_dir), "\n")
cat("Files:\n")
cat(paste0("  ", list.files(out_dir, pattern = "\\.png$")), sep = "\n")
