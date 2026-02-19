# ============================================================================
# generate_test_image.R — Preview social media graphic design with mock data
# ----------------------------------------------------------------------------
# Generates sample Predict+ social media images (1200x675 px, 16:9) without
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
# Mock pitcher data — realistic Predict+ score distribution (mean=100, sd=10)
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
  predict_plus    = c(128, 121, 116, 112, 109, 106, 103, 99, 97, 94, 91, 88, 85, 81),
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
  predict_plus    = c(133, 125, 119, 114, 110, 107, 102, 97, 93, 89, 84, 78),
  status          = "evaluated",
  stringsAsFactors = FALSE
)

mock_pitchers <- rbind(mock_starters, mock_relievers)

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
  min_pitches_starter  = 65,   # mock starters have 65–105 test pitches
  min_pitches_reliever = 15,   # mock relievers have 15–35 test pitches
  output_dir           = out_dir,
  top_n                = 5
)

cat("\nTest images written to:", normalizePath(out_dir), "\n")
cat("Files:\n")
cat(paste0("  ", list.files(out_dir, pattern = "\\.png$")), sep = "\n")
