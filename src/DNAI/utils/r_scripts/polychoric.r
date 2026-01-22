# polychoric.R
# Usage: Rscript polychoric.R <RESULTS_DIR>

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("RESULTS_DIR argument missing")
}

RESULTS_DIR <- args[1]

if (!requireNamespace("psych", quietly = TRUE)) {
  stop("Package 'psych' is not installed")
}

library(psych)

in_path <- file.path(RESULTS_DIR, "df_identities.csv")
out_path <- file.path(RESULTS_DIR, "R_poly.csv")

df <- read.csv(in_path, check.names = FALSE)

# keep ordinal/numeric columns only
X <- df[, sapply(df, is.numeric), drop = FALSE]

# drop zero-variance columns
nzv <- sapply(X, function(col) length(unique(na.omit(col))) > 1)
X <- X[, nzv, drop = FALSE]

pc <- psych::polychoric(X, correct = 0)
R_poly <- pc$rho

write.csv(R_poly, out_path, row.names = TRUE)

cat("Polychoric correlation saved to:", out_path, "\n")
